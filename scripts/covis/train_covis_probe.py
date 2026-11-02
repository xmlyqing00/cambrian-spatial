#!/usr/bin/env python
"""M0: do frozen Cambrian-P features already encode "same looked-at place" (co-visibility)?

Inputs: per-video feature files from extract_pose_token_feats.py and DA3 label files from
make_covis_labels.py (matched by file name). Target: symmetric co-visibility
Cs = (covis + covis^T) / 2 over frame pairs; positives are Cs > --thr.

Scores compared (5-fold cross-validation split by video, stratified by source dataset):
  zero-shot  cosine similarity of per-frame features, -|i-j|, pose-angle heuristics
  learned    pairwise probe on a frozen per-frame feature: LN -> Linear(d,256) -> MLP([zi*zj, |zi-zj|])
             pose-geometry MLP on (rotation angle, optical-axis angle, centre distance, pseudo-gaze distance,
             |i-j|) from Cambrian-P's predicted poses, and from DA3 poses as an upper bound for pose-only cues
Metrics (held-out videos): AUROC over all pairs and over far pairs (|i-j| >= --far_gap, i.e. revisits),
AP on far pairs, Spearman with Cs, and revisit retrieval P@1 (for each frame with a far positive,
is its top-scored far frame co-visible?).

Feature files without camera-token keys (base Cambrian-S models, extracted with --no_camera_tokens)
are supported: camera-token and predicted-pose scores are then skipped.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def pose_enc_to_c2w(pe):
    """Cambrian-P / VGGT absT_quaR_FoV pose encoding [N,9] -> c2w [N,4,4]."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    ext, _ = pose_encoding_to_extri_intri(torch.as_tensor(pe).float()[None], image_size_hw=(384, 384))
    w2c = torch.eye(4, dtype=torch.float64).repeat(ext.shape[1], 1, 1)
    w2c[:, :3, :] = ext[0].double()
    return torch.linalg.inv(w2c).float()


def pair_geometry(c2w):
    """Per-pair pose features [N,N,5] from c2w [N,4,4] (scale-normalised per video)."""
    N = len(c2w)
    R, c = c2w[:, :3, :3], c2w[:, :3, 3]
    scale = (c - c.mean(0)).norm(dim=1).mean().clamp(min=1e-6)
    rel = torch.einsum("iba,jbc->ijac", R, R)  # R_i^T R_j
    rot = torch.arccos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) / np.pi
    ax = R[:, :, 2]
    axis = torch.arccos((ax @ ax.T).clamp(-1, 1)) / np.pi
    dist = torch.cdist(c, c) / scale
    g = c + scale * ax  # pseudo gaze point one "scene radius" along the optical axis
    gdist = torch.cdist(g, g) / scale
    idx = torch.arange(N, dtype=torch.float32)
    gap = (idx[:, None] - idx[None]).abs() / N
    return torch.stack([rot, axis, dist, gdist, gap], -1)


def load_data(feat_dir, label_dir):
    data = []
    for fp in sorted(p for p in glob.glob(os.path.join(feat_dir, "*.npz")) if ".tmp." not in p):
        lp = os.path.join(label_dir, os.path.basename(fp))
        if not os.path.exists(lp):
            continue
        F, L = np.load(fp), np.load(lp)
        if not (F["frame_idx"] == L["frame_idx"]).all():
            print("skip (frame mismatch)", fp)
            continue
        C = torch.from_numpy(L["covis"].astype(np.float32))
        d = dict(
            key=os.path.basename(fp)[:-4],
            ds=os.path.basename(fp).split("__")[0],
            layers=F["layers"].tolist(),
            img_h=torch.from_numpy(F["img_h"]),
            Cs=((C + C.T) / 2).clamp(0, 1),
            geo_da3=pair_geometry(torch.from_numpy(L["c2w"]).float()),
        )
        # camera-token features and predicted poses exist only for checkpoints with a pose head
        for k in ("cam_h", "cam_proj", "cam_trunk"):
            if k in F.files:
                d[k] = torch.from_numpy(F[k])
        if "pose_enc" in F.files:
            d["geo_pred"] = pair_geometry(pose_enc_to_c2w(F["pose_enc"]))
        gz = torch.from_numpy(L["gaze"]).float()
        cc = torch.from_numpy(L["c2w"][:, :3, 3]).float()
        d["gaze_da3"] = torch.cdist(gz, gz) / (cc - cc.mean(0)).norm(dim=1).mean().clamp(min=1e-6)
        data.append(d)
    return data


def frame_feature(d, name):
    if "@" in name:
        kind, layer = name.split("@")
        return d[kind][d["layers"].index(int(layer))]
    return d[name]


class PairProbe(nn.Module):
    def __init__(self, dim, hid=256, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim, elementwise_affine=False), nn.Linear(dim, hid), nn.Dropout(drop))
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x, iu):  # x [B,N,D] -> logits [B,P]
        z = self.proj(x.float())
        zi, zj = z[:, iu[0]], z[:, iu[1]]
        return self.head(torch.cat([zi * zj, (zi - zj).abs()], -1))[..., 0]


class GeoProbe(nn.Module):
    def __init__(self, dim=5, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, g, iu):  # g [B,N,N,5]
        return self.net(g[:, iu[0], iu[1]])[..., 0]


def metrics(scores, data, iu, thr, far_gap):
    gap = (iu[1] - iu[0]).numpy()
    far = gap >= far_gap
    S, Y, Ycont, Far = [], [], [], []
    hits, n_q = 0, 0
    N = data[0]["Cs"].shape[0]
    for s, d in zip(scores, data):
        cs = d["Cs"][iu[0], iu[1]].numpy()
        S.append(s), Y.append(cs > thr), Ycont.append(cs), Far.append(far)
        M = np.full((N, N), -np.inf)
        M[iu[0], iu[1]] = s
        M[iu[1], iu[0]] = s
        pos = d["Cs"].numpy() > thr
        idx = np.arange(N)
        farmask = np.abs(idx[:, None] - idx[None]) >= far_gap
        for i in range(N):
            if (pos[i] & farmask[i]).any():
                cand = np.where(farmask[i])[0]
                hits += pos[i, cand[np.argmax(M[i, cand])]]
                n_q += 1
    S, Y, Ycont, Far = map(np.concatenate, (S, Y, Ycont, Far))
    return dict(
        auroc_all=roc_auc_score(Y, S),
        auroc_far=roc_auc_score(Y[Far], S[Far]),
        ap_far=average_precision_score(Y[Far], S[Far]),
        base_ap_far=float(Y[Far].mean()),
        spearman=spearmanr(S, Ycont).correlation,
        revisit_p1=hits / max(n_q, 1),
    )


def cosine_scores(x, iu):
    x = x.float()
    x = x - x.mean(0)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-6)
    return (x @ x.T)[iu[0], iu[1]].numpy()


def train_probe(kind, name, train, test, iu, args, dev):
    torch.manual_seed(0)
    if kind == "geo":
        model = GeoProbe().to(dev)
        get = lambda ds: torch.stack([d[name] for d in ds]).to(dev)
    else:
        model = PairProbe(frame_feature(train[0], name).shape[-1], args.hid).to(dev)
        get = lambda ds: torch.stack([frame_feature(d, name) for d in ds]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    iu_d = iu.to(dev)
    X_all = get(train)
    Y_all = torch.stack([d["Cs"][iu[0], iu[1]] for d in train]).to(dev)
    g = torch.Generator().manual_seed(0)
    model.train()
    for step in range(args.steps):
        b = torch.randint(0, len(train), (args.batch,), generator=g)
        loss = Fn.binary_cross_entropy_with_logits(model(X_all[b], iu_d), Y_all[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        X_te = get(test)
        return [model(X_te[k : k + 1], iu_d)[0].float().cpu().numpy() for k in range(len(test))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--far_gap", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--probe_layers", default="0,8,16,20,24,28")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    data = load_data(args.feat_dir, args.label_dir)
    N = data[0]["Cs"].shape[0]
    iu = torch.triu_indices(N, N, 1)
    by_ds = {}
    for d in data:
        by_ds.setdefault(d["ds"], []).append(d)
    fold_of = {}
    for k, d in enumerate(sorted(data, key=lambda d: (d["ds"], d["key"]))):
        fold_of[d["key"]] = k % args.folds
    print(f"{len(data)} videos {dict((k, len(v)) for k, v in by_ds.items())}, load {time.time()-t0:.0f}s", flush=True)

    layers = [int(x) for x in args.probe_layers.split(",")]
    has_cam = all("cam_h" in d and "geo_pred" in d for d in data)
    gap_score = -(iu[1] - iu[0]).float().numpy()
    zero_shot = {
        "-abs(i-j)": lambda d: gap_score,
        "-rot(pred pose)": lambda d: -d["geo_pred"][iu[0], iu[1], 0].numpy(),
        "-axis(pred pose)": lambda d: -d["geo_pred"][iu[0], iu[1], 1].numpy(),
        "-axis(DA3 pose)": lambda d: -d["geo_da3"][iu[0], iu[1], 1].numpy(),
        "-gaze dist(DA3 depth)": lambda d: -d["gaze_da3"][iu[0], iu[1]].numpy(),
    }
    if not has_cam:
        del zero_shot["-rot(pred pose)"], zero_shot["-axis(pred pose)"]
    for l in layers:
        zero_shot[f"cos img_h@{l}"] = (lambda l: lambda d: cosine_scores(frame_feature(d, f"img_h@{l}"), iu))(l)
        if l > 0 and has_cam:
            zero_shot[f"cos cam_h@{l}"] = (lambda l: lambda d: cosine_scores(frame_feature(d, f"cam_h@{l}"), iu))(l)
    if has_cam:
        zero_shot["cos cam_trunk"] = lambda d: cosine_scores(d["cam_trunk"], iu)

    learned = [("geo", "geo_pred"), ("geo", "geo_da3")] if has_cam else [("geo", "geo_da3")]
    if has_cam:
        learned += [("feat", f"cam_h@{l}") for l in layers if l > 0]
    learned += [("feat", f"img_h@{l}") for l in layers]
    if has_cam:
        learned += [("feat", "cam_proj"), ("feat", "cam_trunk")]

    results = {}
    for name, fn in zero_shot.items():
        results["zs: " + name] = metrics([fn(d) for d in data], data, iu, args.thr, args.far_gap)
        print(f"{'zs: ' + name:28s} " + " ".join(f"{k}={v:.3f}" for k, v in results['zs: ' + name].items()), flush=True)
    for kind, name in learned:
        per_fold = []
        for f in range(args.folds):
            tr = [d for d in data if fold_of[d["key"]] != f]
            te = [d for d in data if fold_of[d["key"]] == f]
            if not tr or not te:
                continue
            per_fold.append(metrics(train_probe(kind, name, tr, te, iu, args, dev), te, iu, args.thr, args.far_gap))
        tag = ("probe: " if kind == "feat" else "geo-mlp: ") + name
        results[tag] = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
        results[tag].update({k + "_std": float(np.std([m[k] for m in per_fold])) for k in ("auroc_far", "revisit_p1")})
        print(f"{tag:28s} " + " ".join(f"{k}={v:.3f}" for k, v in results[tag].items() if not k.endswith("_std")), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(args=vars(args), n_videos=len(data), results=results), open(os.path.join(args.out_dir, "m0_results.json"), "w"), indent=1)
    cols = ["auroc_all", "auroc_far", "ap_far", "base_ap_far", "spearman", "revisit_p1"]
    with open(os.path.join(args.out_dir, "m0_results.md"), "w") as fh:
        fh.write(f"M0 co-visibility probe: {len(data)} videos, thr={args.thr}, far_gap={args.far_gap}, {args.folds}-fold CV by video\n\n")
        fh.write("| score | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for k, v in results.items():
            fh.write(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |\n")
    print(f"done in {time.time()-t0:.0f}s -> {args.out_dir}/m0_results.md")


if __name__ == "__main__":
    main()
