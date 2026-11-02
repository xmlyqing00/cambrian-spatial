#!/usr/bin/env python
"""M1': training-free place-aware frame selection for evaluation.

Phase 1 (GPU, skipped with --select_only): for each video, decode --num_candidates uniformly spaced
candidate frames and cache the model's own per-frame visual embedding (vision tower -> projector ->
8x8 pooling -> mean over the 64 tokens, i.e. the LLM-input feature "img_h@0" of the M0 probe).

Phase 2 (CPU): for each frame budget K write two index files into --out_dir:
  uniform_k{K}.json     np.linspace(0, total-1, K), identical to the default eval sampling (control)
  place_u{u}_k{K}.json  a uniform skeleton of round(u*K) candidates, completed by greedy
                        facility-location coverage max_S sum_i max_{j in S} s_ij with
                        s = max(0, cos) of per-video mean-centred features, so briefly seen places
                        get frames while repeated views of one place share them; temporal order.
Files map "<dir>/<video file>" -> frame indices and are passed to lmms-eval as
  --model_args ...,max_frames_num=K,frame_index_file=<file>

Run in the `cambrianp` env with PYTHONPATH=$PWD/lmms-eval:$PWD.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image


def facility_location(feat, k, uniform_frac):
    """Indices (into feat) of k frames: uniform skeleton + greedy coverage, sorted."""
    m = feat.shape[0]
    if k >= m:
        return np.arange(m)
    x = feat.float()
    x = Fn.normalize(x - x.mean(0), dim=1)
    sim = (x @ x.T).clamp(min=0)
    n_uni = int(round(k * uniform_frac))
    sel = np.unique(np.linspace(0, m - 1, n_uni, dtype=int)).tolist() if n_uni > 0 else []
    chosen = torch.zeros(m, dtype=torch.bool)
    chosen[sel] = True
    cover = sim[:, sel].max(1).values if sel else torch.zeros(m)
    while len(sel) < k:
        gain = (sim - cover[:, None]).clamp(min=0).sum(0)
        gain[chosen] = -1
        j = int(gain.argmax())
        sel.append(j)
        chosen[j] = True
        cover = torch.maximum(cover, sim[:, j])
    return np.sort(np.array(sel))


def coverage(feat, idx):
    x = Fn.normalize(feat.float() - feat.float().mean(0), dim=1)
    return float((x @ x[idx].T).clamp(min=0).max(1).values.mean())


@torch.inference_mode()
def candidate_features(lm, path, cand, chunk=64):
    from decord import VideoReader, cpu
    from lmms_eval.models.simple.cambrian_p import expand2square

    model, proc = lm.model, lm._image_processor
    bg = tuple(int(x * 255) for x in proc.image_mean)
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    out = []
    for s in range(0, len(cand), chunk):
        frames = vr.get_batch(cand[s : s + chunk].tolist()).asnumpy()
        vr.seek(0)
        pil = [expand2square(Image.fromarray(f, mode="RGB"), bg) for f in frames]
        pix = proc.preprocess(pil, return_tensors="pt")["pixel_values"].half().cuda()
        tok = model.get_2dPool(
            model.encode_images(pix), stride=model.config.mm_spatial_pool_stride,
            target_side_length=model.config.mm_img_tok_num,
        )
        out.append(tok.float().mean(1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def cache_features(args, vids):
    from decord import VideoReader, cpu
    from lmms_eval.models.simple.cambrian_p import CambrianP

    lm = CambrianP(
        pretrained=os.path.abspath(args.ckpt), conv_template="qwen_1_5", model_name="llava_qwen",
        max_frames_num=128, use_camera_tokens=True, camera_tokens_mode="camera_tokens",
        camera_tokens_place="append_to_frame", query_mode="query_after_image",
    )
    lm._config.mm_spatial_pool_stride = lm.mm_spatial_pool_stride
    lm._config.mm_spatial_pool_mode = lm.mm_spatial_pool_mode
    si, sn = map(int, args.shard.split("/"))
    todo = vids[si::sn]
    for k, rel in enumerate(todo):
        dst = os.path.join(args.feat_cache_dir, rel.replace("/", "__").rsplit(".", 1)[0] + ".npz")
        if os.path.exists(dst):
            continue
        path = os.path.join(args.video_root, rel)
        t0 = time.time()
        vr = VideoReader(path, ctx=cpu(0))
        total, fps = len(vr), float(vr.get_avg_fps())
        del vr
        cand = np.unique(np.linspace(0, total - 1, min(total, args.num_candidates), dtype=int))
        try:
            feat = candidate_features(lm, path, cand)
        except Exception as e:  # report and continue; the eval falls back to uniform for missing videos
            print(f"[{k+1}/{len(todo)}] FAIL {rel}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        tmp = dst + ".tmp.npz"
        np.savez(tmp, video=rel, total=np.int64(total), fps=np.float32(fps), cand=cand.astype(np.int64), feat=feat)
        os.replace(tmp, dst)
        print(f"[{k+1}/{len(todo)}] {rel} total={total} cand={len(cand)} {time.time()-t0:.1f}s", flush=True)


def select(args, vids):
    budgets = [int(x) for x in args.budgets.split(",")]
    fracs = [float(x) for x in args.uniform_frac.split(",")]
    files = {f"uniform_k{k}": {} for k in budgets}
    files.update({f"place_u{u:g}_k{k}": {} for u in fracs for k in budgets})
    stats = {name: [] for name in files}
    missing = []
    for rel in vids:
        fp = os.path.join(args.feat_cache_dir, rel.replace("/", "__").rsplit(".", 1)[0] + ".npz")
        if not os.path.exists(fp):
            missing.append(rel)
            continue
        z = np.load(fp)
        total, cand, fps = int(z["total"]), z["cand"], float(z["fps"])
        feat = torch.from_numpy(z["feat"])
        for k in budgets:
            uni = np.linspace(0, total - 1, k, dtype=int)
            uni_c = np.abs(cand[None] - uni[:, None]).argmin(1)  # nearest candidates, for coverage only
            files[f"uniform_k{k}"][rel] = uni.tolist()
            stats[f"uniform_k{k}"].append((coverage(feat, uni_c), 1.0, np.diff(uni).max() / fps))
            for u in fracs:
                idx = cand[facility_location(feat, k, u)]
                assert len(idx) == k, (rel, k, len(idx))
                files[f"place_u{u:g}_k{k}"][rel] = idx.tolist()
                same = len(set(idx.tolist()) & set(uni.tolist())) / k
                cov = coverage(feat, np.searchsorted(cand, idx))
                stats[f"place_u{u:g}_k{k}"].append((cov, same, np.diff(idx).max() / fps))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(vids) - len(missing)} videos selected, {len(missing)} without cached features {missing[:3]}")
    print(f"{'file':22s} coverage  shared-with-uniform  max-gap(s)")
    for name, d in files.items():
        json.dump(d, open(os.path.join(args.out_dir, name + ".json"), "w"))
        s = np.array(stats[name])
        print(f"{name:22s} {s[:, 0].mean():.3f}     {s[:, 1].mean():.2f}                 {s[:, 2].mean():.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_list", required=True, help="text file: one video path per line, relative to --video_root")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--feat_cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpts/Cambrian-P-7B")
    ap.add_argument("--num_candidates", type=int, default=512)
    ap.add_argument("--budgets", default="128,32")
    ap.add_argument("--uniform_frac", default="0.5,0", help="comma-separated skeleton fractions, one place file each")
    ap.add_argument("--shard", default="0/1", help="i/n for phase 1")
    ap.add_argument("--select_only", action="store_true", help="skip phase 1 (use cached features only)")
    args = ap.parse_args()

    vids = [l.strip() for l in open(args.video_list) if l.strip()]
    os.makedirs(args.feat_cache_dir, exist_ok=True)
    if not args.select_only:
        cache_features(args, vids)
    if args.shard == "0/1" or args.select_only:
        select(args, vids)


if __name__ == "__main__":
    main()
