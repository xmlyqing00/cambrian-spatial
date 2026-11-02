#!/usr/bin/env python
"""Step 1 of the VGGT-17 plan: precompute frozen VGGT-1B features for the 128 frames Cambrian-P samples per video.

Anchors = np.linspace(0, N-1, 128, dtype=int), the same frames the lmms-eval loader feeds Cambrian-P.
Per anchor, one 17-frame group [a, a-8s, ..., a-s, a+s, ..., a+8s] (anchor first, so it is VGGT's reference frame),
clamped to the anchor's clip. s = max(round(0.2 s * fps), round(anchor_gap / 17)): 0.2 s on dense videos (VSI-Bench);
on sparse long videos (VSC) the groups tile the whole timeline.
  group_tok   [128,17,2048] f16  last-layer VGGT camera tokens
  group_pose  [128,17,9]    f32  absT_quaR_FoV, in the anchor's camera frame (anchor = identity)
  group_idx   [128,17]           frame indices; group_valid False where a neighbour was clamped at a clip edge
Per clip (whole video, or the segments between --cuts for concatenated videos), one VGGT pass over its anchors:
  scene_tok   [128,2048]    f16  camera tokens        scene_pose [128,9] f32, shared frame = the clip's first anchor
  scene_lookat [128,3]      f32  optical-axis point at the median depth of the central 20% crop
  scene_depth / scene_conf [128,h,w] 14x14-patch means of VGGT depth / depth confidence
Frames are resized so the long side is 518 px (both sides multiples of 14), no crop or padding. On OOM the scene pass
retries at the next --long_sides resolution (default 518 then 448); scene_long_side records which one was used.
--skip_scene writes the group arrays only (enough for vggt_token_source=group, e.g. training videos); on single-clip
videos the scene pass takes about half of the time.
--scene_only adds missing scene geometry to existing group caches atomically, without recomputing or replacing groups.

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=2 python scripts/vggt17/precompute_vggt.py --root ~/.cache/huggingface/cambrians_vsc \
      --glob '10mins/*.mp4' --cuts /data/dataset/vggt17_feats/vsc_cuts.json --out /data/dataset/vggt17_feats/vsc
or both datasets on one GPU shard: scripts/vggt17/run_precompute.sh GPU SHARD NUM_SHARDS
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

N_ANCHORS, HALF, MIN_STRIDE_S = 128, 8, 0.2
G = 2 * HALF + 1
DEV = "cuda"


def plan(n, fps, cuts):
    anchors = np.linspace(0, n - 1, N_ANCHORS, dtype=int)
    stride = max(1, round(MIN_STRIDE_S * fps), round((n - 1) / (N_ANCHORS - 1) / G))
    bounds = np.array([0] + sorted(c for c in cuts if 0 < c < n) + [n])
    seg = np.searchsorted(bounds, anchors, side="right") - 1
    idx = anchors[:, None] + np.array([0] + [k * stride for k in range(-HALF, HALF + 1) if k])
    lo, hi = bounds[seg][:, None], bounds[seg + 1][:, None] - 1
    return anchors, stride, bounds, seg, np.clip(idx, lo, hi), (idx >= lo) & (idx <= hi)


def decode(vr, idx, long_side):
    uniq, inv = np.unique(idx, return_inverse=True)
    x = torch.from_numpy(vr.get_batch(uniq.tolist()).asnumpy()).to(DEV)
    x = x[torch.from_numpy(inv.reshape(-1)).to(DEV)].permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    scale = long_side / max(h, w)
    x = F.interpolate(x, size=(round(h * scale / 14) * 14, round(w * scale / 14) * 14), mode="bicubic",
                      align_corners=False).clamp(0, 1)
    return x.view(*idx.shape, *x.shape[1:])


@torch.no_grad()
def vggt(model, imgs, depth):
    # mirrors VGGT.forward: bf16 aggregator, fp32 heads
    with torch.autocast("cuda", dtype=torch.bfloat16):
        toks, ps_idx = model.aggregator(imgs)
    with torch.autocast("cuda", enabled=False):
        pose = model.camera_head(toks)[-1]
        d = model.depth_head(toks, images=imgs, patch_start_idx=ps_idx, frames_chunk_size=8) if depth else None
    return toks[-1][:, :, 0].float(), pose.float(), d


def lookat(pose, depth, hw):
    ext, _ = pose_encoding_to_extri_intri(pose[None], hw)
    R, t = ext[0, :, :, :3], ext[0, :, :, 3]
    centre = -(R.transpose(1, 2) @ t[..., None])[..., 0]
    H, W = hw
    d = depth[:, int(0.4 * H) : int(0.6 * H), int(0.4 * W) : int(0.6 * W)].flatten(1).median(1).values
    return centre + d[:, None] * R[:, 2, :]


def process(model, path, cuts, group_batch, long_sides, scene_only=False):
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    n, fps = len(vr), float(vr.get_avg_fps())
    anchors, stride, bounds, seg, gidx, gvalid = plan(n, fps, cuts)
    out = dict(n_frames=n, fps=fps, stride=stride, anchors=anchors.astype(np.int32), seg_bounds=bounds.astype(np.int32), seg_id=seg.astype(np.int16), group_idx=gidx.astype(np.int32), group_valid=gvalid)
    if not scene_only:
        out.update(group_tok=np.zeros((N_ANCHORS, G, 2048), np.float16), group_pose=np.zeros((N_ANCHORS, G, 9), np.float32))
        for s in range(0, N_ANCHORS, group_batch):
            imgs = decode(vr, gidx[s : s + group_batch], 518)
            tok, pose, _ = vggt(model, imgs, depth=False)
            out["group_tok"][s : s + group_batch], out["group_pose"][s : s + group_batch] = tok.cpu().numpy(), pose.cpu().numpy()
        out["hw"] = np.array(imgs.shape[-2:], np.int32)
    if not long_sides:  # --skip_scene
        return out

    for long_side in long_sides:
        try:
            sc = dict(scene_tok=np.zeros((N_ANCHORS, 2048), np.float16), scene_pose=np.zeros((N_ANCHORS, 9), np.float32), scene_lookat=np.zeros((N_ANCHORS, 3), np.float32))
            for k in np.unique(seg):
                m = np.where(seg == k)[0]
                imgs = decode(vr, anchors[m][None], long_side)
                tok, pose, (depth, conf) = vggt(model, imgs, depth=True)
                depth, conf, hw = depth[0, ..., 0].float(), conf[0].float(), imgs.shape[-2:]
                pd, pc = F.avg_pool2d(depth[:, None], 14)[:, 0], F.avg_pool2d(conf[:, None], 14)[:, 0]
                if "scene_depth" not in sc:
                    sc["scene_depth"] = np.zeros((N_ANCHORS, *pd.shape[1:]), np.float16)
                    sc["scene_conf"] = np.zeros((N_ANCHORS, *pd.shape[1:]), np.float32)
                sc["scene_tok"][m], sc["scene_pose"][m] = tok[0].cpu().numpy(), pose[0].cpu().numpy()
                sc["scene_lookat"][m] = lookat(pose[0], depth, hw).cpu().numpy()
                sc["scene_depth"][m], sc["scene_conf"][m] = pd.cpu().numpy(), pc.cpu().numpy()
                del imgs, depth, conf
            out.update(sc, scene_long_side=long_side, scene_hw=np.array(hw, np.int32))
            return out
        except torch.cuda.OutOfMemoryError:
            if "imgs" in locals():
                del imgs
            torch.cuda.empty_cache()
    raise RuntimeError(f"scene pass OOM at {list(long_sides)} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="*/*.mp4")
    ap.add_argument("--videos", nargs="+", help="paths relative to --root; overrides --glob")
    ap.add_argument("--video_list", help="file with one path relative to --root per line, processed in file order; overrides --glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", help="JSON {relpath: {'cuts': [first frame of each new clip, ...]}} for concatenated videos")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--group_batch", type=int, default=1, help="17-frame groups per VGGT forward (1 is fastest)")
    ap.add_argument("--long_sides", default="518,448", help="scene-pass resolutions, tried in order on OOM")
    scene_args = ap.add_mutually_exclusive_group()
    scene_args.add_argument("--skip_scene", action="store_true", help="group pass only: no scene_* arrays")
    scene_args.add_argument("--scene_only", action="store_true", help="add shared-frame geometry to existing caches, preserving all group features")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    long_sides = [] if a.skip_scene else [int(s) for s in a.long_sides.split(",")]
    if a.video_list:
        with open(a.video_list) as f:
            vids = [line.strip() for line in f if line.strip()]
    else:
        vids = a.videos or sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, a.glob)))
    i, k = map(int, a.shard.split("/"))
    vids = vids[i::k][: a.limit]
    cuts = json.load(open(a.cuts)) if a.cuts else {}
    model = None
    print(f"{len(vids)} videos -> {a.out}", flush=True)
    t0, done, failed = time.time(), 0, []
    for j, v in enumerate(vids):
        dst = os.path.join(a.out, v[:-4] + ".npz")
        if os.path.exists(dst) and not a.scene_only:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            cached = None
            clip_cuts = cuts.get(v, {}).get("cuts", [])
            if a.scene_only:
                with np.load(dst) as z:
                    if all(key in z.files for key in ("scene_pose", "scene_lookat", "scene_depth", "scene_conf")):
                        continue
                    cached = {key: z[key] for key in z.files}
                if not a.cuts:
                    clip_cuts = cached["seg_bounds"][1:-1].tolist()
            if model is None:
                model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEV).eval()
            out = process(model, os.path.join(root, v), clip_cuts, a.group_batch, long_sides, scene_only=a.scene_only)
            if cached is not None:
                for key in ("n_frames", "anchors", "seg_bounds", "group_idx", "group_valid"):
                    if not np.array_equal(cached[key], out[key]):
                        raise ValueError(f"{v}: new scene pass changed {key}; refusing to replace the cache")
                cached.update(out)
                out = cached
        except Exception as e:
            failed.append(v)
            print(f"[{j + 1}/{len(vids)}] {v}: FAILED {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        out["video"] = v
        with open(dst + ".tmp", "wb") as f:
            np.savez(f, **out)
        os.replace(dst + ".tmp", dst)
        done += 1
        scene = f"scene pass {out['scene_long_side']} px" if "scene_long_side" in out else "no scene pass"
        print(
            f"[{j + 1}/{len(vids)}] {v}: {out['n_frames']} frames @ {out['fps']:.0f} fps, stride {out['stride']}, "
            f"{len(out['seg_bounds']) - 1} clips, {scene}, {time.time() - t:.0f} s, "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB",
            flush=True,
        )
    print(f"done {done}, failed {len(failed)} {failed}, {(time.time() - t0) / 3600:.2f} h", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
