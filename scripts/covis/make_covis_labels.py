#!/usr/bin/env python
"""Pseudo-geometry and co-visibility ("same looked-at place") labels from Depth Anything 3.

For each video, decode exactly the frames Cambrian-P sees at evaluation time
(np.linspace over all frames, see lmms_eval/models/simple/cambrian_p.py:load_video),
run DA3 any-view inference jointly on them, and save one .npz per video with:

  frame_idx [N]        decoded frame indices
  c2w       [N,4,4]    camera-to-world, expressed in frame 0's camera coordinates
  K         [N,3,3]    intrinsics at DA3 processing resolution (proc_hw)
  depth_lr  [N,h,w]    fp16 depth, downsampled by --depth_stride
  covis     [N,N]      covis[i, j] = fraction of frame i's confident surface points that are
                       inside frame j, in front of it, and depth-consistent with frame j
  gaze      [N,3]      3D point hit by the optical-axis ray of each frame (frame-0 coords)
  med_depth [N]        median depth per frame

Run in the `depth-anything` env, e.g.
  python scripts/covis/make_covis_labels.py --video_list list.txt --video_root ROOT --out_dir OUT
"""
import argparse
import os
import time

import numpy as np
import torch
from decord import VideoReader, cpu


def decode_uniform(path, num_frames):
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(idx.tolist()).asnumpy()
    return frames, idx, total, float(vr.get_avg_fps())


@torch.no_grad()
def covisibility(depth, conf, w2c, K, stride=4, conf_pct=20.0, rel_tol=0.15):
    """Asymmetric co-visibility matrix from per-frame depth, poses (w2c) and intrinsics."""
    N, H, W = depth.shape
    dev = depth.device
    c2w = torch.linalg.inv(w2c)
    vs, us = torch.meshgrid(
        torch.arange(stride // 2, H, stride, device=dev),
        torch.arange(stride // 2, W, stride, device=dev),
        indexing="ij",
    )
    vs, us = vs.reshape(-1), us.reshape(-1)
    d = depth[:, vs, us]  # [N,P]
    c = conf[:, vs, us]
    thr = torch.quantile(c.flatten().float()[:: max(1, c.numel() // 1_000_000)], conf_pct / 100.0)
    valid = (d > 0) & torch.isfinite(d) & (c >= thr)
    pix = torch.stack([us.float(), vs.float(), torch.ones_like(us, dtype=torch.float)], 0)  # [3,P]
    Xw = c2w[:, :3, :3] @ ((torch.linalg.inv(K) @ pix) * d[:, None]) + c2w[:, :3, 3:]  # [N,3,P]
    n_valid = valid.sum(1).clamp(min=1).float()
    covis = torch.zeros(N, N, device=dev)
    for j in range(N):
        Xj = w2c[j, :3, :3] @ Xw + w2c[j, :3, 3:]
        z = Xj[:, 2]
        uvw = K[j] @ Xj
        u = uvw[:, 0] / z.clamp(min=1e-6)
        v = uvw[:, 1] / z.clamp(min=1e-6)
        inb = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        Dj = depth[j][v.round().long().clamp(0, H - 1), u.round().long().clamp(0, W - 1)]
        vis = valid & inb & (Dj > 0) & ((z - Dj).abs() <= rel_tol * Dj)
        covis[:, j] = vis.sum(1).float() / n_valid
    return covis


def process_video(model, path, args, device):
    t0 = time.time()
    frames, idx, total, fps = decode_uniform(path, args.num_frames)
    t_dec = time.time() - t0
    pred = model.inference(list(frames), process_res=args.process_res)
    t_da3 = time.time() - t0 - t_dec

    depth = torch.from_numpy(pred.depth).float().to(device)
    conf = torch.from_numpy(pred.conf).float().to(device) if pred.conf is not None else torch.ones_like(depth)
    ext = torch.from_numpy(pred.extrinsics).float().to(device)
    if ext.shape[-2] == 3:  # [N,3,4] -> [N,4,4]
        bottom = torch.tensor([0, 0, 0, 1.0], device=device).expand(ext.shape[0], 1, 4)
        ext = torch.cat([ext, bottom], 1)
    K = torch.from_numpy(pred.intrinsics).float().to(device)
    covis = covisibility(depth, conf, ext, K, stride=args.covis_stride, rel_tol=args.rel_tol)

    c2w = torch.linalg.inv(ext)
    c2w = torch.linalg.inv(c2w[0]) @ c2w  # frame-0 camera coordinates, like Cambrian-P's pose targets
    N, H, W = depth.shape
    cy, cx = H // 2, W // 2
    d_c = depth[:, cy - 2 : cy + 3, cx - 2 : cx + 3].flatten(1).median(1).values  # optical-axis depth
    ray = torch.linalg.inv(K) @ torch.tensor([cx, cy, 1.0], device=device)  # [N,3]
    gaze = (c2w[:, :3, :3] @ (ray * d_c[:, None])[..., None])[..., 0] + c2w[:, :3, 3]
    s = args.depth_stride
    out = dict(
        frame_idx=idx.astype(np.int32),
        total_frames=np.int32(total),
        fps=np.float32(fps),
        proc_hw=np.array([H, W], np.int32),
        c2w=c2w.cpu().numpy().astype(np.float32),
        K=K.cpu().numpy().astype(np.float32),
        depth_lr=depth[:, s // 2 :: s, s // 2 :: s].cpu().numpy().astype(np.float16),
        conf_lr=conf[:, s // 2 :: s, s // 2 :: s].cpu().numpy().astype(np.float16),
        covis=covis.cpu().numpy().astype(np.float16),
        gaze=gaze.cpu().numpy().astype(np.float32),
        med_depth=depth.flatten(1).median(1).values.cpu().numpy().astype(np.float32),
        is_metric=np.int32(getattr(pred, "is_metric", 0) or 0),
    )
    return out, dict(decode_s=t_dec, da3_s=t_da3, total_s=time.time() - t0, n=N, hw=(H, W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_list", required=True, help="text file: one video path per line (relative to --video_root)")
    ap.add_argument("--video_root", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="depth-anything/DA3-GIANT-1.1")
    ap.add_argument("--num_frames", type=int, default=128)
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--covis_stride", type=int, default=4)
    ap.add_argument("--depth_stride", type=int, default=4)
    ap.add_argument("--rel_tol", type=float, default=0.15)
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()

    vids = [l.strip() for l in open(args.video_list) if l.strip()]
    si, sn = map(int, args.shard.split("/"))
    vids = vids[si::sn]
    if args.limit:
        vids = vids[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    for k, rel in enumerate(vids):
        key = rel.replace("/", "__").rsplit(".", 1)[0]
        dst = os.path.join(args.out_dir, key + ".npz")
        if os.path.exists(dst):
            continue
        try:
            out, info = process_video(model, os.path.join(args.video_root, rel), args, device)
        except Exception as e:  # keep going on corrupt videos, but report them
            print(f"[{k+1}/{len(vids)}] FAIL {rel}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        tmp = dst + ".tmp.npz"
        np.savez_compressed(tmp, video=rel, **out)
        os.replace(tmp, dst)
        cv = out["covis"].astype(np.float32)
        adj = np.mean([cv[i, i + 1] for i in range(len(cv) - 1)])
        print(
            f"[{k+1}/{len(vids)}] {rel} n={info['n']} hw={info['hw']} dec={info['decode_s']:.1f}s "
            f"da3={info['da3_s']:.1f}s tot={info['total_s']:.1f}s adj_covis={adj:.2f} "
            f"mean_covis={cv.mean():.2f} mem={torch.cuda.max_memory_allocated()/2**30:.1f}G",
            flush=True,
        )


if __name__ == "__main__":
    main()
