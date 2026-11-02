"""Per-dim mean/std of the VGGT tokens fed to the projector (its fixed input standardization).

python scripts/vggt17/feature_stats.py --feat_dir /data/dataset/vggt17_feats/train --out /data/dataset/vggt17_feats/stats_train_group.npz
"""

import argparse
import glob
import json
import os

import numpy as np

from cambrianp.model.multimodal_projector.vggt_token import VGGT_TOKEN_SOURCES, load_vggt_tokens, vggt_feature_path

ap = argparse.ArgumentParser()
ap.add_argument("--feat_dir", required=True, help="searched recursively for *.npz")
ap.add_argument("--source", default="group", choices=VGGT_TOKEN_SOURCES)
ap.add_argument("--out", required=True)
ap.add_argument("--jsonl", help="use only this training manifest's videos, excluding validation/test from normalization")
args = ap.parse_args()

if args.jsonl:
    with open(args.jsonl) as f:
        rows = (json.loads(line) for line in f if line.strip())
        videos = {d["video"] for d in rows if "video" in d}
    files = sorted(vggt_feature_path(args.feat_dir, v) for v in videos)
else:
    files = sorted(glob.glob(os.path.join(args.feat_dir, "**", "*.npz"), recursive=True))
if not files:
    raise SystemExit(f"no npz under {args.feat_dir}")
total = total_sq = None
n = 0
for f in files:
    x = load_vggt_tokens(f, args.source).astype(np.float64)
    total = x.sum(0) if total is None else total + x.sum(0)
    total_sq = (x * x).sum(0) if total_sq is None else total_sq + (x * x).sum(0)
    n += len(x)
mean = total / n
std = np.sqrt(np.maximum(total_sq / n - mean * mean, 0))
np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32), n_tokens=n, n_videos=len(files), source=args.source, feat_dir=args.feat_dir, jsonl=args.jsonl or "")
print(f"{args.out}: {len(files)} videos, {n} tokens, dim {len(mean)}; |mean| median {np.median(np.abs(mean)):.3f}, " f"std median {np.median(std):.3f} (min {std.min():.4f}, max {std.max():.3f})")
