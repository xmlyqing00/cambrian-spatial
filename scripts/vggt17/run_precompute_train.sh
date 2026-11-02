#!/usr/bin/env bash
# VGGT-17 group features for the training-manifest videos (VSI-590K scans, no VSI-Bench scenes), in manifest order
# (a seeded shuffle, so the finished videos are an unbiased sample). Resumable: finished videos are skipped.
# Usage (repo root): PY=/path/to/env/python scripts/vggt17/run_precompute_train.sh GPU SHARD NUM_SHARDS [LIMIT]
#   e.g. shards 0/2 and 1/2 with LIMIT 500 cover the first 1000 manifest videos.
set -u
GPU=$1 I=$2 N=$3 LIMIT=${4:-}
PY=${PY:-python}
OUT=${OUT:-/data/dataset/vggt17_feats}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" scripts/vggt17/precompute_vggt.py --root ~/datasets/VSI-590K --video_list "$OUT/train_manifest/videos.txt" \
    --out "$OUT/train" --skip_scene --shard "$I/$N" ${LIMIT:+--limit "$LIMIT"}
