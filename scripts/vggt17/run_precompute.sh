#!/usr/bin/env bash
# Precompute VGGT-17 features for VSC 10-min, then VSI-Bench, on one GPU shard (resumable: finished videos are skipped).
# Usage (repo root): PY=/path/to/env/python scripts/vggt17/run_precompute.sh GPU SHARD NUM_SHARDS
set -u
GPU=$1 I=$2 N=$3
PY=${PY:-python}
OUT=${OUT:-/data/dataset/vggt17_feats}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" scripts/vggt17/precompute_vggt.py --root ~/.cache/huggingface/cambrians_vsc --glob '10mins/*.mp4' \
    --cuts "$OUT/vsc_cuts.json" --out "$OUT/vsc" --shard "$I/$N"
"$PY" scripts/vggt17/precompute_vggt.py --root ~/.cache/huggingface/vsibench --glob '*/*.mp4' \
    --out "$OUT/vsibench" --shard "$I/$N"
