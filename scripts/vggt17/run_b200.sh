#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/data/lz1bhc}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PY=${PY:-$BASE/venvs/cambrianp-b200/bin/python}
export PATH="$(dirname "$PY"):$PATH"
export DATA_PATH="$BASE/data/VSI-590K"
export HF_HOME="$BASE/data/huggingface"
export VSI_BENCH_ARROW="$HF_HOME/datasets/nyu-visionx___vsi-bench/full/0.0.0/3a6a1cf9ac54c403509b15260897cc6ddc41b408/vsi-bench-test.arrow"
export CKPT="$BASE/models/Cambrian-P-7B"
export VGGT_DIR="$BASE/data/vggt17_feats/train"
export VGGT_STATS="$BASE/data/vggt17_feats/stats_scene10pct_s42.npz"
export VGGT_SOURCE=group TASK=vsibench
MANIFEST="$BASE/data/vggt17_feats/train_manifest/scene10pct_s42"
EVAL_FEATURES="$BASE/data/vggt17_feats/vsibench"
RUN="$BASE/runs/scene10pct_s42"
export OUT_ROOT="$RUN/evals"
export PYTHONPATH="$REPO/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="$BASE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$BASE/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export BS=1 GRAD_ACC=4 WORKERS=4 MEM_CAP_GB=128 VT_CHUNK=32
export LR=1e-4 PROJ_LR=1e-4 LORA_R=64 LORA_ALPHA=128 EPOCHS=1
export FRAMES=128 SEED=42 SAVE_STEPS=250 PORT=29600
export TRAIN_PY=cambrianp/train/train.py
unset EVAL_JSON LORA GATHER_SCOPE
cd "$REPO"
mkdir -p "$RUN/logs"

features() {
    local root=$1 list=$2 out=$3 tag=$4 p0 p1 status=0
    CUDA_VISIBLE_DEVICES=0 "$PY" -B scripts/vggt17/precompute_vggt.py \
        --root "$root" --video_list "$list" --out "$out" --long_sides 448 --group_batch 1 --shard 0/2 \
        >"$RUN/logs/${tag}_gpu0.log" 2>&1 &
    p0=$!
    CUDA_VISIBLE_DEVICES=1 "$PY" -B scripts/vggt17/precompute_vggt.py \
        --root "$root" --video_list "$list" --out "$out" --long_sides 448 --group_batch 1 --shard 1/2 \
        >"$RUN/logs/${tag}_gpu1.log" 2>&1 &
    p1=$!
    printf 'Preprocessing both GPUs. Logs: %s/logs/%s_gpu{0,1}.log\n' "$RUN" "$tag"
    wait "$p0" || status=1
    wait "$p1" || status=1
    [ "$status" -eq 0 ] || return "$status"
    CUDA_VISIBLE_DEVICES= "$PY" -B - "$out" "$list" <<'PY'
from pathlib import Path
import sys
from cambrianp.model.multimodal_projector.vggt_token import check_vggt_features, vggt_feature_path

root, listing = sys.argv[1:]
videos = Path(listing).read_text().splitlines()
for video in videos:
    check_vggt_features(vggt_feature_path(root, video), "group", "lookat", 128)
print(f"Verified complete group features and shared geometry for {len(videos)} videos")
PY
}

statistics() {
    CUDA_VISIBLE_DEVICES= "$PY" -B scripts/vggt17/feature_stats.py \
        --feat_dir "$VGGT_DIR" --source group --jsonl "$1" --out "$2"
}

train_arm() {
    local tag=$1 vggt=$2 mode=$3 scope=$4
    # Legacy Transformers resumes its own trusted optimizer/RNG files with pickle.
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        VGGT="$vggt" GATHER_MODE="$mode" GATHER_SCOPE="$scope" MAX_STEPS=-1 \
        bash scripts/vggt17/run_train.sh 0,1 "$MANIFEST/train.jsonl" "$RUN/$tag"
}

check_eval() {
    "$PY" -B - "$OUT_ROOT/$1" "$MANIFEST/stats.json" <<'PY'
import json
import math
from pathlib import Path
import sys

root, metadata = map(Path, sys.argv[1:])
expected = json.loads(metadata.read_text())["evaluation"]["qas"]
results = list(root.rglob("results*.json"))
assert results, f"No results written under {root}"
assert any("vsibench" in json.loads(path.read_text()).get("results", {}) for path in results), "Missing VSI-Bench metrics"
sample_files = list(root.rglob("samples_vsibench_*.jsonl"))
assert len(sample_files) == 1, f"Expected one fresh sample log, found {sample_files}"
rows = [json.loads(line) for line in sample_files[0].open() if line.strip()]
assert len(rows) == expected, (len(rows), expected)
assert len({row["doc_id"] for row in rows}) == expected, "Duplicated evaluation questions"
def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
empty = [row["doc_id"] for row in rows if not any(text.strip() for text in strings(row.get("filtered_resps", row.get("resps"))))]
assert not empty, f"{len(empty)} empty/failed generations, e.g. {empty[:5]}"
metrics = [json.loads(path.read_text()).get("results", {}).get("vsibench", {}) for path in results]
score_count = 0
while metrics:
    metric = metrics.pop()
    if isinstance(metric, dict):
        metrics.extend(metric.values())
    elif isinstance(metric, list):
        metrics.extend(metric)
    elif isinstance(metric, (int, float)):
        assert math.isfinite(metric), f"Nonfinite metric under {root}"
        score_count += 1
assert score_count, f"No numeric VSI-Bench metrics under {root}"
print(f"VERIFIED {root.name}: {expected} unique QAs, no empty generations")
PY
}

evaluate() {
    local tag=$1 gpu=$2 port=$3
    [ ! -e "$OUT_ROOT/$tag" ] || { echo "Refusing to mix evaluation logs; output already exists: $OUT_ROOT/$tag" >&2; return 2; }
    if [ "$tag" = base ]; then
        env -u VGGT_DIR -u VGGT_STATS -u LORA -u GATHER_SCOPE \
            HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            bash scripts/vggt17/run_eval.sh "$gpu" "$port" "$tag" all
    elif [ "$tag" = cam ]; then
        env -u VGGT_DIR -u VGGT_STATS -u GATHER_SCOPE \
            LORA="$RUN/cam" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            bash scripts/vggt17/run_eval.sh "$gpu" "$port" "$tag" all
    else
        env -u VGGT_STATS LORA="$RUN/$tag" VGGT_DIR="$EVAL_FEATURES" GATHER_SCOPE="$tag" \
            HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            bash scripts/vggt17/run_eval.sh "$gpu" "$port" "$tag" all
    fi
    check_eval "$tag"
}

case "${1:-}" in
    setup)
        bash scripts/vggt17/setup_b200.sh "$(dirname "$(dirname "$PY")")"
        ;;
    models)
        "$PY" -B - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download("nyu-visionx/Cambrian-P-7B", local_dir=os.environ["CKPT"])
snapshot_download("google/siglip-so400m-patch14-384")
snapshot_download("facebook/VGGT-1B")
PY
        ;;
    subset)
        printf 'cac72cd0c3cbddab60cc6aad4846d98613d64df76f4777f393142ed792e34fcf  %s\n' "$VSI_BENCH_ARROW" | sha256sum --check -
        CUDA_VISIBLE_DEVICES= "$PY" -B scripts/vggt17/prepare_scene_subset.py \
            --jsonl "$DATA_PATH/vsi_590k.jsonl" --root "$DATA_PATH" --hf_home "$HF_HOME" \
            --out "$MANIFEST" --fraction 0.1 --seed 42
        ;;
    smoke-features)
        features "$DATA_PATH" "$MANIFEST/smoke_train_videos.txt" "$VGGT_DIR" smoke
        statistics "$MANIFEST/smoke_train.jsonl" "$RUN/stats_smoke.npz"
        ;;
    smoke)
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TRAIN_PY=scripts/vggt17/train_memprobe.py \
            VGGT=1 GATHER_MODE=lookat GATHER_SCOPE=all VGGT_STATS="$RUN/stats_smoke.npz" \
            MAX_STEPS=20 SAVE_STEPS=20 WORKERS=0 \
            bash scripts/vggt17/run_train.sh 0,1 "$MANIFEST/smoke_train.jsonl" "$RUN/smoke"
        CUDA_VISIBLE_DEVICES=0 "$PY" -B scripts/vggt17/test_lora_roundtrip.py \
            --lora "$RUN/smoke" --ckpt "$CKPT" --vggt_feat_dir "$VGGT_DIR" \
            --smoke_jsonl "$MANIFEST/smoke_train.jsonl" --data_path "$DATA_PATH" \
            >"$RUN/smoke/reload.log" 2>&1
        tail -n 15 "$RUN/smoke/reload.log"
        ;;
    features)
        features "$DATA_PATH" "$MANIFEST/train_videos.txt" "$VGGT_DIR" train
        features "$HF_HOME/vsibench" "$MANIFEST/eval_videos.txt" "$EVAL_FEATURES" eval
        ;;
    stats)
        statistics "$MANIFEST/train.jsonl" "$VGGT_STATS"
        ;;
    train-cam) train_arm cam 0 none self ;;
    train-self) train_arm self 1 lookat self ;;
    train-all) train_arm all 1 lookat all ;;
    eval-base) evaluate base 0 29611 ;;
    eval-cam) evaluate cam 1 29612 ;;
    eval-self) evaluate self 0 29613 ;;
    eval-all) evaluate all 1 29614 ;;
    check-evals)
        for tag in base cam self all; do check_eval "$tag"; done
        ;;
    *)
        echo "Usage: bash scripts/vggt17/run_b200.sh {setup|models|subset|smoke-features|smoke|features|stats|train-cam|train-self|train-all|eval-base|eval-cam|eval-self|eval-all|check-evals}" >&2
        exit 2
        ;;
esac
