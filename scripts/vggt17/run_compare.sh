#!/usr/bin/env bash
# VGGT-17 vs matched control, end to end after the training-feature precompute (run_precompute_train.sh):
#   1. wait for WAIT_PIDS (the precompute shards) to exit; resume the precompute if videos of SET have no features yet
#   2. if some still have none (videos the precompute fails on), rebuild SET from the videos that have them
#   3. stats_train_group.npz (the projector's input standardization) from all training features
#   4. LoRA-train the VGGT arm on GPU_VGGT and the control on GPU_CTRL in parallel (same set and seed; a failed run is
#      retried once, resuming from its last checkpoint)
#   5. full VSI-Bench eval of each adapter on its GPU, then the base model and the VGGT adapter without VGGT tokens
#      (drop-at-test) on whichever GPU is free first
# UPLOAD_POD=pod: after step 3, upload the set's data to the pod (upload_remote.py, md5-verified; log $RUN_ROOT/upload.log).
# TRAIN=0: stop after step 3 (and the upload).
# Finished stages are skipped, so a rerun resumes.
# Usage (repo root): [WAIT_PIDS="pid ..."] [GPU_VGGT=0] [GPU_CTRL=2] [FEATS=/data/dataset/vggt17_feats] [SET=jsonl]
#   [RUN_ROOT=output/vggt17/train] [STATS=$FEATS/stats_train_group.npz] [OUT_ROOT=output/vggt17/evals] [LIMIT=all]
#   [UPLOAD_POD=] [TRAIN=1] [PY=...] [CKPT=...] [DATA_PATH=...] bash scripts/vggt17/run_compare.sh
#   log: $RUN_ROOT/compare.log; evals in $OUT_ROOT/<tag>
set -u
cd "$(dirname "$0")/../.."
unset LORA VGGT VGGT_DIR VGGT_STATS VGGT_SOURCE
export PY=${PY:-/home/lz1bhc/miniforge3/envs/cambrianp/bin/python} PYTHONPATH=$PWD OUT_ROOT=${OUT_ROOT:-output/vggt17/evals}
FEATS=${FEATS:-/data/dataset/vggt17_feats}
SET=${SET:-$FEATS/train_manifest/headroom_pack4_10000.jsonl}
STATS=${STATS:-$FEATS/stats_train_group.npz}
RUN_ROOT=${RUN_ROOT:-output/vggt17/train}
GPU_VGGT=${GPU_VGGT:-0} GPU_CTRL=${GPU_CTRL:-2} LIMIT=${LIMIT:-all}
mkdir -p "$RUN_ROOT"
exec >>"$RUN_ROOT/compare.log" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

for p in ${WAIT_PIDS:-}; do
    log "waiting for PID $p to exit"
    while [ -e "/proc/$p" ]; do sleep 300; done
done
missing() {
    "$PY" - "$SET" "$FEATS/train" <<'EOF'
import json, os, sys
s, d = sys.argv[1:]
vids = {json.loads(l)["video"] for l in open(s)}
print(sum(not os.path.exists(os.path.join(d, os.path.splitext(v)[0] + ".npz")) for v in vids))
EOF
}
n=$(missing) || { log "cannot read $SET"; exit 1; }
if [ "$n" != 0 ]; then
    # an interrupted precompute resumes (finished videos are skipped); shards 0/2 and 1/2 of 500 = the first 1000
    # manifest videos, from which build_packed_set.py draws
    log "$n videos of $SET have no VGGT features: running the precompute for them"
    bash scripts/vggt17/run_precompute_train.sh "$GPU_VGGT" 0 2 500 >>output/vggt17/logs/pre_train_s0.log 2>&1 &
    bash scripts/vggt17/run_precompute_train.sh "$GPU_CTRL" 1 2 500 >>output/vggt17/logs/pre_train_s1.log 2>&1 &
    wait
    n=$(missing) || exit 1
fi
if [ "$n" != 0 ]; then
    log "$n videos of $SET still have no VGGT features (precompute failures): rebuilding it from the videos that have them"
    qas=$("$PY" -c "import json, sys; print(sum(len(json.loads(l)['conversations']) // 2 for l in open(sys.argv[1])))" "$SET")
    SET=$("$PY" scripts/vggt17/build_packed_set.py --feat_dir "$FEATS/train" --num_qas "$qas" --tag feats \
        --out_dir "$(dirname "$SET")" | tail -1)
    [ -f "$SET" ] || { log "rebuild failed"; exit 1; }
fi
if [ ! -f "$STATS" ]; then
    log "VGGT token stats -> $STATS"
    "$PY" scripts/vggt17/feature_stats.py --feat_dir "$FEATS/train" --out "$STATS" || { log "feature_stats failed"; exit 1; }
fi
if [ -n "${UPLOAD_POD:-}" ]; then
    log "uploading the data of $SET to $UPLOAD_POD (log $RUN_ROOT/upload.log)"
    "$PY" scripts/vggt17/upload_remote.py --pod "$UPLOAD_POD" --set "$SET" --feats "$FEATS" \
        --videos "${DATA_PATH:-$HOME/datasets/VSI-590K}" --md5 >>"$RUN_ROOT/upload.log" 2>&1 || { log "upload failed"; exit 1; }
    log "upload OK"
fi
if [ "${TRAIN:-1}" = 0 ]; then
    log "TRAIN=0: set $SET ready, no training"
    exit 0
fi
name=$(basename "$SET" .jsonl)
OUT_V=$RUN_ROOT/vggt_$name OUT_C=$RUN_ROOT/ctrl_$name
CLAIMS=$RUN_ROOT/.eval_claims_$name  # mkdir: atomic claim of an eval either GPU may run
rm -rf "$CLAIMS" && mkdir -p "$CLAIMS"
log "set $SET; VGGT arm $OUT_V (GPU $GPU_VGGT), control $OUT_C (GPU $GPU_CTRL)"

train() {  # VGGT OUT GPU
    for attempt in 1 2; do
        [ -f "$2/adapter_model.bin" ] && return 0
        log "train VGGT=$1 -> $2 on GPU $3 (attempt $attempt)"
        VGGT=$1 VGGT_DIR=$FEATS/train VGGT_STATS=$STATS bash scripts/vggt17/run_train.sh "$3" "$SET" "$2" >/dev/null 2>&1
        log "train VGGT=$1 exit $? (log $2/train.log)"
    done
    [ -f "$2/adapter_model.bin" ]
}

ev() {  # TAG GPU PORT [VAR=value ...]
    local tag=$1 gpu=$2 port=$3
    shift 3
    find "$OUT_ROOT/$tag" -name results.json 2>/dev/null | grep -q . && return 0
    log "eval $tag on GPU $gpu"
    log "$(env "$@" bash scripts/vggt17/run_eval.sh "$gpu" "$port" "$tag" "$LIMIT")"
}

lane() {  # VGGT OUT GPU PORT TAG [eval VAR=value ...]
    local vggt=$1 out=$2 gpu=$3 port=$4 tag=$5
    shift 5
    train "$vggt" "$out" "$gpu" || { log "training $out failed"; return 1; }
    ev "$tag" "$gpu" "$port" LORA="$out" "$@"
    if mkdir "$CLAIMS/base" 2>/dev/null; then
        ev base_full "$gpu" "$port"
    elif [ -f "$OUT_V/adapter_model.bin" ] && mkdir "$CLAIMS/drop" 2>/dev/null; then
        ev "${name}_vggt_drop" "$gpu" "$port" LORA="$OUT_V"
    fi
}

lane 1 "$OUT_V" "$GPU_VGGT" 29711 "${name}_vggt" VGGT_DIR="$FEATS/vsibench" &
lane 0 "$OUT_C" "$GPU_CTRL" 29712 "${name}_ctrl" &
wait
log "all done"
