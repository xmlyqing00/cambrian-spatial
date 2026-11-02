#!/usr/bin/env bash
# VSI-Bench / VSC eval for VGGT-17 runs. Seeded doc order (LMMS_EVAL_SHUFFLE_DOCS=1), so --limit N takes the same N questions in every run.
# Usage: [TASK=vsibench] [VGGT_DIR=/data/dataset/vggt17_feats/vsibench] [VGGT_SOURCE=group] [VGGT_STATS=stats.npz] [LORA=output_dir]
#        [OUT_ROOT=output/vggt17/evals] [CKPT=ckpts/Cambrian-P-7B] [PY=<env>/bin/python] bash scripts/vggt17/run_eval.sh GPU PORT TAG [LIMIT|all]
# A look-at LORA carries its gather settings; [GATHER_SCOPE=self|near|far|all] selects a same-token-budget ablation.
# No VGGT_DIR = no VGGT token (baseline, or the drop-at-test ablation of a VGGT-trained LORA).
# Videos are read from $HF_HOME/vsibench (or $HF_HOME/cambrians_vsc); relative CKPT/LORA paths are relative to the repo root.
set -u
cd "$(dirname "$0")/../.."
gpu=$1 port=$2 tag=$3 lim=${4:-200}
task=${TASK:-vsibench} out=${OUT_ROOT:-output/vggt17/evals}
PY=${PY:-/home/lz1bhc/miniforge3/envs/cambrianp/bin/python}
case $PY in */*) PATH=$(dirname "$PY"):$PATH ;; esac  # accelerate of the same env
extra=""
[ -n "${VGGT_DIR:-}" ] && extra+=",vggt_feat_dir=${VGGT_DIR},vggt_token_source=${VGGT_SOURCE:-group}"
[ -n "${VGGT_STATS:-}" ] && extra+=",vggt_token_stats=${VGGT_STATS}"
[ -n "${LORA:-}" ] && extra+=",lora_path=$(realpath "$LORA")"
[ -n "${GATHER_SCOPE:-}" ] && extra+=",vggt_gather_scope=${GATHER_SCOPE}"
limit_arg=(--limit "$lim")
[ "$lim" = all ] && limit_arg=()
mkdir -p "$out" output/vggt17/logs
CUDA_VISIBLE_DEVICES=$gpu LMMS_EVAL_LAUNCHER=accelerate LMMS_EVAL_SHUFFLE_DOCS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD/lmms-eval:$PWD \
accelerate launch --num_processes=1 --main_process_port "$port" -m lmms_eval --model cambrian_p \
  --model_args "pretrained=${CKPT:-$PWD/ckpts/Cambrian-P-7B},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,use_camera_tokens=true,camera_tokens_mode=camera_tokens,camera_tokens_place=append_to_frame,query_mode=query_after_image$extra" \
  --tasks "$task" --batch_size 1 --log_samples --log_samples_suffix "$tag" \
  --output_path "$out/$tag" "${limit_arg[@]}" > "output/vggt17/logs/eval_$tag.log" 2>&1
rc=$?
# lmms-eval exits 0 even on errors (e.g. OOM); success = results.json written
if find "$out/$tag" -name results.json 2>/dev/null | grep -q .; then echo "EVAL $tag OK (exit $rc)"; else echo "EVAL $tag FAILED (exit $rc, no results.json)"; fi
