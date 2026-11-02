#!/usr/bin/env bash
# LoRA fine-tune of Cambrian-P-7B on VSI-590K VQA. VGGT=1 adds the VGGT-17 token per frame; VGGT=0 is the matched control
# (same seed, data order and LoRA init: only the VGGT token and its projector differ). Inputs match the lmms-eval wrapper
# (scripts/vggt17/test_train_data.py): 128 uniform frames padded to square, a camera token per frame, no pose loss.
# Usage (repo root):
#   [VGGT=0|1] [VGGT_DIR=/data/dataset/vggt17_feats/train] [VGGT_STATS=$VGGT_DIR/../stats_train_group.npz] [PROJ_LR=$LR]
#   [LR=1e-4] [EPOCHS=1] [MAX_STEPS=-1] [BS=1] [GRAD_ACC=8] [SAVE_STEPS=250] [LORA_R=64] [LORA_ALPHA=128] [FRAMES=128]
#   [SEED=42] [WORKERS=4] [PORT=29600] [MEM_CAP_GB=32] [VT_CHUNK=32] [PY=python] [TRAIN_PY=cambrianp/train/train.py]
#   [CKPT=ckpts/Cambrian-P-7B] [DATA_PATH=~/datasets/VSI-590K (root of the jsonl's video paths)]
#   [GATHER_MODE=none|lookat] [GATHER_RADIUS=0.15] [GATHER_SCOPE=all|near|far|self] [FAR_GAP=32]
#   [EVAL_JSON=scene_disjoint_val.jsonl] [EVAL_STEPS=250]
#   scripts/vggt17/run_train.sh GPUS JSONL OUT_DIR [extra train.py args]
# GPUS: comma list, e.g. 0 or 0,2: one process per GPU, plain DDP (DeepSpeed bf16 would cast the fp32 projector to bf16).
# MEM_CAP_GB: per-process CUDA allocator cap (0 = none); on shared GPUs this job then OOMs instead of its neighbours.
#   32 fits 128 frames with VT_CHUNK=32 (peak ~24.5 GiB allocated, ~25 GiB reserved).
# PROJ_LR: peak LR of the VGGT projector, default LR. Its gradient is clipped on its own (vggt_projector_clip_separately),
#   so it cannot rescale the LoRA gradients, which then match the control arm's.
# Resumes from the last complete checkpoint in OUT_DIR. Log: OUT_DIR/train.log.
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,16p "$0"; exit 1; }
GPUS=$1 JSONL=$2 OUT=$3
shift 3
PY=${PY:-python}
VGGT=${VGGT:-0}
GATHER_MODE=${GATHER_MODE:-none}
if [ "$GATHER_MODE" != none ] && [ "$VGGT" != 1 ]; then
    echo "GATHER_MODE=$GATHER_MODE requires VGGT=1" >&2
    exit 1
fi
VGGT_DIR=${VGGT_DIR:-/data/dataset/vggt17_feats/train}
VGGT_STATS=${VGGT_STATS:-$(dirname "$VGGT_DIR")/stats_train_group.npz}
NPROC=$(tr ',' '\n' <<<"$GPUS" | wc -l)
if [ "$NPROC" -gt 1 ]; then
    LAUNCH=("$PY" -m torch.distributed.run --nproc_per_node "$NPROC" --master_port "${PORT:-29600}")
    DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-True}
else
    LAUNCH=("$PY")
    DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-False}
fi

ARGS=(
    --model_name_or_path "${CKPT:-ckpts/Cambrian-P-7B}" --version qwen_1_5
    --data_path "${DATA_PATH:-$HOME/datasets/VSI-590K}" --image_folder "${DATA_PATH:-$HOME/datasets/VSI-590K}"
    --json_file "$JSONL" --lazy_preprocess True
    --vision_tower google/siglip-so400m-patch14-384 --mm_vision_select_layer -2 --mm_projector_type mlp2x_gelu
    --mm_use_im_start_end False --mm_use_im_patch_token False
    --image_aspect_ratio anyres_max_9 --image_grid_pinpoints "${IMAGE_GRID_PINPOINTS:-(1x1)-(6x6)}"
    --mm_patch_merge_type spatial_unpad --mm_newline_position grid
    --mm_img_tok_num 64 --mm_spatial_pool_stride 2 --mm_spatial_pool_mode bilinear
    --frames_upbound "${FRAMES:-128}" --force_sample True --video_pad_to_square True --video_camera_tokens True
    --load_rec_model True --token_num 2 --camera_tokens_mode camera_tokens --camera_tokens_place append_to_frame
    --query_mode query_after_image --query_loss_frame_idx 0
    --enable_camera True --enable_depth False --enable_point False --enable_rec_pose_viz False
    --rec_projector_type mlp1x_gelu --rec_embed_dim 2048 --num_intermediate_layers 4 --rec_model_path None
    --rec_camera_trunk_depth 4 --rec_camera_causal_attn False --rec_dpt_head_type dpt_head --rec_depth_head_patch_start_idx 1
    --rec_loss_weight 0.2 --rec_camera_loss_weight 5.0
    --lora_enable True --lora_llm_only True --lora_r "${LORA_R:-64}" --lora_alpha "${LORA_ALPHA:-128}" --lora_dropout 0.05
    --lm_loss_labels_only True
    --bf16 True --tf32 True --gradient_checkpointing True --attn_implementation flash_attention_2 --model_max_length 32768
    --per_device_train_batch_size "${BS:-1}" --gradient_accumulation_steps "${GRAD_ACC:-8}"
    --learning_rate "${LR:-1e-4}" --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine
    --num_train_epochs "${EPOCHS:-1}" --max_steps "${MAX_STEPS:--1}"
    --save_strategy steps --save_steps "${SAVE_STEPS:-250}" --save_total_limit 2
    --logging_steps 1 --report_to none --dataloader_num_workers "${WORKERS:-4}" --seed "${SEED:-42}"
    --ddp_find_unused_parameters "$DDP_FIND_UNUSED" --max_gpu_mem_gb "${MEM_CAP_GB:-32}" --vision_tower_chunk_size "${VT_CHUNK:-32}"
    --output_dir "$OUT"
)
if [ "$NPROC" -gt 1 ]; then
    # Image-only batches do not use the VGGT projector; reentrant checkpointing cannot safely detect unused DDP parameters.
    ARGS+=(--gradient_checkpointing_use_reentrant False)
fi
if [ "$VGGT" = 1 ]; then
    ARGS+=(--use_vggt_token True --vggt_feat_dir "$VGGT_DIR" --vggt_token_source group --vggt_token_stats "$VGGT_STATS"
        --vggt_gather_mode "$GATHER_MODE" --vggt_gather_radius "${GATHER_RADIUS:-0.15}"
        --vggt_gather_scope "${GATHER_SCOPE:-all}" --vggt_gather_far_gap "${FAR_GAP:-32}"
        --lora_modules_to_save vggt_projector --vggt_projector_lr "${PROJ_LR:-${LR:-1e-4}}")
fi
if [ -n "${EVAL_JSON:-}" ]; then
    ARGS+=(--eval_json_file "$EVAL_JSON" --evaluation_strategy steps --eval_steps "${EVAL_STEPS:-250}"
        --per_device_eval_batch_size 1 --prediction_loss_only True)
fi

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
TRAIN_PY=${TRAIN_PY:-cambrianp/train/train.py}
echo "[$(date '+%F %T')] GPUs $GPUS, VGGT=$VGGT: ${LAUNCH[*]} $TRAIN_PY ${ARGS[*]} $*" >>"$OUT/train.log"
"${LAUNCH[@]}" "$TRAIN_PY" "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
