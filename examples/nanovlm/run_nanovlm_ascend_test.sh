#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

DATASET_PATH="${DATASET_PATH:-/home/ma-user/work/dataset/lmms_engine_test_nanovlm/nanovlm_test.yaml}"
PROCESSOR_NAME="${PROCESSOR_NAME:-/home/ma-user/work/model/NanoVLM_Init}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/model/NanoVLM_Init}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-8000}"
DDP_BACKEND="${DDP_BACKEND:-hccl}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

RUN_NAME="${RUN_NAME:-nanovlm_ascend_test}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ma-user/work/output/nanovlm_ascend_test}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m lmms_engine.launch.cli \
  trainer_type=fsdp2_trainer \
  dataset_config.dataset_path="${DATASET_PATH}" \
  dataset_config.dataset_format=yaml \
  dataset_config.dataset_type=qwen3_vl_iterable \
  dataset_config.processor_config.processor_type=nanovlm \
  dataset_config.processor_config.processor_name="${PROCESSOR_NAME}" \
  dataset_config.packing=false \
  dataset_config.packing_strategy=first_fit \
  dataset_config.packing_length=8192 \
  dataset_config.filter_overlong=true \
  dataset_config.video_backend=qwen_vl_utils \
  dataset_config.video_sampling_strategy=fps \
  dataset_config.video_max_pixels=50176 \
  dataset_config.video_max_frames=512 \
  model_config.load_from_pretrained_path="${MODEL_PATH}" \
  model_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  trainer_args.freeze_modules='["vision_model"]' \
  trainer_args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  trainer_args.learning_rate="${LEARNING_RATE}" \
  trainer_args.weight_decay="${WEIGHT_DECAY}" \
  trainer_args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  trainer_args.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  trainer_args.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  trainer_args.warmup_ratio="${WARMUP_RATIO}" \
  trainer_args.run_name="${RUN_NAME}" \
  trainer_args.output_dir="${OUTPUT_DIR}" \
  trainer_args.ddp_backend="${DDP_BACKEND}" \
  trainer_args.fsdp2=true \
  trainer_args.max_steps="${MAX_STEPS}" \
  trainer_args.save_steps="${MAX_STEPS}" \
  trainer_args.save_total_limit=1 \
  trainer_args.fsdp_config.transformer_layer_cls_to_wrap='["Qwen3DecoderLayer"]' \
  trainer_args.fsdp_config.reshard_after_forward=true \
  trainer_args.sp_ulysses_degree=1 \
  trainer_args.use_rmpad=false \
  trainer_args.dataloader_num_workers=0 \
  trainer_args.dataloader_prefetch_factor=null \
  trainer_args.dataloader_pin_memory=false \
  trainer_args.print_batch_input_steps=1 \
  trainer_args.bf16=true \
  trainer_args.lr_scheduler_type=cosine \
  trainer_args.logging_steps=1 \
  trainer_args.report_to='[]' \
  trainer_args.group_by_length=false
