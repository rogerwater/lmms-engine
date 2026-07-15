#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"

export TOKENIZERS_PARALLELISM=false
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

DATASET_PATH="${DATASET_PATH:-/home/ma-user/work/dataset/Data_NanoVLM/stage1_llava_pretrain.yaml}"
PROCESSOR_NAME="${PROCESSOR_NAME:-/home/ma-user/work/model/NanoVLM_Init}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/model/NanoVLM_Init}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-8000}"
DDP_BACKEND="${DDP_BACKEND:-hccl}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"

LEARNING_RATE="${LEARNING_RATE:-1.0e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-2180}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

PACKING_LENGTH="${PACKING_LENGTH:-51200}"
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-false}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
PRINT_BATCH_INPUT_STEPS="${PRINT_BATCH_INPUT_STEPS:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
ENABLE_PROFILER="${ENABLE_PROFILER:-false}"
PROFILER_START_STEP="${PROFILER_START_STEP:-10}"
PROFILER_WARMUP_STEPS="${PROFILER_WARMUP_STEPS:-1}"
PROFILER_END_STEP="${PROFILER_END_STEP:-13}"
PROFILER_RANKS="${PROFILER_RANKS:-[0]}"

RUN_NAME="${RUN_NAME:-nanovlm_stage1_ascend}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ma-user/work/output/nanovlm_stage1_ascend}"

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
  dataset_config.packing_length="${PACKING_LENGTH}" \
  dataset_config.filter_overlong=true \
  dataset_config.video_backend=qwen_vl_utils \
  dataset_config.video_sampling_strategy=fps \
  dataset_config.video_max_pixels=50176 \
  dataset_config.video_max_frames=512 \
  model_config.load_from_pretrained_path="${MODEL_PATH}" \
  model_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  trainer_args.freeze_modules='["vision_model","language_model"]' \
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
  trainer_args.save_steps="${SAVE_STEPS}" \
  trainer_args.save_total_limit="${SAVE_TOTAL_LIMIT}" \
  trainer_args.fsdp_config.transformer_layer_cls_to_wrap='["Qwen3DecoderLayer"]' \
  trainer_args.fsdp_config.reshard_after_forward="${RESHARD_AFTER_FORWARD}" \
  trainer_args.sp_ulysses_degree=1 \
  trainer_args.use_liger_kernel="${USE_LIGER_KERNEL}" \
  trainer_args.use_rmpad=false \
  trainer_args.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  trainer_args.dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
  trainer_args.dataloader_pin_memory="${DATALOADER_PIN_MEMORY}" \
  trainer_args.dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
  trainer_args.print_batch_input_steps="${PRINT_BATCH_INPUT_STEPS}" \
  trainer_args.bf16=true \
  trainer_args.lr_scheduler_type=cosine \
  trainer_args.logging_steps="${LOGGING_STEPS}" \
  trainer_args.enable_profiler="${ENABLE_PROFILER}" \
  trainer_args.profiler_config.start_step="${PROFILER_START_STEP}" \
  trainer_args.profiler_config.warmup_steps="${PROFILER_WARMUP_STEPS}" \
  trainer_args.profiler_config.end_step="${PROFILER_END_STEP}" \
  trainer_args.profiler_config.ranks="${PROFILER_RANKS}" \
  trainer_args.report_to='[]' \
  trainer_args.group_by_length=false
