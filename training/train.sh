#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET:?Set DATASET to a Hermes JSONL file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --module swift.cli.sft \
  --use_hf True --model "$MODEL" --model_type qwen3 --train_type full --torch_dtype bfloat16 --attn_impl flash_attn \
  --max_model_len 65536 --rope_scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --dataset "$DATASET" --split_dataset_ratio 0 --dataset_num_proc 16 --streaming False --strict False \
  --remove_unused_columns False --dataloader_num_workers 8 --dataloader_drop_last False \
  --external_plugins "$SCRIPT_DIR/hermes_notice_compat.py" --agent_template hermes --loss_scale hermes --response_prefix "" \
  --packing False --max_length 49152 --truncation_strategy delete --deepspeed zero3 --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --learning_rate 5e-6 --weight_decay 0.05 \
  --lr_scheduler_type cosine --warmup_ratio 0.05 --num_train_epochs 4 --seed 42 --save_strategy epoch \
  --report_to none --logging_first_step True --logging_steps 1 --ddp_backend nccl --output_dir "$OUTPUT_DIR"
