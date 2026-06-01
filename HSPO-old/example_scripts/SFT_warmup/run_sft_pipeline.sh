#!/usr/bin/env bash
# ============================================================
# HSPO SFT 4-Step Warm-up Pipeline for ALFWorld
# ============================================================
#
# Runs the staged SFT curriculum from the HSPO implementation plan (§5.2):
#   Step 1: Format SFT     → 学 Plan-Execute 输出格式
#   Step 2: Executor SFT   → 给定 subgoal，学 action
#   Step 3: Planner SFT    → 学 SWITCH 边界 + subgoal 决策
#   Step 4: Light joint SFT → 小 lr 协调 planner/executor 接口
#
# Each step uses the previous step's checkpoint as the starting model.
# Step 1 starts from the base Qwen2.5-Instruct model.
#
# Prerequisites:
#   conda activate verl
#   source config/paths.sh
#
# Step 0 — Data preparation (once):
#   # 0a. Collect expert trajectories with stratified sampling
#   python3 example_scripts/data_preprocess/collect_alfworld_sft_demos.py \
#       --max_per_task 250 \
#       --out_dir $HSPO_SFT_RAW_DIR
#
#   # 0b. Segment & build HSPO SFT data (JSONL + parquet for all 4 splits)
#   python3 example_scripts/data_preprocess/segment_and_build_sft.py \
#       --raw_dir $HSPO_SFT_RAW_DIR \
#       --output_dir $HSPO_SFT_DATA_DIR
#
# Usage:
#   bash run_sft_pipeline.sh 0.5b           # All 4 steps for Qwen2.5-0.5B
#   bash run_sft_pipeline.sh 1.5b           # All 4 steps for Qwen2.5-1.5B
#   bash run_sft_pipeline.sh 7b             # All 4 steps for Qwen2.5-7B
#   bash run_sft_pipeline.sh 0.5b 2         # Resume from step 2
# ============================================================
set -e

# ── Locate HSPO-agent root & source path config ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSPO_AGENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$HSPO_AGENT_ROOT/config/paths.sh"

MODEL_SIZE=${1:-0.5b}
START_STEP=${2:-1}

# ── Model & data paths ────────────────────────────────────────────────────────
BASE_MODEL=$(hspo_model_path "$MODEL_SIZE")
if [ -z "$BASE_MODEL" ]; then
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use 0.5b, 1.5b, or 7b."
    exit 1
fi

case "$MODEL_SIZE" in
    0.5b)
        NPROC=1; MICRO_BATCH=1; TRAIN_BATCH=64
        LR_FORMAT=2e-5; LR_EXECUTOR=2e-5; LR_PLANNER=2e-5; LR_JOINT=5e-6 ;;
    1.5b)
        NPROC=1; MICRO_BATCH=2; TRAIN_BATCH=64
        LR_FORMAT=2e-5; LR_EXECUTOR=2e-5; LR_PLANNER=2e-5; LR_JOINT=5e-6 ;;
    7b)
        NPROC=1; MICRO_BATCH=1; TRAIN_BATCH=32
        LR_FORMAT=1e-5; LR_EXECUTOR=1e-5; LR_PLANNER=1e-5; LR_JOINT=3e-6 ;;
esac

DATA_DIR="$HSPO_SFT_DATA_DIR"
CKPT_DIR=$(hspo_sft_ckpt_dir "$MODEL_SIZE")

# Check data exists
for split in format executor planner all; do
    if [ ! -f "$DATA_DIR/train_${split}.parquet" ]; then
        echo "ERROR: $DATA_DIR/train_${split}.parquet not found."
        echo "Run segment_and_build_sft.py first."
        exit 1
    fi
done

# ── Helper: run one SFT step ──────────────────────────────────────────────────
run_sft_step() {
    local STEP_NAME="$1"
    local TRAIN_PARQUET="$2"
    local VAL_PARQUET="$3"
    local MODEL_IN="$4"
    local OUTPUT_DIR="$5"
    local LR="$6"
    local EPOCHS="$7"

    echo ""
    echo "============================================================"
    echo "  HSPO SFT Step: $STEP_NAME"
    echo "  Model:  $MODEL_IN"
    echo "  Train:  $TRAIN_PARQUET"
    echo "  Output: $OUTPUT_DIR"
    echo "  LR:     $LR"
    echo "  Epochs: $EPOCHS"
    echo "============================================================"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$TRAIN_PARQUET" \
        data.val_files="$VAL_PARQUET" \
        data.train_batch_size="$TRAIN_BATCH" \
        data.micro_batch_size_per_gpu="$MICRO_BATCH" \
        data.max_length=12288 \
        data.truncation=right \
        data.multiturn.enable=true \
        data.multiturn.messages_key=messages \
        model.partial_pretrain="$MODEL_IN" \
        +model.torch_dtype=bfloat16 \
        model.enable_gradient_checkpointing=true \
        model.fsdp_config.cpu_offload=false \
        model.trust_remote_code=false \
        ulysses_sequence_parallel_size=1 \
        +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
        optim.lr="$LR" \
        optim.warmup_steps_ratio=0.05 \
        optim.clip_grad=1.0 \
        trainer.total_epochs="$EPOCHS" \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.project_name=hspo_sft \
        trainer.experiment_name="alfworld_${STEP_NAME}_qwen2.5_${MODEL_SIZE}" \
        'trainer.logger=["console","wandb"]' \
        trainer.seed=42
}

# ── Run pipeline steps ────────────────────────────────────────────────────────

# Resolve checkpoint path: find the latest global_step_* subdirectory
resolve_ckpt() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "$dir"
        return
    fi
    local latest
    latest=$(find "$dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$latest" ] && { [ -f "$latest/tokenizer_config.json" ] || [ -f "$latest/tokenizer.json" ]; }; then
        echo "$latest"
    elif [ -f "$dir/tokenizer_config.json" ] || [ -f "$dir/tokenizer.json" ]; then
        echo "$dir"
    else
        echo "$dir"
    fi
}

# Step 1: Format SFT
if [ "$START_STEP" -le 1 ]; then
    run_sft_step \
        "format" \
        "$DATA_DIR/train_format.parquet" \
        "$DATA_DIR/val_format.parquet" \
        "$BASE_MODEL" \
        "${CKPT_DIR}_step1_format" \
        "$LR_FORMAT" \
        2
fi
CKPT_STEP1=$(resolve_ckpt "${CKPT_DIR}_step1_format")

# Step 2: Executor SFT
if [ "$START_STEP" -le 2 ]; then
    run_sft_step \
        "executor" \
        "$DATA_DIR/train_executor.parquet" \
        "$DATA_DIR/val_executor.parquet" \
        "$CKPT_STEP1" \
        "${CKPT_DIR}_step2_executor" \
        "$LR_EXECUTOR" \
        1
fi
CKPT_STEP2=$(resolve_ckpt "${CKPT_DIR}_step2_executor")

# Step 3: Planner SFT
if [ "$START_STEP" -le 3 ]; then
    run_sft_step \
        "planner" \
        "$DATA_DIR/train_planner.parquet" \
        "$DATA_DIR/val_planner.parquet" \
        "$CKPT_STEP2" \
        "${CKPT_DIR}_step3_planner" \
        "$LR_PLANNER" \
        1
fi
CKPT_STEP3=$(resolve_ckpt "${CKPT_DIR}_step3_planner")

# Step 4: Light joint SFT
if [ "$START_STEP" -le 4 ]; then
    run_sft_step \
        "joint" \
        "$DATA_DIR/train_all.parquet" \
        "$DATA_DIR/val_all.parquet" \
        "$CKPT_STEP3" \
        "${CKPT_DIR}_step4_joint" \
        "$LR_JOINT" \
        1
fi
CKPT_STEP4=$(resolve_ckpt "${CKPT_DIR}_step4_joint")

# ── Symlink final checkpoint for RL training ──────────────────────────────────
FINAL_LINK="${CKPT_DIR}"
if [ -L "$FINAL_LINK" ] || [ -d "$FINAL_LINK" ]; then
    rm -rf "$FINAL_LINK"
fi
ln -sf "$CKPT_STEP4" "$FINAL_LINK"

echo ""
echo "============================================================"
echo "  HSPO SFT Pipeline Complete!"
echo "  Step 1 (format):   $CKPT_STEP1"
echo "  Step 2 (executor): $CKPT_STEP2"
echo "  Step 3 (planner):  $CKPT_STEP3"
echo "  Step 4 (joint):    $CKPT_STEP4"
echo "  Final symlink:     $FINAL_LINK → $CKPT_STEP4"
echo "============================================================"
