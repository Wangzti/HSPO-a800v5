# ============================================================
# HSPO Path Configuration — Single Source of Truth
# ============================================================
#
# This is the ONLY file you need to edit when moving servers.
# All shell scripts source this; Python scripts read the env vars.
#
# Server: autodl (root@autodl-container)
#
# Usage:
#   source config/paths.sh
#
# Override any path BEFORE sourcing:
#   export HSPO_DATA_ROOT=/new/path
#   source config/paths.sh
# ============================================================

# ── Base Paths (edit these for your server) ──────────────────
export HSPO_PROJECT_ROOT="${HSPO_PROJECT_ROOT:-/root/projects/HSPO}"
export HSPO_AGENT_ROOT="${HSPO_AGENT_ROOT:-$HSPO_PROJECT_ROOT/HSPO-agent}"
export HIPER_AGENT_ROOT="${HIPER_AGENT_ROOT:-$HSPO_PROJECT_ROOT/HiPER-agent}"

export HSPO_DATA_ROOT="${HSPO_DATA_ROOT:-/root/autodl-tmp/data}"
export HSPO_CKPT_ROOT="${HSPO_CKPT_ROOT:-/root/autodl-tmp/checkpoints}"
export HSPO_MODEL_ROOT="${HSPO_MODEL_ROOT:-/root/autodl-tmp/models}"
export HSPO_OUTPUT_ROOT="${HSPO_OUTPUT_ROOT:-/root/autodl-tmp/outputs}"

# ── Conda / Infrastructure ───────────────────────────────────
export HSPO_CONDA_ENV_DIR="${HSPO_CONDA_ENV_DIR:-/root/autodl-tmp/conda_envs}"
export HSPO_CONDA_ENV="${HSPO_CONDA_ENV:-verl}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/.cache/huggingface/datasets}"

# ── Python / PATH ────────────────────────────────────────────
if [ -z "${PYTHON_BIN:-}" ] && [ -x "/root/miniconda3/envs/verl/bin/python" ]; then
    export PYTHON_BIN="/root/miniconda3/envs/verl/bin/python"
else
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# ── SFT Data Paths ───────────────────────────────────────────
export HSPO_SFT_RAW_DIR="${HSPO_SFT_RAW_DIR:-$HSPO_DATA_ROOT/sft/alfworld_raw}"
export HSPO_SFT_DATA_DIR="${HSPO_SFT_DATA_DIR:-$HSPO_DATA_ROOT/sft/alfworld_sft}"

# ── RL Training Data (text format, for verl RL) ──────────────
export HSPO_RL_TRAIN_DATA="${HSPO_RL_TRAIN_DATA:-$HSPO_DATA_ROOT/verl-agent/text/train.parquet}"
export HSPO_RL_VAL_DATA="${HSPO_RL_VAL_DATA:-$HSPO_DATA_ROOT/verl-agent/text/test.parquet}"

# ── Model Paths ──────────────────────────────────────────────
export HSPO_MODEL_05B="${HSPO_MODEL_05B:-$HSPO_MODEL_ROOT/Qwen2.5-0.5B-Instruct}"
export HSPO_MODEL_1_5B="${HSPO_MODEL_1_5B:-$HSPO_MODEL_ROOT/Qwen2.5-1.5B-Instruct}"
export HSPO_MODEL_7B="${HSPO_MODEL_7B:-$HSPO_MODEL_ROOT/Qwen2.5-7B-Instruct}"

# ── SePRL (legacy HiPER-agent) Paths ─────────────────────────
export SEPRL_DATA_ROOT="${SEPRL_DATA_ROOT:-$HSPO_DATA_ROOT/seprl}"
export SEPRL_SFT_RAW_DIR="${SEPRL_SFT_RAW_DIR:-$SEPRL_DATA_ROOT/sft/alfworld_raw}"
export SEPRL_SFT_DATA_DIR="${SEPRL_SFT_DATA_DIR:-$SEPRL_DATA_ROOT/sft/alfworld_sft}"
export SEPRL_CKPT_ROOT="${SEPRL_CKPT_ROOT:-$HSPO_CKPT_ROOT/seprl}"
export SEPRL_RL_DATA_DIR="${SEPRL_RL_DATA_DIR:-$HSPO_DATA_ROOT/verl-agent}"

# ── ALFWorld ─────────────────────────────────────────────────
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"

# ── HuggingFace (autodl has internet, keep online by default) ─
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# ════════════════════════════════════════════════════════════
# Helper Functions — use these instead of hardcoding paths
# ════════════════════════════════════════════════════════════

# Get base model path by size:  hspo_model_path 1.5b
hspo_model_path() {
    case "${1:-1.5b}" in
        0.5b)  echo "$HSPO_MODEL_05B" ;;
        1.5b)  echo "$HSPO_MODEL_1_5B" ;;
        7b)    echo "$HSPO_MODEL_7B" ;;
        *)     echo "" ;;
    esac
}

# Get HSPO SFT checkpoint dir:  hspo_sft_ckpt_dir 1.5b
hspo_sft_ckpt_dir() {
    echo "$HSPO_CKPT_ROOT/sft/alfworld_qwen2.5_${1}_instruct_step4_joint"
}

# Get HSPO RL checkpoint dir:  hspo_rl_ckpt_dir 1.5b low_level
hspo_rl_ckpt_dir() {
    echo "$HSPO_CKPT_ROOT/rl/alfworld_${2}_qwen2.5_${1}"
}

# Get HSPO rollout dump dir:  hspo_rollout_dump_dir 1.5b low_level
hspo_rollout_dump_dir() {
    echo "$HSPO_OUTPUT_ROOT/rollout_audit/alfworld_${2}_qwen2.5_${1}"
}

# Get SePRL SFT checkpoint:  seprl_sft_ckpt 0.5b
seprl_sft_ckpt() {
    echo "$SEPRL_CKPT_ROOT/sft/alfworld_qwen2.5_${1}_instruct/global_step_75"
}

# Get HSPO SFT checkpoint (fallback):  hspo_fallback_sft_ckpt 1.5b
hspo_fallback_sft_ckpt() {
    echo "$HSPO_CKPT_ROOT/sft/alfworld_qwen2.5_${1}_instruct_step4_joint/global_step_874"
}

echo "[paths.sh] HSPO paths loaded (data=$HSPO_DATA_ROOT, ckpt=$HSPO_CKPT_ROOT, models=$HSPO_MODEL_ROOT)"
