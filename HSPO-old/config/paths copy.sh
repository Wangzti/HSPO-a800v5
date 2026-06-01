# ============================================================
# HSPO Path Configuration — Single Source of Truth
# ============================================================
#
# This is the ONLY file you need to edit when moving servers.
# All shell scripts source this; Python scripts read the env vars.
#
# Usage:
#   source config/paths.sh
#
# Override any path BEFORE sourcing:
#   export HSPO_DATA_ROOT=/new/path
#   source config/paths.sh
# ============================================================

# ── Base Paths (edit these for your server) ──────────────────
export HSPO_PROJECT_ROOT="${HSPO_PROJECT_ROOT:-/mnt/nfs/ztwang/projects/demos/HSPO}"
export HSPO_AGENT_ROOT="${HSPO_AGENT_ROOT:-$HSPO_PROJECT_ROOT/HSPO-agent}"
export HIPER_AGENT_ROOT="${HIPER_AGENT_ROOT:-$HSPO_PROJECT_ROOT/HiPER-agent}"

export HSPO_DATA_ROOT="${HSPO_DATA_ROOT:-/mnt/nfs/ztwang/data/hspo}"
export HSPO_CKPT_ROOT="${HSPO_CKPT_ROOT:-/mnt/nfs/ztwang/checkpoints/hspo}"
export HSPO_MODEL_ROOT="${HSPO_MODEL_ROOT:-/mnt/nfs/ztwang/models}"
export HSPO_OUTPUT_ROOT="${HSPO_OUTPUT_ROOT:-/mnt/nfs/ztwang/outputs/hspo}"

# ── Conda / Infrastructure ───────────────────────────────────
export HSPO_CONDA_ENV_DIR="${HSPO_CONDA_ENV_DIR:-/mnt/nfs/ztwang/conda_envs}"
export HSPO_CONDA_ENV="${HSPO_CONDA_ENV:-verl}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/nfs/ztwang/.cache/huggingface/datasets}"

# ── Python / PATH ────────────────────────────────────────────
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export HSPO_PYTEST="${HSPO_PYTEST:-$HSPO_CONDA_ENV_DIR/verl-webshop/bin/pytest}"

# ── SFT Data Paths ───────────────────────────────────────────
export HSPO_SFT_RAW_DIR="${HSPO_SFT_RAW_DIR:-$HSPO_DATA_ROOT/sft/alfworld_raw}"
export HSPO_SFT_DATA_DIR="${HSPO_SFT_DATA_DIR:-$HSPO_DATA_ROOT/sft/alfworld_sft}"

# ── RL Training Data (text format, for verl RL) ──────────────
export HSPO_RL_TRAIN_DATA="${HSPO_RL_TRAIN_DATA:-$HOME/data/verl-agent/text/train.parquet}"
export HSPO_RL_VAL_DATA="${HSPO_RL_VAL_DATA:-$HOME/data/verl-agent/text/test.parquet}"

# ── Model Paths ──────────────────────────────────────────────
export HSPO_MODEL_05B="${HSPO_MODEL_05B:-$HSPO_MODEL_ROOT/Qwen2.5-0.5B-Instruct}"
export HSPO_MODEL_1_5B="${HSPO_MODEL_1_5B:-$HSPO_MODEL_ROOT/Qwen2.5-1.5B-Instruct}"
export HSPO_MODEL_7B="${HSPO_MODEL_7B:-$HSPO_MODEL_ROOT/Qwen2.5-7B-Instruct}"

# ── SePRL (legacy HiPER-agent) Paths ─────────────────────────
export SEPRL_DATA_ROOT="${SEPRL_DATA_ROOT:-/mnt/nfs/ztwang/data/seprl}"
export SEPRL_SFT_RAW_DIR="${SEPRL_SFT_RAW_DIR:-$SEPRL_DATA_ROOT/sft/alfworld_raw}"
export SEPRL_SFT_DATA_DIR="${SEPRL_SFT_DATA_DIR:-$SEPRL_DATA_ROOT/sft/alfworld_sft}"
export SEPRL_CKPT_ROOT="${SEPRL_CKPT_ROOT:-/mnt/nfs/ztwang/checkpoints/seprl}"
export SEPRL_RL_DATA_DIR="${SEPRL_RL_DATA_DIR:-/mnt/nfs/ztwang/data/verl-agent}"

# ── ALFWorld ─────────────────────────────────────────────────
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-/home/ztwang/tmp}"

# ── HuggingFace ──────────────────────────────────────────────
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ════════════════════════════════════════════════════════════
# Helper Functions — use these instead of hardcoding paths
# ════════════════════════════════════════════════════════════

# Get base model path by size:  hspo_model_path 0.5b
hspo_model_path() {
    case "${1:-0.5b}" in
        0.5b)  echo "$HSPO_MODEL_05B" ;;
        1.5b)  echo "$HSPO_MODEL_1_5B" ;;
        7b)    echo "$HSPO_MODEL_7B" ;;
        *)     echo "" ;;
    esac
}

# Get HSPO SFT checkpoint dir:  hspo_sft_ckpt_dir 0.5b
hspo_sft_ckpt_dir() {
    echo "$HSPO_CKPT_ROOT/sft/alfworld_qwen2.5_${1}_instruct"
}

# Get HSPO RL checkpoint dir:  hspo_rl_ckpt_dir 0.5b low_level
hspo_rl_ckpt_dir() {
    echo "$HSPO_CKPT_ROOT/rl/alfworld_${2}_qwen2.5_${1}"
}

# Get HSPO rollout dump dir:  hspo_rollout_dump_dir 0.5b low_level
hspo_rollout_dump_dir() {
    echo "$HSPO_OUTPUT_ROOT/rollout_audit/alfworld_${2}_qwen2.5_${1}"
}

# Get SePRL SFT checkpoint:  seprl_sft_ckpt 0.5b
seprl_sft_ckpt() {
    echo "$SEPRL_CKPT_ROOT/sft/alfworld_qwen2.5_${1}_instruct/global_step_75"
}

# Get HSPO SFT checkpoint (fallback):  hspo_fallback_sft_ckpt 0.5b
hspo_fallback_sft_ckpt() {
    case "${1:-0.5b}" in
        0.5b)  echo "$SEPRL_CKPT_ROOT/sft/alfworld_qwen2.5_0.5b_instruct" ;;
        *)     echo "" ;;
    esac
}

echo "[paths.sh] HSPO paths loaded (data=$HSPO_DATA_ROOT, ckpt=$HSPO_CKPT_ROOT, models=$HSPO_MODEL_ROOT)"
