#!/usr/bin/env bash
# ============================================================
# HSPO ALFWorld Training Script
# ============================================================
#
# Phase 1 – Low-Level Executor Training
#   algorithm.hspo.phase=low_level
#   Freeze planner, train executor with PRM-based process reward
#
# Phase 2 – High-Level Planner Training
#   algorithm.hspo.phase=high_level
#   Freeze executor, train planner with macro-PPO
#
# Phase 3 – Joint Training (optional)
#   algorithm.hspo.phase=joint
#
# Prerequisites
# -------------
#   conda activate verl
#   source config/paths.sh
#   pip install -e .
#
# Usage:
#   bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh           # Phase 1 (low-level)
#   bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh high_level # Phase 2
#   bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh joint      # Phase 3

set -x

PHASE=${1:-low_level}
ENGINE=${2:-vllm}
USER_ARGS=()
if [ "$#" -ge 1 ]; then
    shift
fi
if [ "$#" -ge 1 ]; then
    shift
fi
USER_ARGS=("$@")
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
if [ "${RAY_ADDRESS:-}" = "local" ]; then
    unset RAY_ADDRESS
fi
export OMP_NUM_THREADS=1

# ── Locate HSPO-agent root & source path config ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSPO_AGENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$HSPO_AGENT_ROOT/config/paths.sh"

# ── PYTHONPATH: HSPO-agent must come BEFORE HiPER-agent so that HSPO's
#    agent_system (with HSPORewardManager) takes priority over HiPER's.
export PYTHONPATH="$HSPO_AGENT_ROOT:$HIPER_AGENT_ROOT:${PYTHONPATH:-}"
export HSPO_ROOT="$HSPO_AGENT_ROOT"

resolve_hf_checkpoint() {
    local path="$1"
    if [ -z "$path" ] || [ ! -d "$path" ]; then
        return 1
    fi
    if [ -f "$path/tokenizer_config.json" ] || [ -f "$path/tokenizer.json" ]; then
        printf '%s\n' "$path"
        return 0
    fi
    local latest_step
    latest_step="$(find "$path" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [ -n "$latest_step" ] && { [ -f "$latest_step/tokenizer_config.json" ] || [ -f "$latest_step/tokenizer.json" ]; }; then
        printf '%s\n' "$latest_step"
        return 0
    fi
    return 1
}

# ── Paths ──────────────────────────────────────────────────────────────────
# Model selection. Override with MODEL_PATH=/path/to/checkpoint when needed.
MODEL_SIZE=${MODEL_SIZE:-1.5b}
BASE_MODEL=$(hspo_model_path "$MODEL_SIZE")
DEFAULT_SFT_CKPT=$(hspo_sft_ckpt_dir "$MODEL_SIZE")
FALLBACK_SFT_CKPT=$(hspo_fallback_sft_ckpt "$MODEL_SIZE")

if [ -z "$BASE_MODEL" ]; then
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use 0.5b, 1.5b, or 7b, or set MODEL_PATH explicitly."
    exit 1
fi

if [ -z "${MODEL_PATH:-}" ]; then
    if MODEL_PATH="$(resolve_hf_checkpoint "$DEFAULT_SFT_CKPT")"; then
        echo "Using HSPO SFT checkpoint $MODEL_PATH"
    elif [ -n "$FALLBACK_SFT_CKPT" ] && MODEL_PATH="$(resolve_hf_checkpoint "$FALLBACK_SFT_CKPT")"; then
        echo "HSPO SFT checkpoint not found at $DEFAULT_SFT_CKPT, using fallback SFT checkpoint $MODEL_PATH"
    else
        echo "SFT checkpoint not found at $DEFAULT_SFT_CKPT, using base model $BASE_MODEL"
        MODEL_PATH=$BASE_MODEL
    fi
elif ! MODEL_PATH="$(resolve_hf_checkpoint "$MODEL_PATH")"; then
    echo "MODEL_PATH does not look like a complete HuggingFace checkpoint: ${MODEL_PATH:-}"
    exit 1
fi

TRAIN_DATA=${TRAIN_DATA:-$HSPO_RL_TRAIN_DATA}
VAL_DATA=${VAL_DATA:-$HSPO_RL_VAL_DATA}
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Preparing data..."
    "$PYTHON_BIN" "$HSPO_AGENT_ROOT/example_scripts/data_preprocess/prepare.py" \
        --mode text \
        --train_data_size 128 \
        --val_data_size 128
fi

OUT_DIR=${OUT_DIR:-$(hspo_rl_ckpt_dir "$MODEL_SIZE" "$PHASE")}
ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-$(hspo_rollout_dump_dir "$MODEL_SIZE" "$PHASE")}

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo "=== HSPO Pre-flight Checks ==="
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  TRAIN_DATA:  $TRAIN_DATA"
echo "  VAL_DATA:    $VAL_DATA"
echo "  OUT_DIR:     $OUT_DIR"
echo "  PHASE:       $PHASE"

if [ ! -f "$MODEL_PATH/tokenizer_config.json" ] && [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "WARNING: MODEL_PATH does not contain a tokenizer config."
    echo "  Expected HuggingFace checkpoint at: $MODEL_PATH"
    echo "  If SFT hasn't been run yet, run run_sft_pipeline.sh first."
fi

# Verify HSPO package is importable
if ! "$PYTHON_BIN" -c "import hspo" 2>/dev/null; then
    echo "WARNING: hspo package not found on PYTHONPATH."
    echo "  Run: pip install -e $HSPO_AGENT_ROOT"
fi
echo "================================"

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
train_data_size=128
val_data_size=128
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-50}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
if [ -z "${TRAINER_LOGGER:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        TRAINER_LOGGER="['console','wandb']"
    else
        TRAINER_LOGGER="['console']"
    fi
fi
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}

EXTRA_ARGS=()
if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    EXTRA_ARGS+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

# ── Launch ──────────────────────────────────────────────────────────────────
cd "$HIPER_AGENT_ROOT" && "$PYTHON_BIN" -m verl.trainer.main_ppo_hspo \
    algorithm.adv_estimator=grpo \
    +algorithm.hspo.enabled=true \
    +algorithm.hspo.phase=${PHASE} \
    +algorithm.hspo.gamma_low=0.95 \
    +algorithm.hspo.lam_low=0.90 \
    +algorithm.hspo.eta_done=1.0 \
    +algorithm.hspo.tau_done=0.9 \
    +algorithm.hspo.lambda_invalid=1.0 \
    +algorithm.hspo.lambda_side=0.5 \
    +algorithm.hspo.lambda_step=0.01 \
    +algorithm.hspo.max_segment_len=8 \
    +algorithm.hspo.switch_loss_type=ce \
    +algorithm.hspo.switch_loss_coef=0.1 \
    +algorithm.hspo.gamma_high=0.95 \
    +algorithm.hspo.lam_high=0.95 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.hspo_switch_loss_coef=0.1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.use_two_heads_critic=False \
    critic.use_three_heads_critic=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnvOptions \
    env.seed=6 \
    env.max_steps=${MAX_STEPS} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    reward_model.reward_manager=hspo \
    trainer.logger=${TRAINER_LOGGER} \
    trainer.log_val_generations=1 \
    trainer.project_name='hspo_alfworld' \
    trainer.experiment_name="hspo_${PHASE}_qwen2.5_${MODEL_SIZE}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=disable \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir=${OUT_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DUMP_DIR} \
    ray_init.num_cpus=${RAY_NUM_CPUS:-8} \
    "${EXTRA_ARGS[@]}" \
    "${USER_ARGS[@]}"
