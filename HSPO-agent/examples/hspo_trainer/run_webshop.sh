#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE="vllm"
GPU_ID="${GPU_ID:-0}"
USER_ARGS=()

# Backward-compatible positional engine: run_webshop.sh vllm
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    ENGINE="$1"
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            USER_ARGS+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
if [ "${RAY_ADDRESS:-}" = "local" ]; then
    unset RAY_ADDRESS
fi
if [ -z "${PYTHON_CMD:-}" ]; then
    if [ -x "/root/miniconda3/envs/verl/bin/python3" ]; then
        PYTHON_CMD="/root/miniconda3/envs/verl/bin/python3"
    else
        PYTHON_CMD="python3"
    fi
fi
"$PYTHON_CMD" -c "import sys; print('[run_webshop] python =', sys.executable)"

# GPUs: must match Ray-visible GPU count (see `ray status` / nvidia-smi).
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-$N_GPUS_PER_NODE}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
SFT_CKPT_PATH="${SFT_CKPT_PATH:-}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/verl-agent/text}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/verl_agent_webshop/hspo_qwen2.5_1.5b}"
WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-/root/autodl-tmp/data/webshop}"
WANDB_ROOT="${WANDB_ROOT:-/root/autodl-tmp/wandb}"

mkdir -p "$RAY_TMPDIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$WANDB_ROOT/runs" "$WANDB_ROOT/cache" "$WANDB_ROOT/config" "$WANDB_ROOT/data"

export TMPDIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"
export WEBSHOP_DATA_ROOT="$WEBSHOP_DATA_ROOT"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_ROOT/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Per-process thread caps (avoid thread explosion in Ray workers).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.05}"
train_data_size="${TRAIN_DATA_SIZE:-4}"
val_data_size="${VAL_DATA_SIZE:-16}"
group_size="${GROUP_SIZE:-2}"
trainer_logger="${TRAINER_LOGGER:-['console','wandb']}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-True}"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Missing parquet files:"
    echo "  TRAIN_FILE=$TRAIN_FILE"
    echo "  VAL_FILE=$VAL_FILE"
    echo "Please set DATA_DIR, TRAIN_FILE, or VAL_FILE to existing parquet files."
    exit 1
fi

if [ -n "$SFT_CKPT_PATH" ]; then
    if [ ! -f "$SFT_CKPT_PATH/tokenizer_config.json" ] && [ ! -f "$SFT_CKPT_PATH/tokenizer.json" ]; then
        echo "SFT_CKPT_PATH is set but not a HuggingFace checkpoint: $SFT_CKPT_PATH"
        exit 1
    fi
    ACTOR_MODEL_PATH="$SFT_CKPT_PATH"
    echo "Using SFT_CKPT_PATH as actor model: $ACTOR_MODEL_PATH"
else
    ACTOR_MODEL_PATH="$MODEL_PATH"
fi

if [ ! -f "$ACTOR_MODEL_PATH/tokenizer_config.json" ] && [ ! -f "$ACTOR_MODEL_PATH/tokenizer.json" ]; then
    echo "MODEL_PATH does not look like a HuggingFace checkpoint: $ACTOR_MODEL_PATH"
    echo "Please set MODEL_PATH (or SFT_CKPT_PATH) to a directory containing tokenizer_config.json or tokenizer.json."
    exit 1
fi

if [ ! -f "$WEBSHOP_DATA_ROOT/items_shuffle.json" ] || [ ! -f "$WEBSHOP_DATA_ROOT/items_ins_v2.json" ] || [ ! -f "$WEBSHOP_DATA_ROOT/items_human_ins.json" ]; then
    echo "Missing WebShop data under $WEBSHOP_DATA_ROOT"
    echo "Expected items_shuffle.json, items_ins_v2.json, and items_human_ins.json."
    exit 1
fi

# Ensure Lucene index exists for WebShop search engine.
SEARCH_ENGINE_DIR="$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine"
INDEX_DIR="$SEARCH_ENGINE_DIR/indexes"
INDEX_1K_DIR="$SEARCH_ENGINE_DIR/indexes_1k"
if [ "$WEBSHOP_USE_SMALL" = "True" ] || [ "$WEBSHOP_USE_SMALL" = "true" ]; then
    TARGET_INDEX_DIR="$INDEX_1K_DIR"
else
    TARGET_INDEX_DIR="$INDEX_DIR"
fi
if [ ! -d "$TARGET_INDEX_DIR" ]; then
    echo "WebShop index missing: $TARGET_INDEX_DIR"
    echo "Building WebShop search index (this may take a while)..."
    (
        cd "$SEARCH_ENGINE_DIR" || exit 1
        mkdir -p resources resources_100 resources_1k resources_100k indexes indexes_100 indexes_1k indexes_100k
        "$PYTHON_CMD" convert_product_file_format.py
        if [ "$WEBSHOP_USE_SMALL" = "True" ] || [ "$WEBSHOP_USE_SMALL" = "true" ]; then
            "$PYTHON_CMD" -m pyserini.index.lucene \
                --collection JsonCollection \
                --input resources_1k \
                --index indexes_1k \
                --generator DefaultLuceneDocumentGenerator \
                --threads 1 \
                --storePositions --storeDocvectors --storeRaw
        else
            "$PYTHON_CMD" -m pyserini.index.lucene \
                --collection JsonCollection \
                --input resources \
                --index indexes \
                --generator DefaultLuceneDocumentGenerator \
                --threads 1 \
                --storePositions --storeDocvectors --storeRaw
        fi
    )
fi

# Lucene index validity check: at least one segments* file must exist.
if ! compgen -G "$TARGET_INDEX_DIR/segments*" > /dev/null; then
    echo "Invalid WebShop index: $TARGET_INDEX_DIR (missing segments* files)."
    echo "Please ensure pyserini/java are available and rerun."
    exit 1
fi

"$PYTHON_CMD" -m recipe.hspo.main_hspo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=Webshop \
    env.webshop.use_small="${WEBSHOP_USE_SMALL}" \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='hspo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-25}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-500}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-200}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}" \
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-2}" \
    ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_init._temp_dir=$RAY_TMPDIR \
    "${USER_ARGS[@]}"
