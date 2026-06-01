set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi
# Many parallel ALFWorld envs (≈ train_batch_size * env.rollout.n + val_batch_size) can OOM the box
# and kill raylet (heartbeat lost -> "node marked dead"). Raise limits and keep defaults modest.
ulimit -u 65536 2>/dev/null || true
# export VLLM_ATTENTION_BACKEND=XFORMERS
# Same machine, two trainings: give each terminal a different card (otherwise both default to GPU 0).
#   CUDA_VISIBLE_DEVICES=0 bash ...   OR   GPU_ID=0 bash ...
#   CUDA_VISIBLE_DEVICES=1 bash ...   OR   GPU_ID=1 bash ...
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES
elif [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
else
    export CUDA_VISIBLE_DEVICES=0
fi
if [ "${RAY_ADDRESS:-}" = "local" ]; then
    unset RAY_ADDRESS
fi

# Two concurrent Ray jobs must not share the same session dir; if you set GPU_ID, we pick a per-GPU tmp path.
if [ -z "${RAY_TMPDIR:-}" ] && [ -n "${GPU_ID:-}" ]; then
    export RAY_TMPDIR="/root/autodl-tmp/ray_tmp_g${GPU_ID}"
fi

# GPUs: must match Ray-visible GPU count (see `ray status` / nvidia-smi). Default single-GPU.
# Two-card one job: export CUDA_VISIBLE_DEVICES=0,1 && N_GPUS_PER_NODE=2 bash ...
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-$N_GPUS_PER_NODE}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/verl-agent/text}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/verl_agent_alfworld/grpo_qwen2.5_1.5b}"
WANDB_ROOT="${WANDB_ROOT:-/root/autodl-tmp/wandb}"
mkdir -p "$RAY_TMPDIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$WANDB_ROOT/runs" "$WANDB_ROOT/cache" "$WANDB_ROOT/config" "$WANDB_ROOT/data"
export TMPDIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_ROOT/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Per-process thread caps (OpenBLAS defaults to many threads x hundreds of Ray workers -> EAGAIN / raylet crash).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.25}"

# Match gigpo_trainer/run_alfworld.sh-style defaults; increase via env only if RAM/CPUs allow.
train_data_size="${TRAIN_DATA_SIZE:-8}"
val_data_size="${VAL_DATA_SIZE:-16}"
group_size="${GROUP_SIZE:-4}"
trainer_logger="${TRAINER_LOGGER:-['console','wandb']}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-0}"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Missing parquet files:"
    echo "  TRAIN_FILE=$TRAIN_FILE"
    echo "  VAL_FILE=$VAL_FILE"
    echo "Please set DATA_DIR, TRAIN_FILE, or VAL_FILE to existing parquet files."
    exit 1
fi

if [ ! -f "$MODEL_PATH/tokenizer_config.json" ] && [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "MODEL_PATH does not look like a HuggingFace checkpoint: $MODEL_PATH"
    echo "Please set MODEL_PATH to a directory containing tokenizer_config.json or tokenizer.json."
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=$dataloader_num_workers \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2} \
    ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_init._temp_dir=$RAY_TMPDIR \
    $@
