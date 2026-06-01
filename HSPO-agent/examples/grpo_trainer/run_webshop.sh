set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi
ulimit -u 65536
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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/verl_agent_webshop/grpo_qwen2.5_1.5b}"
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

# Per-process thread caps (OpenBLAS defaults to many threads x many Ray workers).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.05}"

# PERF_PRESET=stable : safer around vLLM wake_up(kv_cache) between train steps (may be slower).
# PERF_PRESET=speed   : OK when peak VRAM (~60GiB on ~80GiB) leaves headroom; larger micro-batches,
#                        no CUDA-graph disable, optimizer stays on GPU. Re-enable stable if step-2 OOM returns.
PERF_PRESET="${PERF_PRESET:-stable}"
if [ "$PERF_PRESET" = "stable" ]; then
    : "${GPU_MEMORY_UTILIZATION:=0.70}"
    : "${ROLLOUT_MAX_NUM_SEQS:=32}"
    : "${MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ACTOR_PARAM_OFFLOAD:=true}"
    : "${ACTOR_OPTIMIZER_OFFLOAD:=true}"
    : "${MAX_PROMPT_LENGTH:=3072}"
    : "${ACTOR_PPO_MICRO_BATCH_PER_GPU:=4}"
    : "${ROLLOUT_ENFORCE_EAGER:=true}"
    : "${ROLLOUT_FREE_CACHE_ENGINE:=true}"
    : "${ROLLOUT_LOGPROB_MICRO_PER_GPU:=16}"
    : "${REF_LOGPROB_MICRO_PER_GPU:=16}"
else
    # speed (default): use leftover VRAM to cut actor accumulation and vLLM eager overhead.
    : "${GPU_MEMORY_UTILIZATION:=0.82}"
    : "${ROLLOUT_MAX_NUM_SEQS:=40}"
    : "${MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ACTOR_PARAM_OFFLOAD:=true}"
    : "${ACTOR_OPTIMIZER_OFFLOAD:=false}"
    : "${MAX_PROMPT_LENGTH:=3072}"
    : "${ACTOR_PPO_MICRO_BATCH_PER_GPU:=8}"
    : "${ROLLOUT_ENFORCE_EAGER:=false}"
    : "${ROLLOUT_FREE_CACHE_ENGINE:=false}"
    : "${ROLLOUT_LOGPROB_MICRO_PER_GPU:=24}"
    : "${REF_LOGPROB_MICRO_PER_GPU:=24}"
fi
# Must satisfy: chunked_prefill implies max_num_batched_tokens >= max_model_len (~ MAX_PROMPT + response).

train_data_size="${TRAIN_DATA_SIZE:-8}"
val_data_size="${VAL_DATA_SIZE:-16}"
group_size="${GROUP_SIZE:-4}"
trainer_logger="${TRAINER_LOGGER:-['console','wandb']}"

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

if [ ! -f "$WEBSHOP_DATA_ROOT/items_shuffle.json" ] || [ ! -f "$WEBSHOP_DATA_ROOT/items_ins_v2.json" ] || [ ! -f "$WEBSHOP_DATA_ROOT/items_human_ins.json" ]; then
    echo "Missing WebShop data under $WEBSHOP_DATA_ROOT"
    echo "Expected items_shuffle.json, items_ins_v2.json, and items_human_ins.json."
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='grpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=${TRAINER_TEST_FREQ:-10} \
    trainer.total_training_steps=500 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2} \
    ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_init._temp_dir=$RAY_TMPDIR \
    $@
