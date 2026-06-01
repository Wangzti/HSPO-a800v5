set -x
ENGINE="${ENGINE:-vllm}"
# export VLLM_ATTENTION_BACKEND=XFORMERS
# 指定物理 GPU：任选其一
#   CUDA_VISIBLE_DEVICES=1 bash examples/hspo_trainer/run_alfworld.sh
#   HSPO_GPU=1 bash examples/hspo_trainer/run_alfworld.sh   # 仅在未设置 CUDA_VISIBLE_DEVICES 时生效
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HSPO_GPU:-0}}"
if [ "${RAY_ADDRESS:-}" = "local" ]; then
    unset RAY_ADDRESS
fi

# GPUs: must match Ray-visible GPU count (see `ray status` / nvidia-smi). Default single-GPU.
# Two-card example: export CUDA_VISIBLE_DEVICES=0,1 && N_GPUS_PER_NODE=2 bash ...
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-$N_GPUS_PER_NODE}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/verl-agent/text}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/verl_agent_alfworld/hspo_qwen2.5_1.5b}"
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

# vLLM's CuMemAllocator is incompatible with PyTorch expandable segments.
# If inherited from shell/profile, remove it to avoid init-time AssertionError.
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments:True"* ]]; then
    echo "[HSPO] Detected incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    echo "[HSPO] Removing expandable_segments:True for vLLM compatibility."
    if [ "${PYTORCH_CUDA_ALLOC_CONF}" = "expandable_segments:True" ]; then
        unset PYTORCH_CUDA_ALLOC_CONF
    else
        PYTORCH_CUDA_ALLOC_CONF="$(echo "${PYTORCH_CUDA_ALLOC_CONF}" | sed -E 's/(^|,)\s*expandable_segments:True\s*(,|$)/\1\2/g; s/^,+//; s/,+$//; s/,,+/,/g')"
        [ -z "${PYTORCH_CUDA_ALLOC_CONF}" ] && unset PYTORCH_CUDA_ALLOC_CONF
        export PYTORCH_CUDA_ALLOC_CONF
    fi
fi

# Per-process thread caps (OpenBLAS defaults to many threads x hundreds of Ray workers -> EAGAIN / raylet crash).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ALFWorld creates train_batch_size * group_size + val_batch_size Ray actors.
num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.25}"

train_data_size="${TRAIN_DATA_SIZE:-8}"
val_data_size="${VAL_DATA_SIZE:-16}"
group_size="${GROUP_SIZE:-4}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-512}"
actor_ppo_micro_bsz="${ACTOR_PPO_MICRO_BSZ:-32}"
rollout_logprob_micro_bsz="${ROLLOUT_LOGPROB_MICRO_BSZ:-32}"
ref_logprob_micro_bsz="${REF_LOGPROB_MICRO_BSZ:-32}"
warmup_epochs="${WARMUP_EPOCHS:-10}"
executor_epochs="${EXECUTOR_EPOCHS:-100}"
if [ "${HSPO_DIRECT_JOINT:-0}" = "1" ]; then
    warmup_epochs=0
    executor_epochs=0
    echo "[HSPO] HSPO_DIRECT_JOINT=1 -> warmup_epochs=0, executor_epochs=0"
fi
# Single-GPU hybrid: FSDP actor + vLLM share VRAM. Default ppo max_num_seqs=1024 makes vLLM reserve
# huge KV cache -> "No available memory for the cache blocks". Cap to batch*group + headroom.
rollout_max_num_seqs="${HSPO_VLLM_MAX_NUM_SEQS:-$(( train_data_size * group_size + 48 ))}"
vllm_gpu_mem_util="${HSPO_VLLM_GPU_MEM_UTIL:-0.72}"
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
if [ -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    shard_cnt="$(compgen -G "$MODEL_PATH/model-*.safetensors" | wc -l || true)"
    if [ "${shard_cnt}" = "0" ]; then
        echo "MODEL_PATH looks incomplete: found model.safetensors.index.json but no model-*.safetensors shards."
        echo "Current MODEL_PATH=$MODEL_PATH"
        echo "If you use old SFT checkpoints, ensure shards were uploaded/copied completely."
        exit 1
    fi
fi

python3 -m recipe.hspo.main_hspo \
    --config-name=hspo_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=$dataloader_num_workers \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_bsz \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_logprob_micro_bsz \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$vllm_gpu_mem_util \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_logprob_micro_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_coef=0.01 \
    algorithm.hspo.abg_min_group_size=2 \
    algorithm.hspo.abg_target_size=4 \
    algorithm.hspo.abg_max_branch=4 \
    algorithm.hspo.abg_step_cost=-0.1 \
    algorithm.hspo.abg_completion_bonus=2.0 \
    algorithm.hspo.low_clip_epsilon=0.2 \
    algorithm.hspo.low_beta=0.01 \
    algorithm.hspo.high_gamma=0.99 \
    algorithm.hspo.high_lam=0.95 \
    algorithm.hspo.high_clip_epsilon=0.2 \
    algorithm.hspo.high_beta=0.01 \
    algorithm.hspo.switch_alpha=0.5 \
    algorithm.hspo.alpha_t=0.5 \
    algorithm.hspo.alpha_l=1.0 \
    algorithm.hspo.alpha_h=1.0 \
    algorithm.hspo.alpha_sft=0.1 \
    algorithm.hspo.norm_adv=True \
    algorithm.hspo.warmup_epochs=$warmup_epochs \
    algorithm.hspo.executor_epochs=$executor_epochs \
    env.env_name=alfworld/AlfredTWEnvOptions \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='hspo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2} \
    ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_init._temp_dir=$RAY_TMPDIR \
    $@
