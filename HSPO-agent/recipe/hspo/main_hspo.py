"""
HSPO: Hierarchical Subgoal-conditioned Process Optimization.
Main entry point.

Usage:
    python3 -m recipe.hspo.main_hspo --config-name=hspo_trainer
"""

import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from recipe.hspo.hspo_ray_trainer import HSPORTrainer, ResourcePoolManager, Role
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="hspo_trainer", version_base=None)
def main(config):
    run_hspo(config)


def run_hspo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        merged_runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({
            **ray_init_kwargs,
            "runtime_env": merged_runtime_env,
        })
        ray_init_dict = OmegaConf.to_container(ray_init_kwargs)
        if ray_init_dict.get("address") is not None:
            ray_init_dict.pop("num_cpus", None)
            ray_init_dict.pop("num_gpus", None)
        print(f"ray init kwargs: {ray_init_dict}")
        ray.init(**ray_init_dict)

    # ---- Resolve config ----
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # ---- Copy model to local ----
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    print(f"[HSPO] Model path: {local_path}")

    # ---- Create HSPO environments ----
    print("[HSPO] Creating environments...")
    from recipe.hspo.env_manager import make_hspo_envs
    envs, val_envs = make_hspo_envs(config)
    print("[HSPO] Environments created.")

    # ---- Tokenizer & processor ----
    print("[HSPO] Loading tokenizer & processor...")
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(
        local_path, trust_remote_code=trust_remote_code, use_fast=True,
    )
    print("[HSPO] Tokenizer & processor loaded.")

    # ---- Worker classes ----
    actor_strategy = config.actor_rollout_ref.actor.strategy

    if actor_strategy in ("fsdp", "fsdp2"):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup
    elif actor_strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(f"Unsupported strategy: {actor_strategy}")

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    # ---- Resource pool ----
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }

    # ---- Reward manager ----
    reward_manager_name = config.reward_model.get("reward_manager", "episode")
    if reward_manager_name == "episode":
        from agent_system.reward_manager import EpisodeRewardManager
        reward_manager_cls = EpisodeRewardManager
    else:
        raise NotImplementedError(
            f"Unknown reward manager: {reward_manager_name}"
        )

    reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=0, normalize_by_length=False,
    )
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=1, normalize_by_length=False,
    )

    # ---- Resource pool manager ----
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping,
    )

    # ---- Trajectory collector ----
    from agent_system.multi_turn_rollout import TrajectoryCollector
    traj_collector = TrajectoryCollector(
        config=config, tokenizer=tokenizer, processor=processor,
    )

    # ---- Datasets ----
    print("[HSPO] Loading datasets...")
    from verl.utils.dataset.rl_dataset import collate_fn

    train_dataset = _create_rl_dataset(
        config.data.train_files, config.data, tokenizer, processor,
    )
    val_dataset = _create_rl_dataset(
        config.data.val_files, config.data, tokenizer, processor,
    )
    train_sampler = _create_rl_sampler(config.data, train_dataset)
    print("[HSPO] Datasets loaded.")

    # ---- Trainer ----
    trainer = HSPORTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
        traj_collector=traj_collector,
        envs=envs,
        val_envs=val_envs,
    )

    print("[HSPO] Initializing workers (loading model to GPU, this takes a few minutes)...")
    trainer.init_workers()
    print("[HSPO] Workers initialized. Starting training...")
    trainer.fit()


def _create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_type

    if data_config.get("custom_cls", {}).get("path", None) is not None:
        dataset_cls = load_extern_type(
            data_config.custom_cls.path, data_config.custom_cls.name,
        )
        if not issubclass(dataset_cls, Dataset):
            raise TypeError("Custom dataset class must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def _create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        gen = torch.Generator()
        gen.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=gen)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
