"""
HSPO: Ray-based FSDP PPO Trainer with Plan-Execute + ABG support.

Training flow:
1. Environment rollout with Plan-Execute prompts (<switch>/<subgoal>/<action>)
2. HSPO advantage computation (ABG low-level + macro high-level)
3. Token-level credit routing losses
4. FSDP worker group update
"""

import ast
import numbers
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking, ValidationGenerationsLogger

from recipe.hspo import core_hspo
from recipe.hspo.milestone_scorer import get_milestone_scorer


# ---------------------------------------------------------------------------
#  Enums & Config
# ---------------------------------------------------------------------------

class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class TrainingPhase(str, Enum):
    WARMUP = "warmup"       # SFT-like format training
    EXECUTOR = "executor"   # Low-level action optimization
    JOINT = "joint"         # Full HSPO optimization


@dataclass
class ResourcePoolManager:
    resource_pool_spec: Dict[str, List[int]]
    mapping: Dict[Role, str]
    resource_pool_dict: Dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=pool_name,
            )
            self.resource_pool_dict[pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        total = 0
        for process_on_nodes in self.resource_pool_spec.values():
            total += sum(process_on_nodes)
        return total

    def _check_resource_available(self):
        node_resources = ray.state.available_resources_per_node()
        node_gpus = {
            node: info.get("GPU", 0) if "GPU" in info else info.get("NPU", 0)
            for node, info in node_resources.items()
        }
        available = sum(node_gpus.values())
        required = sum(
            n_gpus for proc in self.resource_pool_spec.values() for n_gpus in proc
        )
        if available < required:
            raise ValueError(
                f"Available GPUs ({available}) < required ({required})"
            )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def compute_response_mask(data: DataProto) -> torch.Tensor:
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def normalize_trainer_loggers(logger_cfg: Any) -> List[str]:
    """Normalize Hydra `trainer.logger` into backend names compatible with GiGPO / `Tracking`."""
    if logger_cfg is None:
        return ["console"]
    try:
        if OmegaConf.is_config(logger_cfg):
            logger_cfg = OmegaConf.to_container(logger_cfg, resolve=True)
    except Exception:
        pass
    if isinstance(logger_cfg, str):
        s = logger_cfg.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError, TypeError):
                pass
        s = s.strip("'\"")
        return [s] if s else ["console"]
    if isinstance(logger_cfg, (list, tuple)):
        return [str(x) for x in logger_cfg]
    return ["console"]


def _metrics_dict_to_floats(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Keep only numeric scalars for `Tracking` / `LocalLogger` (GiGPO-style one-line metrics)."""
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, numbers.Number) and not isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif hasattr(v, "item") and callable(getattr(v, "item", None)):
            try:
                out[k] = float(v.item())
            except Exception:
                continue
    return out


def _extract_rollout_metrics(data: DataProto) -> Dict[str, float]:
    """Extract GiGPO-like rollout scalar metrics from tensor and non-tensor fields."""
    metrics: Dict[str, float] = {}

    try:
        prompts = data.batch.get("prompts", None)
        responses = data.batch.get("responses", None)
        if prompts is not None:
            p_lens = (prompts != 0).sum(dim=-1).float()
            metrics["prompt_length/mean"] = float(torch.mean(p_lens).item())
            metrics["prompt_length/max"] = float(torch.max(p_lens).item())
            metrics["prompt_length/min"] = float(torch.min(p_lens).item())
        if responses is not None:
            r_lens = (responses != 0).sum(dim=-1).float()
            metrics["response_length/mean"] = float(torch.mean(r_lens).item())
            metrics["response_length/max"] = float(torch.max(r_lens).item())
            metrics["response_length/min"] = float(torch.min(r_lens).item())
    except Exception:
        pass

    nt = getattr(data, "non_tensor_batch", {}) or {}
    for key in [
        "success_rate",
        "webshop_task_score (not success_rate)",
        "episode_rewards",
        "episode_lengths",
        "tool_callings",
    ]:
        if key not in nt:
            continue
        try:
            arr = np.asarray(nt[key], dtype=np.float64).reshape(-1)
            if arr.size == 0:
                continue
            if key == "episode_rewards":
                name = "episode/reward/mean"
            elif key == "episode_lengths":
                name = "episode/length/mean"
            elif key == "tool_callings":
                name = "episode/tool_call_count/mean"
            elif key == "success_rate":
                name = "episode/success_rate"
            else:
                name = f"episode/{key}"
            metrics[name] = float(np.mean(arr))
        except Exception:
            continue

    # Include task split success rates if environment manager provides them.
    for key, value in nt.items():
        if "success_rate" not in str(key) or str(key) == "success_rate":
            continue
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                metrics[f"episode/{key}"] = float(np.mean(arr))
        except Exception:
            continue

    if "is_action_valid" in nt:
        try:
            valid = np.asarray(nt["is_action_valid"], dtype=np.float64).reshape(-1)
            if valid.size > 0:
                metrics["episode/valid_action_ratio"] = float(np.mean(valid))
        except Exception:
            pass

    return metrics


def finish_tracking_safe(tracking: Optional[Tracking]) -> None:
    """Best-effort shutdown for `Tracking` (mirrors ``Tracking.__del__`` without relying on GC)."""
    if tracking is None:
        return
    loggers = getattr(tracking, "logger", {})
    if "wandb" in loggers:
        try:
            loggers["wandb"].finish(exit_code=0)
        except Exception:
            pass
    if "swanlab" in loggers:
        try:
            loggers["swanlab"].finish()
        except Exception:
            pass
    if "vemlp_wandb" in loggers:
        try:
            loggers["vemlp_wandb"].finish(exit_code=0)
        except Exception:
            pass
    if "tensorboard" in loggers:
        try:
            loggers["tensorboard"].finish()
        except Exception:
            pass
    if "clearml" in loggers:
        try:
            loggers["clearml"].finish()
        except Exception:
            pass


def masked_mean(x: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    if axis is None:
        return (x * mask).sum() / mask.sum().clamp(min=1e-8)
    return (x * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1e-8)


def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty: str = "kl",
) -> tuple:
    response_mask = compute_response_mask(data)
    token_level_scores = data.batch["token_level_scores"]
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"],
        data.batch["ref_log_prob"],
        kl_penalty=kl_penalty,
    )
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, response_mask)
    current_kl = torch.mean(current_kl).item()
    batch_size = data.batch.batch_size[0]
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    return data, {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }


# ---------------------------------------------------------------------------
#  HSPO Trainer
# ---------------------------------------------------------------------------

class HSPORTrainer:
    """Ray-based HSPO trainer."""

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping: Dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls,
        reward_fn,
        val_reward_fn,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collate_fn,
        train_sampler: Sampler,
        device_name: str = "cuda",
        traj_collector=None,
        envs=None,
        val_envs=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        self.collate_fn = collate_fn
        self.traj_collector = traj_collector
        self.envs = envs
        self.val_envs = val_envs
        self.device_name = device_name

        # HSPO config
        self.hspo_cfg = core_hspo.HSPOConfig()
        if hasattr(config.algorithm, "hspo"):
            for k in self.hspo_cfg.__dataclass_fields__:
                if hasattr(config.algorithm.hspo, k):
                    setattr(self.hspo_cfg, k, getattr(config.algorithm.hspo, k))

        # Milestone scorer
        self.milestone_scorer = get_milestone_scorer(config.env.env_name)

        # Training phase
        self.phase = TrainingPhase.WARMUP
        self.current_epoch = 0

        # Worker infrastructure
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls

        # Worker groups (initialized in init_workers)
        self.actor_rollout_wg = None

        # KL controller
        self.kl_ctrl = core_algos.AdaptiveKLController(
            init_kl_coef=config.algorithm.kl_coef,
            target_kl=config.algorithm.target_kl,
            horizon=config.algorithm.kl_horizon,
        )

        # Dataloaders
        from torch.utils.data import DataLoader
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.data.val_batch_size,
            sampler=None,
            collate_fn=collate_fn,
            drop_last=False,
        )
        # Keep GiGPO/HiPER semantics: total_training_steps has priority.
        configured_steps = self.config.trainer.get("total_training_steps", None)
        if configured_steps is not None and int(configured_steps) > 0:
            self.total_training_steps = int(configured_steps)
        else:
            self.total_training_steps = max(1, len(self.train_dataloader) * int(self.config.trainer.total_epochs))

        # Ref model not loaded for HSPO (critic-free, no separate ref policy)
        self._has_ref = False

        # Validation logger
        self.val_generations_logger = ValidationGenerationsLogger()
        self._trainer_loggers = normalize_trainer_loggers(config.trainer.get("logger"))

    # ------------------------------------------------------------------
    #  Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        # Single ActorRollout worker group (critic-free, no separate ref)
        ar_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        ar_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )

        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        self.actor_rollout_wg = self.ray_worker_group_cls(
            resource_pool=ar_pool,
            ray_cls_with_init=ar_cls,
            device_name=self.device_name,
            **wg_kwargs,
        )
        self.actor_rollout_wg.init_model()

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------

    def fit(self):
        print("=" * 60)
        print("HSPO Training Started")
        print("=" * 60)

        config_flat = OmegaConf.to_container(self.config, resolve=True)
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self._trainer_loggers,
            config=config_flat,
        )

        try:
            # Optional pre-train validation for parity with GiGPO/HiPER.
            if self.config.trainer.get("val_before_train", True):
                init_val = self._validate(0)
                init_payload = _metrics_dict_to_floats(init_val)
                init_payload["training/epoch"] = 0.0
                init_payload["training/global_step"] = 0.0
                init_payload["hspo/phase_index"] = {
                    TrainingPhase.WARMUP: 0.0,
                    TrainingPhase.EXECUTOR: 1.0,
                    TrainingPhase.JOINT: 2.0,
                }[self.phase]
                tracking.log(data=init_payload, step=0)
                if self.config.trainer.get("val_only", False):
                    return

            progress_bar = tqdm(total=self.total_training_steps, desc="Training Progress")
            train_iter = iter(self.train_dataloader)

            for global_step in range(1, self.total_training_steps + 1):
                try:
                    batch_dict = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    batch_dict = next(train_iter)

                epoch = (global_step - 1) // max(1, len(self.train_dataloader))
                self.current_epoch = epoch
                self._update_phase(epoch)

                metrics = self._train_step(batch_dict)

                # Step-based validation/checkpointing, consistent with GiGPO usage.
                test_freq = int(self.config.trainer.get("test_freq", 0) or 0)
                save_freq = int(self.config.trainer.get("save_freq", 0) or 0)
                is_last = global_step >= self.total_training_steps
                if test_freq > 0 and (is_last or global_step % test_freq == 0):
                    metrics.update(self._validate(global_step))
                if save_freq > 0 and (is_last or global_step % save_freq == 0):
                    self._save_checkpoint(global_step)

                log_payload = _metrics_dict_to_floats(metrics)
                log_payload["training/epoch"] = float(epoch)
                log_payload["training/global_step"] = float(global_step)
                log_payload["hspo/phase_index"] = {
                    TrainingPhase.WARMUP: 0.0,
                    TrainingPhase.EXECUTOR: 1.0,
                    TrainingPhase.JOINT: 2.0,
                }[self.phase]
                tracking.log(data=log_payload, step=global_step)
                progress_bar.update(1)

            progress_bar.close()
        finally:
            finish_tracking_safe(tracking)

    def _update_phase(self, epoch: int):
        warmup = self.config.algorithm.hspo.get("warmup_epochs", 10)
        executor_epochs = self.config.algorithm.hspo.get("executor_epochs", 100)

        if epoch < warmup:
            self.phase = TrainingPhase.WARMUP
        elif epoch < warmup + executor_epochs:
            self.phase = TrainingPhase.EXECUTOR
        else:
            self.phase = TrainingPhase.JOINT

    # ------------------------------------------------------------------
    #  One training epoch
    # ------------------------------------------------------------------

    def _train_step(self, batch_dict: Dict) -> Dict:
        phase = self.phase

        # 1. Collect rollouts
        with Timer(name="rollout", text="Rollout time: {:.2f}s"):
            gen_batch = DataProto.from_single_dict(batch_dict)
            rollout_data = self._collect_rollouts(gen_batch=gen_batch, is_train=True)

        if rollout_data is None:
            print("[WARNING] No rollout data collected")
            return {}

        # 2. Compute rewards
        with Timer(name="reward", text="Reward time: {:.2f}s"):
            rollout_data = self._compute_rewards(rollout_data, is_train=True)

        # 3. Get reference log-probs (needed for KL reward penalty or KL loss)
        if self._has_ref and (self.config.algorithm.use_kl_in_reward or self.config.actor_rollout_ref.actor.use_kl_loss):
            with Timer(name="ref", text="Ref log-prob time: {:.2f}s"):
                rollout_data = self._compute_ref_log_probs(rollout_data)

        # 4. Get old log-probs from actor
        with Timer(name="old_log_prob", text="Old log-prob time: {:.2f}s"):
            old_log_prob_output = self.actor_rollout_wg.compute_log_prob(rollout_data)
            rollout_data.batch["old_log_probs"] = old_log_prob_output.batch["old_log_probs"]

        # 5. Apply KL penalty (only if ref model is available)
        if self._has_ref:
            rollout_data, kl_metrics = apply_kl_penalty(
                rollout_data, self.kl_ctrl,
                kl_penalty=self.config.algorithm.get("kl_penalty", "kl"),
            )
        else:
            rollout_data.batch["token_level_rewards"] = rollout_data.batch["token_level_scores"]
            kl_metrics = {}

        # 6. Compute HSPO advantages
        with Timer(name="advantage", text="Advantage time: {:.2f}s"):
            rollout_data = self._compute_advantages(rollout_data)

        # 7. PPO update
        with Timer(name="update", text="Update time: {:.2f}s"):
            loss_metrics = self._ppo_update(rollout_data, phase)

        # GiGPO-style dense scalars (sequence / episode stats).
        data_metrics = {}
        try:
            data_metrics = compute_data_metrics(batch=rollout_data, use_critic=False)
        except Exception:
            data_metrics = {}
        rollout_metrics = _extract_rollout_metrics(rollout_data)

        return {**kl_metrics, **loss_metrics, **data_metrics, **rollout_metrics}

    # ------------------------------------------------------------------
    #  Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self, gen_batch: DataProto, is_train: bool = True) -> Optional[DataProto]:
        envs = self.envs if is_train else self.val_envs
        if envs is None:
            return None

        gen_batch_output = self.traj_collector.multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=self.actor_rollout_wg,
            envs=envs,
            is_train=is_train,
        )
        return gen_batch_output

    # ------------------------------------------------------------------
    #  Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(self, data: DataProto, is_train: bool = True) -> DataProto:
        reward_fn = self.reward_fn if is_train else self.val_reward_fn
        reward_tensor, _ = compute_reward(data, reward_fn)
        data.batch["token_level_scores"] = reward_tensor
        return data

    # ------------------------------------------------------------------
    #  Reference log-probabilities
    # ------------------------------------------------------------------

    def _compute_ref_log_probs(self, data: DataProto) -> DataProto:
        ref_output = self.actor_rollout_wg.compute_ref_log_prob(data)
        data.batch["ref_log_prob"] = ref_output.batch["ref_log_prob"]
        return data

    # ------------------------------------------------------------------
    #  HSPO advantage computation
    # ------------------------------------------------------------------

    def _compute_advantages(self, data: DataProto) -> DataProto:
        data.batch["response_mask"] = compute_response_mask(data)

        data = core_hspo.compute_hspo_advantage(
            data,
            self.hspo_cfg,
            self.tokenizer,
            self.milestone_scorer,
            self.config.env.env_name,
        )
        return data

    # ------------------------------------------------------------------
    #  PPO update step
    # ------------------------------------------------------------------

    def _ppo_update(self, data: DataProto, phase: TrainingPhase) -> Dict:
        """Run one PPO update with HSPO token-level credit routing.

        HSPO loss is computed inside the worker so gradients flow from
        fresh log-probs through the model.
        """
        data.batch["response_mask"] = compute_response_mask(data)

        use_kl = (
            self.config.actor_rollout_ref.actor.use_kl_loss
            or self.config.algorithm.use_kl_in_reward
        )

        data.meta_info["hspo_cfg_dict"] = asdict(self.hspo_cfg)
        data.meta_info["hspo_phase"] = phase.value
        data.meta_info["hspo_use_kl"] = use_kl
        output = self.actor_rollout_wg.update_actor_hspo(data)
        return output.meta_info["metrics"]

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _maybe_log_val_generations(self, val_output: Optional[DataProto], step: int) -> None:
        """Log a small validation table to wandb/swanlab/… (`verl` ``ValidationGenerationsLogger`` API)."""
        generations_to_log = int(self.config.trainer.get("log_val_generations", 0))
        if generations_to_log <= 0 or val_output is None or len(val_output) == 0:
            return

        samples: List[tuple] = []
        for i in range(len(val_output)):
            di = val_output[i]
            try:
                prompt_ids = di.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = di.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = di.batch["responses"]
                valid_response_length = di.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                prompt_str = self.tokenizer.decode(
                    valid_prompt_ids, skip_special_tokens=False,
                )
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=False,
                )
                er = di.non_tensor_batch["episode_rewards"]
                score = float(np.asarray(er).reshape(()))
                samples.append((prompt_str, response_str, score))
            except Exception:
                continue

        if not samples:
            return

        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.val_generations_logger.log(self._trainer_loggers, samples, step)

    def _validate(self, epoch: int) -> Dict:
        if self.val_envs is None:
            return {}

        print(f"Validating epoch {epoch} ...")

        old_envs = self.envs
        self.envs = self.val_envs

        try:
            batch_dict = next(iter(self.val_dataloader))
            gen_batch = DataProto.from_single_dict(batch_dict)
            val_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
        finally:
            self.envs = old_envs

        metrics = {}
        if val_output is not None:
            # AlfredWorld attaches overall + per-task-type rates (pick_and_place_*, …);
            # GiGO/ppo logs these as episode/* via compute_data_metrics — mirror for val/KPIs.
            for k, arr in val_output.non_tensor_batch.items():
                if "success_rate" not in str(k):
                    continue
                try:
                    m = float(np.mean(np.asarray(arr, dtype=np.float64)))
                    safe = str(k).replace("/", "_")
                    metrics[f"val/{safe}_mean"] = m
                except Exception:
                    continue

            er = val_output.non_tensor_batch.get("episode_rewards")
            if er is not None:
                try:
                    metrics["val/episode_reward_mean"] = float(
                        np.mean(np.asarray(er, dtype=np.float64))
                    )
                except Exception:
                    pass

            el = val_output.non_tensor_batch.get("episode_lengths")
            if el is not None:
                try:
                    metrics["val/episode_length_mean"] = float(
                        np.mean(np.asarray(el, dtype=np.float64))
                    )
                except Exception:
                    pass

            self._maybe_log_val_generations(val_output, epoch)

        print(f"Validation epoch {epoch}:")
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if isinstance(v, numbers.Real) and not isinstance(v, bool):
                print(f"  {k}: {float(v):.6g}")
            else:
                print(f"  {k}: {v}")
        task_splits = [
            k
            for k in metrics
            if k.startswith("val/") and k.endswith("_mean") and "success_rate" in k and k != "val/success_rate_mean"
        ]
        if val_output is not None and metrics and not task_splits:
            avail = sorted(
                nk for nk in val_output.non_tensor_batch if "success_rate" in str(nk)
            )
            print(
                "  (未找到六类任务拆分指标；non_tensor_batch 中含 success_rate 的键: "
                f"{avail}。若仅有 success_rate，请确认 reset 时 info 含 extra.gamefile。)"
            )
        return metrics

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, global_step: int):
        ckpt_dir = os.path.join(
            self.config.trainer.get("ckpt_dir", "./checkpoints"),
            f"global_step_{global_step}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(os.path.join(ckpt_dir, "actor"))
        print(f"Checkpoint saved to {ckpt_dir}")
