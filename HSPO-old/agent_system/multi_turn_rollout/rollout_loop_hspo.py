# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO Trajectory Collector (rollout loop).

Extends the HiPER TrajectoryCollector with three HSPO-specific behaviours:

1. Token-mask computation – after each LLM call, PlanExecuteParser + TokenMaskBuilder
   tag every response token as belonging to <switch>, <subgoal>, or <action>.

2. PRM-based low-level reward – AlfworldRulePRM (or any PRMBase subclass) is
   called after each env.step() to produce r_t^L.

3. Process λ-return – at each segment boundary (switch=SWITCH, PRM done > τ,
   or max_segment_len reached), compute the segment-level λ-return and write it
   back into all steps of that segment.

The resulting DataProto has extra non_tensor_batch keys used by HSPORewardManager:
    low_rewards       – r_t^L (already computed; reward manager just places them)
    action_mask_end   – index of last action token in the response
    subgoal_mask_end  – index of last subgoal token (for macro reward placement)
    switch_mask_end   – index of last switch token
    switch_target     – "SWITCH" | "KEEP" (for supervised switch CE loss)
    macro_rewards     – R_k^H (terminal + side + redundancy; zero unless episode end)
    advantages_low    – process λ-return for action tokens (put into 'advantages' for PPO)
    phase             – training phase string
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, filter_group_data

from hspo.config import HSPOConfig
from hspo.parser import PlanExecuteParser
from hspo.token_mask import TokenMaskBuilder
from hspo.advantages import compute_process_return, compute_macro_gae, compute_value_targets
from hspo.prm.alfworld_prm import AlfworldRulePRM
from hspo.prm.base import PRMBase, PRMOutput
from hspo.env_state import parse_alfworld_state, build_task_meta


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _last_true_idx(mask: torch.Tensor) -> int:
    """Return index of the last True element, or -1 if none."""
    idxs = mask.nonzero(as_tuple=False)
    return int(idxs[-1, 0]) if len(idxs) > 0 else -1


def _copy_mask_to_response(mask: torch.Tensor, response_length: int) -> torch.Tensor:
    """Right-pad/truncate a response-local mask to the rollout response length."""
    out = torch.zeros(response_length, dtype=torch.bool)
    if mask is None:
        return out
    n = min(int(mask.numel()), response_length)
    if n > 0:
        out[:n] = mask[:n].bool()
    return out


def _canonicalise_subgoal_type(subgoal_text: str) -> str:
    """
    Heuristic: map NL subgoal to canonical type.
    Overridden by environment-provided subgoal_type when available.
    """
    sl = subgoal_text.lower()
    if "find" in sl or "locate" in sl:
        return "FIND_OBJECT"
    if "pick" in sl:
        return "PICK_OBJECT"
    if "clean" in sl:
        return "CLEAN_OBJECT"
    if "heat" in sl or "microwave" in sl:
        return "HEAT_OBJECT"
    if "cool" in sl or "fridge" in sl:
        return "COOL_OBJECT"
    if "place" in sl or "put" in sl:
        return "PLACE_OBJECT"
    if "go to" in sl and ("sink" in sl or "basin" in sl):
        return "GO_TO_TOOL"
    if "go to" in sl and ("fridge" in sl or "refrigerator" in sl):
        return "GO_TO_TOOL"
    if "go to" in sl and "microwave" in sl:
        return "GO_TO_TOOL"
    if "go to" in sl:
        return "GO_TO_RECEPTACLE"
    if "examine" in sl:
        return "EXAMINE_OBJECT"
    return "FIND_OBJECT"  # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Segment tracker (maintains state for one environment across steps)
# ─────────────────────────────────────────────────────────────────────────────

class _SegmentTracker:
    """Tracks the current subgoal segment for one environment instance."""

    def __init__(self, hspo_cfg: HSPOConfig, prm: PRMBase) -> None:
        self.cfg = hspo_cfg
        self.prm = prm
        self.reset()

    def reset(self) -> None:
        self.current_subgoal: str = ""
        self.current_subgoal_type: str = ""
        self.segment_low_rewards: List[float] = []
        self.segment_step_indices: List[int] = []   # global step indices
        self.segment_len: int = 0
        self.task_meta: Dict[str, Any] = {}
        self.state_segment_start: Dict[str, Any] = {}
        self.previous_subgoals: List[str] = []
        # SWITCH is supervised at the *next* model call after a subgoal is
        # completed. The first call must create an initial subgoal.
        self.next_switch_target: str = "SWITCH"

    def update_task_meta(self, info: Dict) -> None:
        """Extract task_meta from env info dict."""
        self.task_meta = {
            "target_obj":         info.get("target_obj", info.get("object", "")),
            "target_obj2":        info.get("target_obj2", ""),
            "target_receptacle":  info.get("target_receptacle", info.get("receptacle", "")),
            "task_type":          info.get("task_type", ""),
            "requires_clean":     info.get("task_type", "") in ("clean", "3"),
            "requires_heat":      info.get("task_type", "") in ("heat", "4"),
            "requires_cool":      info.get("task_type", "") in ("cool", "5"),
        }

    def on_switch(self, new_subgoal: str, state: Dict) -> Tuple[List[float], List[int]]:
        """Called when SWITCH is detected. Returns (old_rewards, old_step_indices)."""
        old_rewards = list(self.segment_low_rewards)
        old_indices = list(self.segment_step_indices)
        if self.current_subgoal:
            self.previous_subgoals.append(self.current_subgoal)
        self.current_subgoal = new_subgoal
        self.current_subgoal_type = _canonicalise_subgoal_type(new_subgoal)
        self.segment_low_rewards = []
        self.segment_step_indices = []
        self.segment_len = 0
        self.state_segment_start = dict(state)
        self.next_switch_target = "KEEP"
        return old_rewards, old_indices

    def add_step(
        self,
        global_step_idx: int,
        state_before: Dict,
        state_after: Dict,
        action: str,
        is_valid: bool,
    ) -> Tuple[float, Optional[PRMOutput]]:
        """Compute PRM reward for this step and accumulate into segment."""
        prm_out: Optional[PRMOutput] = None
        if not self.current_subgoal:
            # No subgoal yet (first step before first SWITCH)
            r = 0.0
        else:
            prm_out = self.prm.score(
                subgoal_type=self.current_subgoal_type,
                subgoal_text=self.current_subgoal,
                state_before=state_before,
                state_after=state_after,
                action=action,
                task_meta=self.task_meta,
            )
            # Override validity from env signal if available
            if not is_valid:
                prm_out.valid = 0.0
            r = prm_out.low_level_reward(
                eta_done=self.cfg.eta_done,
                tau_done=self.cfg.tau_done,
                lambda_invalid=self.cfg.lambda_invalid,
                lambda_side=self.cfg.lambda_side,
                lambda_step=self.cfg.lambda_step,
            )

        self.segment_low_rewards.append(r)
        self.segment_step_indices.append(global_step_idx)
        self.segment_len += 1
        return r, prm_out

    def segment_advantages(self) -> List[float]:
        """Compute process λ-return for the current segment."""
        if not self.segment_low_rewards:
            return []
        adv = compute_process_return(
            self.segment_low_rewards,
            gamma=self.cfg.gamma_low,
            lam=self.cfg.lam_low,
            normalise=(len(self.segment_low_rewards) > 1),
        )
        return adv.tolist()

    def redundancy_penalty(self, subgoal: str) -> float:
        """Penalise repetition of a recently completed subgoal (string match)."""
        if not subgoal:
            return 0.0
        sg_l = subgoal.lower()
        # Check last 3 subgoals
        for prev in self.previous_subgoals[-3:]:
            if prev.lower() == sg_l:
                return self.cfg.eta_redundancy
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main HSPO TrajectoryCollector
# ─────────────────────────────────────────────────────────────────────────────

class HSPOTrajectoryCollector:
    """
    HSPO trajectory collector.  Drop-in replacement for HiPER's TrajectoryCollector
    with PRM-based reward shaping and token-level credit masks.

    Parameters
    ----------
    config       : Hydra config (same as HiPER; algorithm.hspo must be present)
    tokenizer    : HF tokenizer
    processor    : multimodal processor (None for text-only)
    prm          : PRMBase instance (defaults to AlfworldRulePRM)
    """

    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
        processor=None,
        prm: Optional[PRMBase] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        # HSPO config (injected as config.algorithm.hspo or default)
        from omegaconf import OmegaConf
        from dataclasses import asdict

        if hasattr(config.algorithm, "hspo"):
            cfg_dict = OmegaConf.to_container(config.algorithm.hspo, resolve=True)
            self.hspo_cfg = HSPOConfig(**cfg_dict)
        else:
            self.hspo_cfg = HSPOConfig()

        self.prm = prm if prm is not None else AlfworldRulePRM()
        self.parser = PlanExecuteParser()
        self.mask_builder = TokenMaskBuilder(tokenizer)

    # ------------------------------------------------------------------ #
    # Observation preprocessing (reused from HiPER; text-only version)    #
    # ------------------------------------------------------------------ #

    def preprocess_single_sample(self, item: int, gen_batch: DataProto, obs: Dict) -> Dict:
        raw_prompt = gen_batch.non_tensor_batch["raw_prompt"][item]
        obs_text = obs.get("text", [None])[item] if obs.get("text") else None

        obs_content = obs_text or ""
        import numpy as np_mod
        chat = np_mod.array([{"content": obs_content, "role": "user"}])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:]

        row_dict = {
            "input_ids":      input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids":   position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "anchor_obs":     None,
            "index":          item,
            "data_source":    gen_batch.non_tensor_batch.get("data_source", [""] * len(gen_batch.batch))[item],
        }
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()
        return row_dict

    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        batch_size = len(gen_batch.batch["input_ids"])
        processed = [self.preprocess_single_sample(i, gen_batch, obs) for i in range(batch_size)]
        batch = collate_fn(processed)
        return DataProto.from_single_dict(data=batch, meta_info=gen_batch.meta_info)

    # ------------------------------------------------------------------ #
    # Assemble DataProto from collected steps                              #
    # ------------------------------------------------------------------ #

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        success_rate = {k: np.mean(v) for k, v in success.items()}
        effective_batch = []
        for bs, steps in enumerate(total_batch_list):
            for data in steps:
                if not data.get("active_masks", False):
                    continue
                assert traj_uid[bs] == data["traj_uid"]
                data["episode_rewards"] = episode_rewards[bs]
                data["episode_lengths"] = episode_lengths[bs]
                data["tool_callings"] = tool_callings[bs]
                for k, v in success_rate.items():
                    data[k] = v
                effective_batch.append(data)
        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    # ------------------------------------------------------------------ #
    # HSPO multi-turn loop                                                 #
    # ------------------------------------------------------------------ #

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ) -> Tuple:
        batch_size = len(gen_batch.batch)
        phase = self.hspo_cfg.phase

        # Reset envs and segment trackers
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))

        trackers = [
            _SegmentTracker(self.hspo_cfg, self.prm) for _ in range(batch_size)
        ]
        for i, info in enumerate(infos):
            trackers[i].update_task_meta(info)

        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        uid_batch = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)

        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        # Parse initial obs text → structured state for PRM
        obs_texts_init = obs.get("text", [""] * batch_size) if isinstance(obs, dict) else [""] * batch_size
        prev_states: List[Dict] = []
        for i in range(batch_size):
            adm_cmds_i = infos[i].get("admissible_commands", None)
            prev_states.append(parse_alfworld_state(obs_texts_init[i], adm_cmds_i, ""))
            # Supplement tracker task_meta from env's task description if available
            task_desc_i = infos[i].get("task_description", infos[i].get("description", obs_texts_init[i]))
            meta = build_task_meta(task_desc_i)
            for k, v in meta.items():
                if v and not trackers[i].task_meta.get(k):
                    trackers[i].task_meta[k] = v

        # per-env advantage accumulator: maps global_step_idx → advantage_low
        adv_buffers: List[Dict[int, float]] = [{} for _ in range(batch_size)]

        for _step in range(self.config.env.max_steps):
            active_masks = ~is_done

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)

            text_responses = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )

            # ── Parse + build masks ───────────────────────────────────── #
            parse_results = self.parser.parse_batch(text_responses)
            switch_mask_ends = np.full(batch_size, -1, dtype=np.int32)
            subgoal_mask_ends = np.full(batch_size, -1, dtype=np.int32)
            action_mask_ends = np.full(batch_size, -1, dtype=np.int32)
            response_length = int(batch.batch["responses"].shape[-1])
            switch_masks = torch.zeros((batch_size, response_length), dtype=torch.bool)
            subgoal_masks = torch.zeros((batch_size, response_length), dtype=torch.bool)
            action_masks = torch.zeros((batch_size, response_length), dtype=torch.bool)
            switch_targets = np.array([""] * batch_size, dtype=object)
            parsed_switches = np.array(["KEEP"] * batch_size, dtype=object)
            parsed_subgoals = np.array([""] * batch_size, dtype=object)
            parsed_actions = np.array([""] * batch_size, dtype=object)
            format_valid = np.zeros(batch_size, dtype=bool)
            missing_switch = np.zeros(batch_size, dtype=bool)
            missing_subgoal = np.zeros(batch_size, dtype=bool)
            action_parse_ok = np.zeros(batch_size, dtype=bool)
            keep_subgoal_consistent = np.ones(batch_size, dtype=bool)

            for i, (resp_text, pr) in enumerate(zip(text_responses, parse_results)):
                format_valid[i] = bool(pr.valid_format)
                missing_switch[i] = "missing_switch" in pr.errors
                missing_subgoal[i] = "missing_subgoal" in pr.errors
                action_parse_ok[i] = bool(pr.action)
                if pr.valid_format:
                    try:
                        masks = self.mask_builder.build_from_parse_result(resp_text, pr)
                        switch_mask_ends[i] = _last_true_idx(masks["switch_mask"])
                        subgoal_mask_ends[i] = _last_true_idx(masks["subgoal_mask"])
                        action_mask_ends[i] = _last_true_idx(masks["action_mask"])
                        switch_masks[i] = _copy_mask_to_response(masks["switch_mask"], response_length)
                        subgoal_masks[i] = _copy_mask_to_response(masks["subgoal_mask"], response_length)
                        action_masks[i] = _copy_mask_to_response(masks["action_mask"], response_length)
                    except Exception:
                        pass  # masks stay -1 on failure
                    parsed_switches[i] = pr.switch or "KEEP"
                    parsed_subgoals[i] = pr.subgoal or ""
                    parsed_actions[i] = pr.action or ""
                elif pr.action:
                    parsed_actions[i] = pr.action

            # ── Environment step ──────────────────────────────────────── #
            # Pass the full structured response to the environment manager.
            # Its projection extracts the executable action while also storing
            # switch/subgoal in memory for the next prompt's current sub-goal.
            env_actions = list(text_responses)
            next_obs, rewards, dones, infos = envs.step(env_actions)

            rewards_np = torch_to_numpy(rewards) if isinstance(rewards, torch.Tensor) else np.asarray(rewards, dtype=np.float32)
            dones_np = torch_to_numpy(dones) if isinstance(dones, torch.Tensor) else np.asarray(dones, dtype=bool)
            if rewards_np.ndim == 2:
                rewards_np = rewards_np.squeeze(1)
            if dones_np.ndim == 2:
                dones_np = dones_np.squeeze(1)

            is_valid_flags = np.array([info.get("is_action_valid", True) for info in infos], dtype=bool)

            # ── PRM rewards + segment accounting ─────────────────────── #
            low_rewards = np.zeros(batch_size, dtype=np.float32)
            macro_rewards = np.zeros(batch_size, dtype=np.float32)
            prm_done_after = np.zeros(batch_size, dtype=np.float32)
            prm_progress_delta = np.zeros(batch_size, dtype=np.float32)
            switch_target_match = np.zeros(batch_size, dtype=bool)
            premature_switch = np.zeros(batch_size, dtype=bool)
            delayed_switch = np.zeros(batch_size, dtype=bool)

            for i in range(batch_size):
                if not active_masks[i]:
                    continue

                tracker = trackers[i]
                curr_state = prev_states[i]
                switch_targets[i] = tracker.next_switch_target

                # Parse next obs text → structured state (use next_obs for index i)
                next_obs_text = ""
                if isinstance(next_obs, dict) and "text" in next_obs:
                    next_obs_text = next_obs["text"][i]
                elif isinstance(next_obs, (list, np.ndarray)) and i < len(next_obs):
                    next_obs_text = str(next_obs[i])
                feedback_i = infos[i].get("feedback", "")
                adm_cmds_i = infos[i].get("admissible_commands", None)
                next_state = parse_alfworld_state(next_obs_text, adm_cmds_i, feedback_i)

                sw = parsed_switches[i]
                subgoal_from_parse = parse_results[i].subgoal or tracker.current_subgoal
                previous_subgoal = tracker.current_subgoal

                # Handle SWITCH: finalise old segment, start new one
                if sw == "SWITCH" or not tracker.current_subgoal:
                    old_rewards, old_indices = tracker.on_switch(subgoal_from_parse, curr_state)
                    # Compute advantages for just-completed segment and write back
                    if old_rewards:
                        old_advs = compute_process_return(
                            old_rewards,
                            gamma=self.hspo_cfg.gamma_low,
                            lam=self.hspo_cfg.lam_low,
                            normalise=(len(old_rewards) > 1),
                        ).tolist()
                        for idx, adv in zip(old_indices, old_advs):
                            adv_buffers[i][idx] = adv

                # Compute PRM step reward and accumulate
                r_low, prm_out = tracker.add_step(
                    global_step_idx=_step,
                    state_before=curr_state,
                    state_after=next_state,
                    action=parsed_actions[i],
                    is_valid=bool(is_valid_flags[i]),
                )
                low_rewards[i] = r_low
                if prm_out is not None:
                    prm_done_after[i] = float(prm_out.done_after)
                    prm_progress_delta[i] = float(prm_out.progress_after - prm_out.progress_before)

                # SWITCH supervision target is derived from subgoal completion
                # after this action, and is applied to the next model call.
                subgoal_done_now = bool(prm_done_after[i] >= self.hspo_cfg.tau_done)
                tracker.next_switch_target = (
                    "SWITCH"
                    if ((not tracker.current_subgoal) or subgoal_done_now or bool(dones_np[i]))
                    else "KEEP"
                )
                switch_target_match[i] = (sw == switch_targets[i])
                premature_switch[i] = (sw == "SWITCH" and switch_targets[i] == "KEEP")
                delayed_switch[i] = (sw == "KEEP" and switch_targets[i] == "SWITCH")
                if sw == "KEEP" and previous_subgoal:
                    keep_subgoal_consistent[i] = (
                        str(parsed_subgoals[i]).strip().lower()
                        == str(previous_subgoal).strip().lower()
                    )

                # Macro reward: only non-zero at episode termination
                if dones_np[i]:
                    task_success = float(rewards_np[i] > 0.5)
                    side_p = tracker.redundancy_penalty(subgoal_from_parse)
                    macro_rewards[i] = task_success - side_p
                    # Finalise last segment advantages
                    seg_advs = tracker.segment_advantages()
                    for idx, adv in zip(tracker.segment_step_indices, seg_advs):
                        adv_buffers[i][idx] = adv

                prev_states[i] = dict(next_state)
                tracker.update_task_meta(infos[i])

                # Force-switch at max_segment_len
                if tracker.segment_len >= self.hspo_cfg.max_segment_len:
                    seg_advs = tracker.segment_advantages()
                    seg_indices = list(tracker.segment_step_indices)
                    for idx, adv in zip(seg_indices, seg_advs):
                        adv_buffers[i][idx] = adv
                    tracker.on_switch(tracker.current_subgoal, next_state)  # continue same subgoal

            # ── Assemble step data for DataProto ─────────────────────── #
            episode_rewards[active_masks] += rewards_np[active_masks]
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch["rewards"]         = rewards_np.astype(object)
            batch.non_tensor_batch["active_masks"]    = active_masks.astype(object)
            batch.non_tensor_batch["dones"]           = dones_np.astype(object)
            batch.non_tensor_batch["turn_idx"]        = np.full(batch_size, _step, dtype=np.int32)
            batch.non_tensor_batch["low_rewards"]     = low_rewards.astype(object)
            batch.non_tensor_batch["macro_rewards"]   = macro_rewards.astype(object)
            batch.non_tensor_batch["action_mask_end"] = action_mask_ends.astype(object)
            batch.non_tensor_batch["subgoal_mask_end"] = subgoal_mask_ends.astype(object)
            batch.non_tensor_batch["switch_mask_end"] = switch_mask_ends.astype(object)
            batch.non_tensor_batch["switch_target"]   = switch_targets
            batch.non_tensor_batch["format_valid"]    = format_valid.astype(object)
            batch.non_tensor_batch["missing_switch"]  = missing_switch.astype(object)
            batch.non_tensor_batch["missing_subgoal"] = missing_subgoal.astype(object)
            batch.non_tensor_batch["action_parse_ok"] = action_parse_ok.astype(object)
            batch.non_tensor_batch["keep_subgoal_consistent"] = keep_subgoal_consistent.astype(object)
            batch.non_tensor_batch["prm_done_after"]  = prm_done_after.astype(object)
            batch.non_tensor_batch["prm_progress_delta"] = prm_progress_delta.astype(object)
            batch.non_tensor_batch["switch_target_match"] = switch_target_match.astype(object)
            batch.non_tensor_batch["premature_switch"] = premature_switch.astype(object)
            batch.non_tensor_batch["delayed_switch"] = delayed_switch.astype(object)
            batch.non_tensor_batch["phase"]           = np.array([phase] * batch_size, dtype=object)
            batch.batch["switch_mask"] = switch_masks.to(batch.batch["responses"].device)
            batch.batch["subgoal_mask"] = subgoal_masks.to(batch.batch["responses"].device)
            batch.batch["action_mask"] = action_masks.to(batch.batch["responses"].device)
            # advantage_low placeholder — filled in second pass below
            batch.batch["advantage_low"] = torch.zeros(batch_size, dtype=torch.float32)

            batch_list = to_list_of_dict(batch)
            for i in range(batch_size):
                batch_list[i]["_global_step_idx"] = _step
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones_np)
            obs = next_obs

            if is_done.all():
                break

        # Second pass: flush remaining open segments and write back advantages
        for i in range(batch_size):
            tracker = trackers[i]
            if tracker.segment_low_rewards:
                seg_advs = tracker.segment_advantages()
                for idx, adv in zip(tracker.segment_step_indices, seg_advs):
                    adv_buffers[i][idx] = adv

        # Back-fill advantage_low for every collected step
        for i in range(batch_size):
            for step_data in total_batch_list[i]:
                step_idx = step_data.pop("_global_step_idx", None)
                if step_idx is not None:
                    step_data["advantage_low"] = adv_buffers[i].get(step_idx, 0.0)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        total_batch_list, ep_rews, ep_lens, success, traj_uid, tool_callings = \
            self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=ep_rews,
            episode_lengths=ep_lens,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
