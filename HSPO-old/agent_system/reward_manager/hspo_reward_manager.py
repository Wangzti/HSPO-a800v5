# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO Reward Manager

Reads PRM-enriched trajectory data from the rollout and converts it into
two distinct reward signals:

  Low-level reward  r_t^L  – per-action, based on PRM progress/done/validity
  High-level reward R_k^H  – per-macro-step, terminal + side-effect + redundancy

Both signals must already be stored in DataProto.non_tensor_batch by the
rollout loop (rollout_loop_hspo.py).  This manager only converts them into
the reward_tensor layout expected by the verl PPO trainer.

The reward_tensor has shape (batch, max_resp_len).
  • For low-level steps:  reward placed at the *last action token* position
                         (= last token covered by action_mask).
  • For high-level steps: reward placed at the *last subgoal token* position
                         (= last token covered by subgoal_mask).

The token-level placement is determined by the masks stored in non_tensor_batch.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from verl import DataProto


class HSPORewardManager:
    """
    HSPO reward manager for the verl PPO trainer.

    Expects the following non_tensor_batch keys to be populated by
    rollout_loop_hspo.py:

    Per-step (low-level) keys
    -------------------------
    "low_rewards"     : np.ndarray (float32)  – r_t^L per step
    "action_mask_end" : np.ndarray (int32)    – index of last action token
    "switch_target"   : np.ndarray (str)      – "SWITCH" | "KEEP" | "" (no switch loss)
    "switch_mask_end" : np.ndarray (int32)    – index of last switch token

    Per-episode / macro keys (set once at episode end)
    --------------------------------------------------
    "macro_rewards"   : np.ndarray (float32)  – R_k^H per macro step
    "subgoal_mask_end": np.ndarray (int32)    – last subgoal token per step
    "phase"           : str                   – "low_level" | "high_level" | "joint"
    """

    def __init__(self, tokenizer, num_examine: int = 0, hspo_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.hspo_cfg = hspo_cfg  # HSPOConfig or None

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        phase = data.non_tensor_batch.get("phase", np.array(["low_level"] * len(data)))[0]
        if isinstance(phase, bytes):
            phase = phase.decode()

        for i in range(len(data)):
            item = data[i]

            # ── response validity check ───────────────────────────────── #
            attn = item.batch["attention_mask"]
            prompt_len = item.batch["prompts"].shape[-1]
            valid_resp_len = int(attn[prompt_len:].sum().item())
            if valid_resp_len <= 0:
                continue

            # ── low-level reward (placed at last action token) ─────────── #
            low_reward = float(item.non_tensor_batch.get("low_rewards", 0.0))
            action_end = int(item.non_tensor_batch.get("action_mask_end", valid_resp_len - 1))
            action_end = min(action_end, valid_resp_len - 1)

            # ── high-level / macro reward (placed at last subgoal token) ─ #
            macro_reward = float(item.non_tensor_batch.get("macro_rewards", 0.0))
            subgoal_end = int(item.non_tensor_batch.get("subgoal_mask_end", -1))

            if phase == "low_level":
                # Only place low-level reward
                reward_tensor[i, action_end] = low_reward
            elif phase == "high_level":
                # Only place macro reward
                if subgoal_end >= 0:
                    subgoal_end = min(subgoal_end, valid_resp_len - 1)
                    reward_tensor[i, subgoal_end] = macro_reward
            else:  # joint
                reward_tensor[i, action_end] += low_reward
                if subgoal_end >= 0:
                    subgoal_end = min(subgoal_end, valid_resp_len - 1)
                    reward_tensor[i, subgoal_end] += macro_reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
