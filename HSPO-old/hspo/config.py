# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPOConfig – single source of truth for all HSPO hyper-parameters.

The dataclass is intentionally flat so it can be serialised directly to/from
OmegaConf, then injected into ``config.algorithm.hspo`` inside main_ppo_hspo.py.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HSPOConfig:
    # ------------------------------------------------------------------ #
    # Feature flag — when False the trainer falls back to HiPER/HAE       #
    # ------------------------------------------------------------------ #
    enabled: bool = False

    # ------------------------------------------------------------------ #
    # Training curriculum phase                                            #
    # ------------------------------------------------------------------ #
    # "sft"        – SFT warm-up only, no RL gradient                     #
    # "low_level"  – freeze planner, train executor with PRM reward        #
    # "high_level" – freeze executor, train planner with macro-PPO         #
    # "joint"      – alternate executor / planner updates                  #
    phase: Literal["sft", "low_level", "high_level", "joint"] = "low_level"

    # ------------------------------------------------------------------ #
    # Low-level executor parameters                                        #
    # ------------------------------------------------------------------ #
    # Process λ-return (no critic)
    gamma_low: float = 0.95
    lam_low: float = 0.90

    # PPO clip for executor
    clip_ratio_low: float = 0.2

    # Max actions inside one subgoal segment before forced SWITCH
    max_segment_len: int = 8

    # ------------------------------------------------------------------ #
    # PRM reward shaping coefficients                                      #
    # ------------------------------------------------------------------ #
    eta_done: float = 1.0          # bonus when PRM.done > tau_done
    tau_done: float = 0.9          # threshold to trigger done bonus
    lambda_invalid: float = 1.0    # penalty per invalid action
    lambda_side: float = 0.5       # penalty per unit side-effect increase
    lambda_step: float = 0.01      # small per-step living cost

    # ------------------------------------------------------------------ #
    # SWITCH supervision                                                   #
    # "ce"  – cross-entropy vs {done → SWITCH, else → KEEP}              #
    # "none" – no explicit switch loss (only implicit via action credit)   #
    # ------------------------------------------------------------------ #
    switch_loss_type: Literal["ce", "none"] = "ce"
    switch_loss_coef: float = 0.1

    # ------------------------------------------------------------------ #
    # High-level planner / Macro-PPO parameters                           #
    # ------------------------------------------------------------------ #
    gamma_high: float = 0.95
    lam_high: float = 0.95

    # PPO clip for planner (subgoal tokens)
    clip_ratio_high: float = 0.2

    # High-level macro reward coefficients
    beta_side: float = 0.5         # global side-effect penalty
    eta_redundancy: float = 0.2    # redundant subgoal penalty

    # ------------------------------------------------------------------ #
    # Token-mask loss weights                                              #
    # ------------------------------------------------------------------ #
    # During joint training the total loss is:
    #   L = w_action * L_action + w_subgoal * L_subgoal + w_switch * L_switch
    w_action: float = 1.0
    w_subgoal: float = 1.0
    w_switch: float = 0.1

    # ------------------------------------------------------------------ #
    # Subgoal curriculum mix (fractions must sum to 1.0)                  #
    # ------------------------------------------------------------------ #
    subgoal_mix_expert: float = 0.6
    subgoal_mix_heuristic: float = 0.2
    subgoal_mix_sft_planner: float = 0.1
    subgoal_mix_perturbed: float = 0.1

    def __post_init__(self) -> None:
        valid_phases = {"sft", "low_level", "high_level", "joint"}
        if self.phase not in valid_phases:
            raise ValueError(f"HSPOConfig.phase must be one of {sorted(valid_phases)}, got '{self.phase}'")
        total_mix = (
            self.subgoal_mix_expert
            + self.subgoal_mix_heuristic
            + self.subgoal_mix_sft_planner
            + self.subgoal_mix_perturbed
        )
        if abs(total_mix - 1.0) > 1e-6:
            raise ValueError(f"Subgoal curriculum mix fractions must sum to 1.0, got {total_mix:.4f}")
        for name in ("gamma_low", "lam_low", "gamma_high", "lam_high"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError(f"HSPOConfig.{name} must be in (0, 1], got {v}")
