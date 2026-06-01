# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Abstract base class for HSPO Process Reward Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PRMOutput:
    """Standardised PRM signals for one action step."""
    progress_before: float    # progress toward subgoal before action
    progress_after: float     # progress toward subgoal after action
    done_after: float         # P(subgoal done) after action  (0.0 or 1.0 for rules)
    valid: float              # 1.0 if action was environmentally valid, else 0.0
    side_effect_before: float # side-effect cost before action
    side_effect_after: float  # side-effect cost after action

    # Convenience: scalar low-level reward (computed by reward manager)
    def low_level_reward(
        self,
        eta_done: float = 1.0,
        tau_done: float = 0.9,
        lambda_invalid: float = 1.0,
        lambda_side: float = 0.5,
        lambda_step: float = 0.01,
    ) -> float:
        r = self.progress_after - self.progress_before
        r += eta_done * float(self.done_after >= tau_done)
        r -= lambda_invalid * float(self.valid < 0.5)
        r -= lambda_side * max(0.0, self.side_effect_after - self.side_effect_before)
        r -= lambda_step
        return r


class PRMBase(ABC):
    """Abstract interface all rule-PRMs must implement."""

    @abstractmethod
    def score(
        self,
        subgoal_type: str,
        subgoal_text: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        action: str,
        task_meta: Dict[str, Any],
    ) -> PRMOutput:
        """
        Evaluate one action step w.r.t. a subgoal.

        Parameters
        ----------
        subgoal_type   : canonical type string (e.g. "CLEAN_OBJECT")
        subgoal_text   : NL description of the current subgoal
        state_before   : environment state dict *before* action
        state_after    : environment state dict *after* action
        action         : action string taken
        task_meta      : task-level metadata (target object, receptacle, etc.)
        """
        ...
