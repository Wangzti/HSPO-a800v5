# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0
"""Small HSPO token-mask helpers used by tests and diagnostics."""

from __future__ import annotations

from typing import Literal, Optional

import torch


Phase = Literal["low_level", "high_level", "joint"]


def build_phase_response_loss_mask(
    action_mask: torch.Tensor,
    subgoal_mask: torch.Tensor,
    switch_mask: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    phase: Phase = "low_level",
) -> torch.Tensor:
    """
    Build the response-local PPO loss mask for an HSPO phase.

    SWITCH tokens are intentionally excluded from PPO phase masks. They are
    reserved for a separate supervised termination objective.
    """
    if response_mask is None:
        response_mask = torch.ones_like(action_mask, dtype=torch.bool)
    response_mask = response_mask.bool()
    action_mask = action_mask.bool() & response_mask
    subgoal_mask = subgoal_mask.bool() & response_mask

    if phase == "low_level":
        return action_mask
    if phase == "high_level":
        return subgoal_mask
    if phase == "joint":
        return (action_mask | subgoal_mask) & response_mask
    raise ValueError(f"Unsupported HSPO phase: {phase}")


def build_decoupled_advantages(
    action_mask: torch.Tensor,
    subgoal_mask: torch.Tensor,
    switch_mask: torch.Tensor,
    low_advantage: float,
    high_advantage: float,
) -> torch.Tensor:
    """Assign A_L only to action tokens and A_H only to subgoal tokens."""
    adv = torch.zeros_like(action_mask, dtype=torch.float32)
    adv = torch.where(action_mask.bool(), torch.full_like(adv, float(low_advantage)), adv)
    adv = torch.where(subgoal_mask.bool(), torch.full_like(adv, float(high_advantage)), adv)
    # switch_mask is accepted to make the invariant explicit in tests.
    adv = torch.where(switch_mask.bool(), torch.zeros_like(adv), adv)
    return adv
