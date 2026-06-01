# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0
"""Unit tests for HSPO token-level credit isolation."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hspo.masked_loss import build_decoupled_advantages, build_phase_response_loss_mask


def test_phase_masks_exclude_other_credit_channels():
    # token layout: [switch, subgoal, subgoal, action, action]
    switch_mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    subgoal_mask = torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool)
    action_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)

    low_mask = build_phase_response_loss_mask(action_mask, subgoal_mask, switch_mask, phase="low_level")
    high_mask = build_phase_response_loss_mask(action_mask, subgoal_mask, switch_mask, phase="high_level")

    assert torch.equal(low_mask, action_mask)
    assert torch.equal(high_mask, subgoal_mask)
    assert not (low_mask & switch_mask).any()
    assert not (high_mask & switch_mask).any()


def test_toy_credit_conflict_keeps_switch_neutral():
    # A_H = -1 should touch only <subgoal>; A_L = +1 should touch only <action>.
    switch_mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    subgoal_mask = torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool)
    action_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)

    advantages = build_decoupled_advantages(
        action_mask=action_mask,
        subgoal_mask=subgoal_mask,
        switch_mask=switch_mask,
        low_advantage=1.0,
        high_advantage=-1.0,
    )

    assert torch.all(advantages[subgoal_mask] < 0)
    assert torch.all(advantages[action_mask] > 0)
    assert torch.all(advantages[switch_mask] == 0)
