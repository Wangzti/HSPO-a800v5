# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Unit tests for HSPO advantage computation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from hspo.advantages import compute_process_return, compute_macro_gae, compute_value_targets


class TestProcessReturn:

    def test_single_step(self):
        adv = compute_process_return([1.0], gamma=0.95, lam=0.9, normalise=False)
        assert adv.shape == (1,)
        assert adv[0].item() == pytest.approx(1.0)

    def test_multi_step_discounting(self):
        rewards = [1.0, 0.0, 0.0]
        adv = compute_process_return(rewards, gamma=1.0, lam=1.0, normalise=False)
        # All discounted into first step: A[0] = 1+0+0 = 1.0
        assert adv[0].item() == pytest.approx(1.0)
        assert adv[1].item() == pytest.approx(0.0)
        assert adv[2].item() == pytest.approx(0.0)

    def test_discount_factor(self):
        rewards = [0.0, 1.0]
        gl = 0.95 * 0.9
        adv = compute_process_return(rewards, gamma=0.95, lam=0.9, normalise=False)
        # A[0] = 0 + gl*1 = gl
        assert adv[0].item() == pytest.approx(gl)
        # A[1] = 1
        assert adv[1].item() == pytest.approx(1.0)

    def test_normalise_zero_mean(self):
        rewards = [1.0, 2.0, 3.0]
        adv = compute_process_return(rewards, normalise=True)
        assert abs(adv.mean().item()) < 1e-5

    def test_returns_tensor(self):
        adv = compute_process_return([0.5, 0.5], normalise=False)
        assert isinstance(adv, torch.Tensor)

    def test_tensor_input(self):
        rewards = torch.tensor([1.0, 0.5, 0.0])
        adv = compute_process_return(rewards, normalise=False)
        assert adv.shape == (3,)


class TestMacroGAE:

    def test_single_macro_step(self):
        adv = compute_macro_gae([1.0], [0.5], [0.0], gamma=0.95, lam=0.95, normalise=False)
        # delta = 1.0 + 0.95*0.0 - 0.5 = 0.5
        assert adv.shape == (1,)
        assert adv[0].item() == pytest.approx(0.5)

    def test_multi_step_gae(self):
        rewards = [0.0, 1.0]
        values = [0.0, 0.0]
        next_values = [0.0, 0.0]
        adv = compute_macro_gae(rewards, values, next_values, gamma=1.0, lam=1.0, normalise=False)
        # delta[1] = 1.0, delta[0] = 0.0 + 1.0*(delta[1]) = 1.0
        assert adv[1].item() == pytest.approx(1.0)
        assert adv[0].item() == pytest.approx(1.0)

    def test_normalise(self):
        rewards = [1.0, 0.5, 0.0]
        values = [0.5, 0.3, 0.0]
        next_values = [0.3, 0.0, 0.0]
        adv = compute_macro_gae(rewards, values, next_values, normalise=True)
        assert abs(adv.mean().item()) < 1e-5


class TestValueTargets:

    def test_td0_targets(self):
        targets = compute_value_targets([1.0, 0.0], [0.5, 0.0], gamma=0.95)
        # targets[0] = 1.0 + 0.95*0.5 = 1.475
        assert targets[0].item() == pytest.approx(1.475)
        # targets[1] = 0.0 + 0.95*0.0 = 0.0
        assert targets[1].item() == pytest.approx(0.0)
