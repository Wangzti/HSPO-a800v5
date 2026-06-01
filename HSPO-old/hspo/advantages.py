# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO Advantage Computation

Two advantage estimators:

1. compute_process_return  – critic-free λ-return for the low-level executor.
   A_t^L = Σ_{l=t}^{T_seg-1} (γ_L · λ_L)^{l-t} · r_l^L
   Batch-normalised within each segment (μ=0, σ=1).

2. compute_macro_gae  – GAE for the high-level planner (macro critic V_H).
   δ_k = R_k^H + γ_H · V_H(s_{b_{k+1}}) - V_H(s_{b_k})
   A_k^H = δ_k + γ_H · λ_H · A_{k+1}^H

Both functions operate on plain Python lists or 1-D torch Tensors and always
return a 1-D float32 Tensor.
"""

from __future__ import annotations

from typing import List, Union

import torch


Floats = Union[List[float], torch.Tensor]


def _to_tensor(x: Floats) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level: process λ-return (no critic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_process_return(
    rewards: Floats,
    gamma: float = 0.95,
    lam: float = 0.90,
    normalise: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Critic-free λ-return for one segment.

    A_t = r_t + (γλ)·r_{t+1} + (γλ)^2·r_{t+2} + …

    Parameters
    ----------
    rewards : sequence of per-step low-level rewards r_t^L
    gamma   : discount factor
    lam     : λ for multi-step return
    normalise : batch-normalise across the segment (recommended)
    eps     : numerical stability for normalisation

    Returns
    -------
    Tensor of shape (T,) – one advantage per action step.
    """
    r = _to_tensor(rewards)
    T = r.shape[0]
    returns = torch.zeros(T, dtype=torch.float32)
    G = 0.0
    gl = gamma * lam
    for t in reversed(range(T)):
        G = r[t].item() + gl * G
        returns[t] = G

    if normalise and T > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)

    return returns


# ─────────────────────────────────────────────────────────────────────────────
# High-level: macro GAE (with critic V_H)
# ─────────────────────────────────────────────────────────────────────────────

def compute_macro_gae(
    rewards: Floats,
    values: Floats,
    next_values: Floats,
    gamma: float = 0.95,
    lam: float = 0.95,
    normalise: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GAE for the macro (planner) level.

    δ_k = R_k^H + γ · V_H(s_{b_{k+1}}) - V_H(s_{b_k})
    A_k^H = δ_k + (γλ) · A_{k+1}^H

    Parameters
    ----------
    rewards     : high-level macro rewards R_k^H
    values      : V_H(s_{b_k})  – value at the start of each macro step
    next_values : V_H(s_{b_{k+1}}) – value at the end (next macro boundary)
    gamma       : macro discount
    lam         : GAE λ
    normalise   : normalise advantages across the episode
    eps         : numerical stability

    Returns
    -------
    Tensor of shape (K,) – macro advantage per subgoal.
    """
    R = _to_tensor(rewards)
    V = _to_tensor(values)
    Vn = _to_tensor(next_values)

    K = R.shape[0]
    adv = torch.zeros(K, dtype=torch.float32)
    gl = gamma * lam
    gae = 0.0
    for k in reversed(range(K)):
        delta = R[k].item() + gamma * Vn[k].item() - V[k].item()
        gae = delta + gl * gae
        adv[k] = gae

    if normalise and K > 1:
        adv = (adv - adv.mean()) / (adv.std() + eps)

    return adv


# ─────────────────────────────────────────────────────────────────────────────
# Value target for macro critic training
# ─────────────────────────────────────────────────────────────────────────────

def compute_value_targets(
    rewards: Floats,
    next_values: Floats,
    gamma: float = 0.95,
) -> torch.Tensor:
    """
    TD(0) value targets: y_k = R_k^H + γ · V_H(s_{b_{k+1}})

    Used to train the macro critic V_H with MSE loss.
    """
    R = _to_tensor(rewards)
    Vn = _to_tensor(next_values)
    return R + gamma * Vn
