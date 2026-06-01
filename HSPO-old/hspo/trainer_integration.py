# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0
"""
HSPO Trainer Integration — bridges rollout data → verl PPO trainer.

Exposes two functions consumed by ray_trainer.py and dp_actor.py:

  compute_hspo_advantages()   – build per-token advantages from rollout data
  compute_switch_ce_loss()    – supervised CE loss on <switch> tokens
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from hspo.advantages import compute_process_return, compute_macro_gae
from hspo.config import HSPOConfig


def compute_hspo_advantages(
    action_mask: torch.Tensor,
    subgoal_mask: torch.Tensor,
    response_mask: torch.Tensor,
    advantage_low: torch.Tensor,
    phase: str,
    hspo_cfg: HSPOConfig,
    macro_rewards: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    traj_uids: Optional[list] = None,
) -> torch.Tensor:
    """
    Build per-token advantages for HSPO training.

    Parameters
    ----------
    action_mask : (B, L) bool
    subgoal_mask : (B, L) bool
    response_mask : (B, L) bool
    advantage_low : (B,) float32
        Pre-computed low-level advantage per step (from process λ-return).
    phase : str
        "low_level", "high_level", or "joint".
    hspo_cfg : HSPOConfig
    macro_rewards : (B,) float32 or None
        Macro reward per step (non-zero at episode boundaries).
    values : (B, L) float32 or None
        Critic values (needed for high_level phase).
    traj_uids : list[str] or None
        Trajectory UIDs for grouping macro steps.

    Returns
    -------
    advantages : (B, L) float32
    """
    B, L = action_mask.shape
    device = action_mask.device
    advantages = torch.zeros(B, L, dtype=torch.float32, device=device)

    # Combine with response_mask to exclude padding
    act_mask = action_mask.bool() & response_mask.bool()
    subg_mask = subgoal_mask.bool() & response_mask.bool()

    # ── Low-level advantages: spread A_L onto action tokens ──────────── #
    if phase in ("low_level", "joint"):
        adv_low_2d = advantage_low.float().to(device).unsqueeze(-1)  # (B, 1)
        advantages = advantages + adv_low_2d * act_mask.float()

    # ── High-level advantages: macro GAE onto subgoal tokens ─────────── #
    if phase in ("high_level", "joint") and macro_rewards is not None and values is not None:
        # Build macro GAE per trajectory
        if traj_uids is not None:
            unique_trajs = sorted(set(traj_uids))
            for traj_id in unique_trajs:
                traj_indices = [i for i, tid in enumerate(traj_uids) if tid == traj_id]
                if not traj_indices:
                    continue
                _add_macro_gae_for_trajectory(
                    advantages=advantages,
                    subgoal_mask=subg_mask,
                    traj_indices=traj_indices,
                    macro_rewards=macro_rewards,
                    values=values,
                    hspo_cfg=hspo_cfg,
                )
        else:
            _add_macro_gae_for_trajectory(
                advantages=advantages,
                subgoal_mask=subg_mask,
                traj_indices=list(range(B)),
                macro_rewards=macro_rewards,
                values=values,
                hspo_cfg=hspo_cfg,
            )

    return advantages


def _add_macro_gae_for_trajectory(
    advantages: torch.Tensor,
    subgoal_mask: torch.Tensor,
    traj_indices: list,
    macro_rewards: torch.Tensor,
    values: torch.Tensor,
    hspo_cfg: HSPOConfig,
) -> None:
    """
    Compute macro GAE for one trajectory and write into advantages tensor.
    """
    K = len(traj_indices)
    if K == 0:
        return

    R = macro_rewards[traj_indices].float()  # (K,)
    # Extract V_H at subgoal positions: mean value over subgoal tokens per step
    V = torch.zeros(K, device=advantages.device)
    Vn = torch.zeros(K, device=advantages.device)
    for k, idx in enumerate(traj_indices):
        subg_toks = subgoal_mask[idx]
        if subg_toks.any():
            V[k] = values[idx][subg_toks].mean()
        # Next value: value at next step's subgoal, or 0 if terminal
        if k + 1 < K:
            next_idx = traj_indices[k + 1]
            next_subg = subgoal_mask[next_idx]
            if next_subg.any():
                Vn[k] = values[next_idx][next_subg].mean()

    adv_H = compute_macro_gae(
        rewards=R,
        values=V,
        next_values=Vn,
        gamma=hspo_cfg.gamma_high,
        lam=hspo_cfg.lam_high,
        normalise=(K > 1),
    )

    for k, idx in enumerate(traj_indices):
        subg_toks = subgoal_mask[idx]
        if subg_toks.any():
            advantages[idx][subg_toks] += adv_H[k]


def compute_switch_ce_loss(
    log_prob: torch.Tensor,
    switch_mask: torch.Tensor,
    switch_target: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Supervised cross-entropy loss on switch tokens.

    switch_target: (B,) with values 1.0 (SWITCH) or 0.0 (KEEP).

    For each switch token, we compute binary CE:
      L = -[target * log(p) + (1-target) * log(1-p)]

    The loss is averaged over all switch tokens across the batch.

    Parameters
    ----------
    log_prob : (B, L) float32
        Token log-probabilities from the actor.
    switch_mask : (B, L) bool
        Mask marking switch-span tokens.
    switch_target : (B,) float32
        Target label per row (1.0 = SWITCH, 0.0 = KEEP).
    response_mask : (B, L) bool or None
        If provided, intersected with switch_mask.

    Returns
    -------
    Scalar loss averaged over all switch tokens.
    """
    mask = switch_mask.bool()
    if response_mask is not None:
        mask = mask & response_mask.bool()

    if not mask.any():
        return torch.tensor(0.0, device=log_prob.device)

    # Per-row: mean log_prob over switch tokens, then apply per-row target
    # For CE, we need p(switch=SWITCH) for each row.
    # We approximate by mean log_prob over switch-span tokens.
    log_p_switch = (log_prob * mask.float()).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    # log_p(switch)=log_p_switch, log_p(keep)=log(1-exp(log_p_switch))
    # Use binary CE with logits directly on switch token log-probs
    logits = log_p_switch  # scalar per row: average log-prob over switch tokens
    target = switch_target.float().to(log_prob.device)

    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss


def build_switch_ce_targets(switch_targets: list, device=None) -> torch.Tensor:
    """
    Convert string switch targets to float tensor.

    "SWITCH" → 1.0, "KEEP" → 0.0, "" / None → 0.0
    """
    vals = []
    for st in switch_targets:
        if isinstance(st, bytes):
            st = st.decode()
        if st == "SWITCH":
            vals.append(1.0)
        else:
            vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32, device=device)
