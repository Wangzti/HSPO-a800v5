# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
HSPO-specific metrics computation for wandb logging.

Reads HSPO keys from DataProto.non_tensor_batch (populated by the rollout collector)
and returns a flat metrics dict suitable for `logger.log(data=metrics, step=...)`.
"""

from typing import Any, Dict, Optional
import numpy as np


def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    """Mean of array, returning *default* for empty arrays."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return default
    return float(np.nanmean(arr))


def _safe_rate(arr: np.ndarray, default: float = 0.0) -> float:
    """Fraction of True/positive values in a bool-ish array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return default
    return float(np.mean(arr.astype(np.float64)))


def compute_hspo_metrics(batch, prefix: str = "hspo") -> Dict[str, Any]:
    """Extract HSPO-specific metrics from a DataProto batch produced by HSPOTrajectoryCollector.

    The batch.non_tensor_batch is expected to contain keys filled by the HSPO rollout loop:
      low_rewards, macro_rewards, advantage_low, format_valid, missing_switch,
      missing_subgoal, action_parse_ok, switch_target_match, premature_switch,
      delayed_switch, keep_subgoal_consistent, prm_done_after, prm_progress_delta,
      switch_target, phase.

    Returns a flat dict of scalar metrics keyed as ``{prefix}/{name}``.
    """
    nb = batch.non_tensor_batch
    metrics: Dict[str, Any] = {}

    # ── Low-level process reward ───────────────────────────────────────────
    if "low_rewards" in nb:
        lr = np.asarray(nb["low_rewards"], dtype=np.float64)
        metrics[f"{prefix}/low_reward/mean"] = _safe_mean(lr)
        metrics[f"{prefix}/low_reward/max"] = float(np.max(lr)) if lr.size else 0.0
        metrics[f"{prefix}/low_reward/min"] = float(np.min(lr)) if lr.size else 0.0

    # ── Low-level advantage (process λ-return) ─────────────────────────────
    if "advantage_low" in nb:
        al = np.asarray(nb["advantage_low"], dtype=np.float64)
        metrics[f"{prefix}/advantage_low/mean"] = _safe_mean(al)
        metrics[f"{prefix}/advantage_low/std"] = float(np.std(al)) if al.size else 0.0

    # ── Macro reward (high-level) ──────────────────────────────────────────
    if "macro_rewards" in nb:
        mr = np.asarray(nb["macro_rewards"], dtype=np.float64)
        metrics[f"{prefix}/macro_reward/mean"] = _safe_mean(mr)
        metrics[f"{prefix}/macro_reward/max"] = float(np.max(mr)) if mr.size else 0.0

    # ── Format validity ────────────────────────────────────────────────────
    if "format_valid" in nb:
        metrics[f"{prefix}/format_valid_rate"] = _safe_rate(nb["format_valid"])

    if "missing_switch" in nb:
        metrics[f"{prefix}/missing_switch_rate"] = _safe_rate(nb["missing_switch"])

    if "missing_subgoal" in nb:
        metrics[f"{prefix}/missing_subgoal_rate"] = _safe_rate(nb["missing_subgoal"])

    if "action_parse_ok" in nb:
        metrics[f"{prefix}/action_parse_ok_rate"] = _safe_rate(nb["action_parse_ok"])

    # ── SWITCH / termination metrics ───────────────────────────────────────
    if "switch_target_match" in nb:
        metrics[f"{prefix}/switch_accuracy"] = _safe_rate(nb["switch_target_match"])

    if "premature_switch" in nb:
        metrics[f"{prefix}/premature_switch_rate"] = _safe_rate(nb["premature_switch"])

    if "delayed_switch" in nb:
        metrics[f"{prefix}/delayed_switch_rate"] = _safe_rate(nb["delayed_switch"])

    if "keep_subgoal_consistent" in nb:
        metrics[f"{prefix}/keep_subgoal_consistency"] = _safe_rate(nb["keep_subgoal_consistent"])

    # ── PRM signals ────────────────────────────────────────────────────────
    if "prm_done_after" in nb:
        pd = np.asarray(nb["prm_done_after"], dtype=np.float64)
        metrics[f"{prefix}/prm_done/mean"] = _safe_mean(pd)
        metrics[f"{prefix}/prm_done_frac_over_tau"] = _safe_rate(pd >= 0.9)  # default tau_done=0.9

    if "prm_progress_delta" in nb:
        pp = np.asarray(nb["prm_progress_delta"], dtype=np.float64)
        metrics[f"{prefix}/prm_progress_delta/mean"] = _safe_mean(pp)

    # ── Training phase (informational) ─────────────────────────────────────
    if "phase" in nb:
        phases = np.asarray(nb["phase"])
        if phases.size:
            from collections import Counter
            for ph, cnt in Counter(phases.tolist()).items():
                metrics[f"{prefix}/phase/{ph}"] = cnt

    return metrics
