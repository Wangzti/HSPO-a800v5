# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO Trajectory Segmentation (Phase 2 SFT data builder).

Takes a complete rollout trajectory (list of StepRecords) and segments it
into SegmentRecords based on SWITCH decisions.

A SWITCH boundary is triggered when:
1. The model explicitly emits <switch>SWITCH</switch>
2. The step count within the segment reaches max_segment_len (force-close)

Each SegmentRecord carries the steps, and per-step `low_reward` values are
read from StepRecord.low_reward. The helper `segment_advantages()` computes
the process λ-return for a given segment without modifying it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from hspo.types import EpisodeRecord, MacroTransition, SegmentRecord, StepRecord
from hspo.advantages import compute_process_return
from hspo.config import HSPOConfig


def segment_trajectory(
    steps: List[StepRecord],
    hspo_cfg: Optional[HSPOConfig] = None,
) -> List[SegmentRecord]:
    """
    Segment a list of StepRecords (one episode) into SegmentRecords.

    Parameters
    ----------
    steps      : ordered list of StepRecord for one episode
    hspo_cfg   : HSPOConfig; uses defaults if None

    Returns
    -------
    List of SegmentRecord in episode order.
    """
    cfg = hspo_cfg or HSPOConfig()

    if not steps:
        return []

    segments: List[SegmentRecord] = []
    seg_idx: int = 0
    current_steps: List[StepRecord] = []
    current_subgoal: str = ""
    current_subgoal_type: str = ""

    def _flush(subgoal_done: bool = False) -> None:
        nonlocal seg_idx
        if not current_steps:
            return
        seg = SegmentRecord(
            segment_idx=seg_idx,
            subgoal=current_subgoal,
            subgoal_type=current_subgoal_type,
            steps=list(current_steps),
            subgoal_completed=subgoal_done,
        )
        segments.append(seg)
        seg_idx += 1
        current_steps.clear()

    for step in steps:
        is_switch = (step.switch == "SWITCH") or (not current_subgoal)

        if is_switch and current_steps:
            # Subgoal is considered completed if the SWITCH is model-initiated
            _flush(subgoal_done=True)

        if is_switch:
            current_subgoal = step.subgoal or current_subgoal
            current_subgoal_type = _infer_subgoal_type(current_subgoal)

        current_steps.append(step)

        # Force-close at max_segment_len
        if len(current_steps) >= cfg.max_segment_len:
            _flush(subgoal_done=False)

    # Close final open segment
    _flush(subgoal_done=steps[-1].done if steps else False)

    return segments


def segment_advantages(seg: SegmentRecord, hspo_cfg: Optional[HSPOConfig] = None) -> List[float]:
    """Compute process λ-return for a SegmentRecord (non-mutating)."""
    cfg = hspo_cfg or HSPOConfig()
    rewards = [s.low_reward for s in seg.steps]
    if not rewards:
        return []
    advs = compute_process_return(
        rewards,
        gamma=cfg.gamma_low,
        lam=cfg.lam_low,
        normalise=(len(rewards) > 1),
    )
    return advs.tolist()


def build_macro_transitions(
    segments: List[SegmentRecord],
    steps: List[StepRecord],
    episode: EpisodeRecord,
) -> List[MacroTransition]:
    """
    Convert a segmented trajectory into MacroTransition records for
    high-level planner training.

    MacroTransition k spans from the start of segment k to the start of
    segment k+1 (or episode end).

    task_reward is 1.0 only on the last transition if the episode was won.
    """
    if not segments:
        return []

    # Build a step-index → obs lookup
    step_obs: Dict[int, str] = {s.t: s.obs for s in steps}

    transitions: List[MacroTransition] = []
    for k, seg in enumerate(segments):
        seg_start_t = seg.steps[0].t
        obs_at_start = step_obs.get(seg_start_t, "")

        if k + 1 < len(segments):
            next_start_t = segments[k + 1].steps[0].t
            obs_at_next = step_obs.get(next_start_t, "")
            task_rew = 0.0
        else:
            obs_at_next = seg.steps[-1].next_obs
            task_rew = 1.0 if episode.won else 0.0

        mt = MacroTransition(
            k=k,
            state_obs=obs_at_start,
            subgoal=seg.subgoal,
            next_state_obs=obs_at_next,
            segment=seg,
            task_reward=task_rew,
            macro_reward=task_rew,  # side/redundancy added by caller if needed
        )
        transitions.append(mt)

    return transitions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _infer_subgoal_type(subgoal_text: str) -> str:
    sl = subgoal_text.lower()
    if "find" in sl or "locate" in sl or "look for" in sl:
        return "FIND_OBJECT"
    if "pick" in sl:
        return "PICK_OBJECT"
    if "clean" in sl:
        return "CLEAN_OBJECT"
    if "heat" in sl or "microwave" in sl:
        return "HEAT_OBJECT"
    if "cool" in sl or "fridge" in sl or "refrigerator" in sl:
        return "COOL_OBJECT"
    if "place" in sl or "put" in sl:
        return "PLACE_OBJECT"
    if "examine" in sl or "look at" in sl:
        return "EXAMINE_OBJECT"
    if "go to" in sl:
        return "GO_TO_RECEPTACLE"
    return "FIND_OBJECT"


def compute_trajectory_stats(
    segments: List[SegmentRecord],
    episode: Optional[EpisodeRecord] = None,
) -> Dict[str, Any]:
    """
    Per-trajectory summary statistics over segments.

    Returns a dict with:
        num_segments, num_steps, mean_segment_len,
        total_low_reward, success, switch_diversity
    """
    if not segments:
        return {}

    total_steps = sum(s.segment_len for s in segments)
    total_low = sum(s.low_reward for seg in segments for s in seg.steps)
    success = episode.won if episode is not None else segments[-1].subgoal_completed

    # Fraction of non-trivial switches (changed subgoal text)
    switches_total = max(len(segments) - 1, 1)
    switches_distinct = sum(
        1 for k in range(1, len(segments))
        if segments[k].subgoal.strip() != segments[k - 1].subgoal.strip()
    )

    return {
        "num_segments":     len(segments),
        "num_steps":        total_steps,
        "mean_segment_len": total_steps / len(segments),
        "total_low_reward": float(total_low),
        "success":          bool(success),
        "switch_diversity": switches_distinct / switches_total,
    }
