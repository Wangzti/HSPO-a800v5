# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO data contracts.

These dataclasses define the canonical shapes that cross module boundaries:
  rollout → reward manager → advantage computation → loss functions

Design rule: all fields are JSON-serialisable primitives or plain lists/dicts
so episodes can be saved/loaded without pickle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Per-step record (one LLM call = one step)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Everything produced / received at a single environment step."""
    t: int                          # step index within the episode
    obs: str                        # raw observation text
    model_output: str               # raw LLM output string
    switch: str                     # "SWITCH" or "KEEP"
    subgoal: str                    # current subgoal NL text
    action: str                     # extracted action text
    valid_format: bool              # output parsed correctly
    action_valid: bool              # action accepted by environment
    env_reward: float               # terminal reward from env (0.0 most steps)
    done: bool
    next_obs: str = ""
    state_metadata: Dict[str, Any] = field(default_factory=dict)
    # Token-level span positions in the *response* token sequence
    switch_span: List[int] = field(default_factory=list)   # [start, end)
    subgoal_span: List[int] = field(default_factory=list)  # [start, end)
    action_span: List[int] = field(default_factory=list)   # [start, end)
    # PRM signals (filled by reward manager)
    prm_progress_before: float = 0.0
    prm_progress_after: float = 0.0
    prm_done_after: float = 0.0
    prm_valid: float = 1.0
    prm_side_before: float = 0.0
    prm_side_after: float = 0.0
    low_reward: float = 0.0        # r_t^L computed by HSPO reward manager
    # Advantage (set after advantage computation)
    advantage_low: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Segment record (one subgoal = one segment = list of steps)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentRecord:
    """All steps belonging to a single subgoal segment."""
    segment_idx: int
    subgoal: str
    subgoal_type: str              # canonical type e.g. "CLEAN_OBJECT"
    steps: List[StepRecord] = field(default_factory=list)
    subgoal_completed: bool = False
    segment_len: int = 0

    def __post_init__(self):
        self.segment_len = len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Macro transition (one subgoal decision by the planner)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroTransition:
    """High-level transition: planner issues subgoal g_k, executor runs it."""
    k: int                         # macro step index
    state_obs: str                 # observation at macro boundary (s_{b_k})
    subgoal: str
    next_state_obs: str            # observation at next macro boundary
    segment: SegmentRecord
    # High-level reward components
    task_reward: float = 0.0       # R_task (1 if terminal success, else 0)
    side_penalty: float = 0.0      # β · C_side
    redundancy_penalty: float = 0.0  # η · C_red
    macro_reward: float = 0.0      # R_k^H = task_reward - side - redundancy
    # Macro-PPO advantage (set after macro GAE computation)
    advantage_high: float = 0.0
    value_target: float = 0.0
    # Subgoal token positions in the planner's *response* token sequence
    subgoal_span: List[int] = field(default_factory=list)
    switch_span: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Full episode record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    episode_id: str
    task_desc: str
    task_type: str
    env_name: str                  # "alfworld" or "webshop"
    gamefile: Optional[str] = None
    segments: List[SegmentRecord] = field(default_factory=list)
    macro_transitions: List[MacroTransition] = field(default_factory=list)
    won: bool = False
    total_steps: int = 0
    total_reward: float = 0.0
    invalid_action_count: int = 0
    format_error_count: int = 0
    _needs_resegment: bool = False  # set by conversion scripts; cleared after segmentation
