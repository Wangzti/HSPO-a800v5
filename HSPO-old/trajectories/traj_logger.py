# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Trajectory logging module for HiPER-agent.

Provides a unified trajectory storage format and writer for both
ALFWorld and WebShop environments under the SeP-RL algorithm.

Typical usage
-------------
    from trajectories.traj_logger import TrajLogger, make_step, make_meta, make_outcome

    logger = TrajLogger("/path/to/log/dir")
    meta = make_meta(env="webshop", model="qwen2.5-7b-instruct", source="sft_collection",
                     task_desc="Buy a red shirt under $20", session_idx=500)
    steps = [make_step(t=0, obs="...", available_actions=["search[shirt]"],
                       action="search[shirt]", subgoal="Find a red shirt",
                       switch="SWITCH", reward=0.0, is_valid=True, done=False)]
    outcome = make_outcome(won=True, total_reward=1.0, num_steps=len(steps))
    path = logger.log_episode(meta, steps, outcome)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Record constructors
# ---------------------------------------------------------------------------

def make_step(
    t: int,
    obs: str,
    available_actions: list[str],
    action: str,
    subgoal: str,
    switch: str,
    reward: float,
    is_valid: bool,
    done: bool,
) -> dict[str, Any]:
    """Create a single trajectory step record.

    Parameters
    ----------
    t:
        Zero-based step index within the episode.
    obs:
        Raw (or formatted) observation string at this step.
    available_actions:
        List of admissible action strings available at this step.
    action:
        Action actually taken (one of ``available_actions`` or a fallback).
    subgoal:
        Natural-language sub-goal active at this step.
    switch:
        ``"SWITCH"`` if the sub-goal changed, ``"KEEP"`` otherwise.
    reward:
        Scalar reward received after taking this action.
    is_valid:
        Whether the chosen action was in ``available_actions``.
    done:
        Whether the episode ended after this step.

    Returns
    -------
    dict
        Step record with all provided fields.
    """
    return {
        "t": t,
        "obs": obs,
        "available_actions": available_actions,
        "action": action,
        "subgoal": subgoal,
        "switch": switch,
        "reward": reward,
        "is_valid": is_valid,
        "done": done,
    }


def make_meta(env: str, model: str, source: str, **kwargs: Any) -> dict[str, Any]:
    """Create an episode metadata record.

    Parameters
    ----------
    env:
        Environment name (e.g. ``"webshop"``, ``"alfworld"``).
    model:
        Model identifier used to generate the trajectory
        (e.g. ``"qwen2.5-7b-instruct"``).
    source:
        Data source label (e.g. ``"sft_collection"``, ``"rl_rollout"``).
    **kwargs:
        Additional metadata fields (e.g. ``task_desc``, ``task_type``,
        ``session_idx``, ``gamefile``).

    Returns
    -------
    dict
        Metadata record, always including a UTC ISO-8601 ``timestamp``.
    """
    record: dict[str, Any] = {
        "env": env,
        "model": model,
        "source": source,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    record.update(kwargs)
    return record


def make_outcome(won: bool, total_reward: float, num_steps: int, **kwargs: Any) -> dict[str, Any]:
    """Create an episode outcome record.

    Parameters
    ----------
    won:
        Whether the agent successfully completed the task.
    total_reward:
        Sum of all per-step rewards (or final task score for WebShop).
    num_steps:
        Total number of steps taken in the episode.
    **kwargs:
        Additional outcome fields (e.g. ``task_score``, ``timeout``).

    Returns
    -------
    dict
        Outcome record with all provided fields.
    """
    record: dict[str, Any] = {
        "won": won,
        "total_reward": total_reward,
        "num_steps": num_steps,
    }
    record.update(kwargs)
    return record


# ---------------------------------------------------------------------------
# TrajLogger class
# ---------------------------------------------------------------------------

class TrajLogger:
    """Writes episode trajectories to disk in JSON and/or JSONL format.

    Each call to :meth:`log_episode` produces one atomic JSON file named
    ``traj_{counter:05d}.json``.  The counter is initialised from the number
    of ``traj_*.json`` files already present in ``log_dir`` so that multiple
    runs append naturally.

    :meth:`log_episode_jsonl` instead appends a single JSON line to a shared
    JSONL file, which is convenient for streaming RL training data collection.

    Parameters
    ----------
    log_dir:
        Directory where trajectory files will be written.  Created
        (including parents) if it does not already exist.
    """

    def __init__(self, log_dir: str | os.PathLike) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Initialise counter from files already on disk so that resumed runs
        # do not overwrite previous trajectories.
        existing = list(self.log_dir.glob("traj_*.json"))
        self._counter: int = len(existing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episode(
        self,
        meta: dict[str, Any],
        steps: list[dict[str, Any]],
        outcome: dict[str, Any],
    ) -> Path:
        """Write one episode to a new JSON file.

        The write is *atomic*: data is first written to a sibling temp file in
        the same directory, then renamed into place.  This prevents partially-
        written files from being picked up by downstream readers.

        Parameters
        ----------
        meta:
            Episode metadata (use :func:`make_meta`).
        steps:
            Ordered list of step records (use :func:`make_step`).
        outcome:
            Episode outcome (use :func:`make_outcome`).

        Returns
        -------
        Path
            Absolute path of the written trajectory file.
        """
        record = {"meta": meta, "steps": steps, "outcome": outcome}
        out_path = self.log_dir / f"traj_{self._counter:05d}.json"

        # Atomic write: temp file in the same directory ensures same filesystem
        # for the rename, which is guaranteed to be atomic on POSIX.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.log_dir, prefix=".tmp_traj_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            os.replace(tmp_path, out_path)
        except Exception:
            # Clean up temp file on error; re-raise to the caller.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._counter += 1
        return out_path

    def log_episode_jsonl(
        self,
        meta: dict[str, Any],
        steps: list[dict[str, Any]],
        outcome: dict[str, Any],
        filename: str = "trajs.jsonl",
    ) -> None:
        """Append one episode as a single JSON line to a JSONL file.

        Suitable for streaming data collection during RL training, where
        episodes are written continuously and consumed by the training loop.

        Parameters
        ----------
        meta:
            Episode metadata (use :func:`make_meta`).
        steps:
            Ordered list of step records (use :func:`make_step`).
        outcome:
            Episode outcome (use :func:`make_outcome`).
        filename:
            Name of the JSONL file within ``log_dir``.  Defaults to
            ``"trajs.jsonl"``.  The file is created if it does not exist and
            opened in append mode otherwise.
        """
        record = {"meta": meta, "steps": steps, "outcome": outcome}
        jsonl_path = self.log_dir / filename
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
