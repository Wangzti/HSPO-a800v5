#!/usr/bin/env python3
# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
Convert raw ALFWorld expert trajectories (from collect_alfworld_sft_demos.py)
into HSPO EpisodeRecord JSONL format.

This is the bridge between raw trajectory collection and HSPO segmentation/SFT building.

Usage
-----
    python3 example_scripts/data_preprocess/convert_raw_to_hspo_episodes.py \
        --in_dir /mnt/nfs/ztwang/data/hspo/sft/alfworld_raw \
        --out_path /mnt/nfs/ztwang/data/hspo/sft/alfworld_episodes.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _raw_step_to_step_record(step: dict, t: int, total_steps: int) -> Dict[str, Any]:
    """Convert a raw trajectory step dict to a StepRecord-compatible dict."""
    return {
        "t": t,
        "obs": step.get("obs", ""),
        "model_output": "",  # expert has no model output
        "switch": step.get("switch", "KEEP"),
        "subgoal": step.get("subgoal", ""),
        "action": step.get("action", ""),
        "valid_format": True,  # expert actions are always valid
        "action_valid": True,
        "env_reward": 0.0,
        "done": (t == total_steps - 1),
        "next_obs": "",
        "state_metadata": {},
        "low_reward": 0.0,
        "advantage_low": 0.0,
    }


def _raw_traj_to_episode(traj: dict, episode_idx: int) -> Dict[str, Any]:
    """Convert one raw trajectory dict to an EpisodeRecord-compatible dict.

    All steps are placed into a single dummy segment. HSPO segmentation
    will be applied later by segment_and_build_sft.py during SFT building.
    """
    raw_steps: List[dict] = traj.get("steps", [])
    n_steps = len(raw_steps)

    steps = [_raw_step_to_step_record(s, t, n_steps) for t, s in enumerate(raw_steps)]

    # Put all steps into a single segment with the episode's task as subgoal
    dummy_segment = {
        "segment_idx": 0,
        "subgoal": "",
        "subgoal_type": "",
        "steps": steps,
        "subgoal_completed": traj.get("won", False),
    }

    task_type = traj.get("task_type", "unknown")

    return {
        "episode_id": f"sft_traj_{episode_idx:05d}",
        "task_desc": traj.get("task_desc", ""),
        "task_type": task_type,
        "env_name": "alfworld",
        "gamefile": traj.get("gamefile", None),
        "segments": [dummy_segment],
        "won": traj.get("won", False),
        "total_steps": n_steps,
        "total_reward": 1.0 if traj.get("won", False) else 0.0,
        "_needs_resegment": True,  # signal to SFTBuilder to re-segment
    }


def convert(in_dir: str, out_path: str) -> None:
    in_path = Path(in_dir)
    traj_files = sorted(in_path.glob("traj_*.json"))
    if not traj_files:
        raise FileNotFoundError(f"No traj_*.json files found in {in_dir}")

    print(f"Found {len(traj_files)} trajectory files.")

    n_written = 0
    n_skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, fp in enumerate(traj_files):
            with open(fp) as tf:
                traj = json.load(tf)
            if not traj.get("won", False):
                n_skipped += 1
                continue
            episode = _raw_traj_to_episode(traj, idx)
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} episodes to {out_path} ({n_skipped} skipped, not won).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw ALFWorld trajectories to HSPO EpisodeRecord JSONL"
    )
    parser.add_argument("--in_dir", required=True, help="Directory of raw traj_*.json files")
    parser.add_argument("--out_path", required=True, help="Output JSONL file path")
    args = parser.parse_args()
    convert(args.in_dir, args.out_path)


if __name__ == "__main__":
    main()
