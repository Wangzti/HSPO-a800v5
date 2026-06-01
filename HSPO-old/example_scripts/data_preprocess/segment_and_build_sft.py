#!/usr/bin/env python3
# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
Segment collected trajectories and build HSPO SFT datasets.

Accepts either:
  - A JSONL file of EpisodeRecord dicts (--episodes), or
  - A directory of raw trajectory JSON files (--raw_dir) from
    collect_alfworld_sft_demos.py

Runs HSPO segmentation and writes four SFT JSONL files + parquet:
    sft_format.jsonl, sft_executor.jsonl, sft_planner.jsonl, sft_all.jsonl
    train.parquet, val.parquet

Usage
-----
    # From EpisodeRecord JSONL:
    python segment_and_build_sft.py \\
        --episodes /path/to/episodes.jsonl \\
        --output_dir /path/to/sft_data/

    # Directly from raw trajectories:
    python segment_and_build_sft.py \\
        --raw_dir /path/to/raw_trajectories/ \\
        --output_dir /path/to/sft_data/ \\
        [--quality_threshold 0.3] \\
        [--gamma_low 0.95] [--lam_low 0.90] [--max_segment_len 8]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure HSPO-agent is on the path when run directly
_SCRIPT_DIR = Path(__file__).resolve().parent
_HSPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_HSPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSPO_ROOT))

from hspo.types import EpisodeRecord, SegmentRecord, StepRecord
from hspo.config import HSPOConfig
from hspo.sft_builder import SFTBuilder


def load_episodes(path: str) -> list:
    """Load EpisodeRecord dicts from a JSONL file."""
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            episodes.append(_dict_to_episode(raw))
    return episodes


def load_raw_episodes(raw_dir: str) -> list:
    """Load raw ALFWorld trajectory JSON files and convert to EpisodeRecords.

    Each raw file has: task_desc, gamefile, task_type, steps[], won.
    Steps are packed into a single dummy segment; HSPO segmentation will
    be applied by SFTBuilder because _needs_resegment=True.
    """
    in_path = Path(raw_dir)
    traj_files = sorted(in_path.glob("traj_*.json"))
    if not traj_files:
        raise FileNotFoundError(f"No traj_*.json files found in {raw_dir}")

    episodes = []
    n_skipped = 0
    for idx, fp in enumerate(traj_files):
        with open(fp, "r", encoding="utf-8") as f:
            traj = json.load(f)
        if not traj.get("won", False):
            n_skipped += 1
            continue

        raw_steps = traj.get("steps", [])
        n_steps = len(raw_steps)

        # Build StepRecord list
        step_recs = []
        for t, s in enumerate(raw_steps):
            step_recs.append(StepRecord(
                t=t,
                obs=s.get("obs", ""),
                model_output="",
                switch=s.get("switch", "KEEP"),
                subgoal=s.get("subgoal", ""),
                action=s.get("action", ""),
                valid_format=True,
                action_valid=True,
                env_reward=0.0,
                done=(t == n_steps - 1),
                next_obs="",
                state_metadata={},
                low_reward=0.0,
                advantage_low=0.0,
            ))

        # Single dummy segment containing all steps
        dummy_seg = SegmentRecord(
            segment_idx=0,
            subgoal="",
            subgoal_type="",
            steps=step_recs,
            subgoal_completed=traj.get("won", False),
        )

        episodes.append(EpisodeRecord(
            episode_id=f"sft_traj_{idx:05d}",
            task_desc=traj.get("task_desc", ""),
            task_type=traj.get("task_type", "unknown"),
            env_name="alfworld",
            gamefile=traj.get("gamefile", None),
            segments=[dummy_seg],
            won=traj.get("won", False),
            total_steps=n_steps,
            total_reward=1.0 if traj.get("won", False) else 0.0,
            _needs_resegment=True,
        ))

    if n_skipped:
        print(f"  Skipped {n_skipped} non-winning trajectories")
    return episodes


def _dict_to_step(d: dict) -> StepRecord:
    return StepRecord(
        t=d.get("t", 0),
        obs=d.get("obs", ""),
        model_output=d.get("model_output", ""),
        switch=d.get("switch", "KEEP"),
        subgoal=d.get("subgoal", ""),
        action=d.get("action", ""),
        valid_format=d.get("valid_format", True),
        action_valid=d.get("action_valid", True),
        env_reward=d.get("env_reward", 0.0),
        done=d.get("done", False),
        next_obs=d.get("next_obs", ""),
        state_metadata=d.get("state_metadata", {}),
        low_reward=d.get("low_reward", 0.0),
        advantage_low=d.get("advantage_low", 0.0),
    )


def _dict_to_segment(d: dict) -> SegmentRecord:
    steps = [_dict_to_step(s) for s in d.get("steps", [])]
    return SegmentRecord(
        segment_idx=d.get("segment_idx", 0),
        subgoal=d.get("subgoal", ""),
        subgoal_type=d.get("subgoal_type", ""),
        steps=steps,
        subgoal_completed=d.get("subgoal_completed", False),
    )


def _dict_to_episode(d: dict) -> EpisodeRecord:
    segments = [_dict_to_segment(s) for s in d.get("segments", [])]
    return EpisodeRecord(
        episode_id=d.get("episode_id", ""),
        task_desc=d.get("task_desc", ""),
        task_type=d.get("task_type", ""),
        env_name=d.get("env_name", "alfworld"),
        gamefile=d.get("gamefile", None),
        segments=segments,
        won=d.get("won", False),
        total_steps=d.get("total_steps", 0),
        total_reward=d.get("total_reward", 0.0),
        _needs_resegment=d.get("_needs_resegment", False),
    )


def main():
    parser = argparse.ArgumentParser(description="HSPO: segment trajectories and build SFT datasets")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--episodes", help="Input JSONL of EpisodeRecord dicts")
    input_group.add_argument("--raw_dir", help="Directory of raw traj_*.json files (from collect_alfworld_sft_demos.py)")
    parser.add_argument("--output_dir", required=True, help="Output directory for SFT JSONLs and parquet files")
    parser.add_argument("--quality_threshold", type=float, default=0.3)
    parser.add_argument("--gamma_low", type=float, default=0.95)
    parser.add_argument("--lam_low", type=float, default=0.90)
    parser.add_argument("--max_segment_len", type=int, default=8)
    args = parser.parse_args()

    cfg = HSPOConfig(
        gamma_low=args.gamma_low,
        lam_low=args.lam_low,
        max_segment_len=args.max_segment_len,
    )

    if args.raw_dir:
        print(f"Loading raw trajectories from {args.raw_dir} ...")
        episodes = load_raw_episodes(args.raw_dir)
    else:
        print(f"Loading episodes from {args.episodes} ...")
        episodes = load_episodes(args.episodes)
    print(f"  Loaded {len(episodes)} episodes")

    builder = SFTBuilder(quality_threshold=args.quality_threshold, hspo_cfg=cfg)
    counts = builder.build_and_write(episodes, args.output_dir)

    # Output parquet for ALL splits (needed by verl SFT trainer for each step)
    for split_name in ["format", "executor", "planner", "all"]:
        recs = builder.load_jsonl(str(Path(args.output_dir) / f"sft_{split_name}.jsonl"))
        if recs:
            builder.jsonl_to_parquet_messages(
                recs,
                train_path=str(Path(args.output_dir) / f"train_{split_name}.parquet"),
                val_path=str(Path(args.output_dir) / f"val_{split_name}.parquet"),
                val_frac=0.05,
            )

    print(f"\nSFT datasets written to {args.output_dir}/")
    for split, n in counts.items():
        print(f"  sft_{split}.jsonl : {n} records")
        print(f"  train_{split}.parquet + val_{split}.parquet")


if __name__ == "__main__":
    main()
