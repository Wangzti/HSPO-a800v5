# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
HSPO SFT Dataset Builder.

Converts segmented rollout trajectories into four JSONL datasets:

  sft_format   – teach the model the <switch>/<subgoal>/<action> output format
  sft_executor – low-level executor examples (observation → action, subgoal fixed)
  sft_planner  – high-level planner examples (macro obs → subgoal decision)
  sft_all      – union of the above three

Each record is a dict with keys:
    prompt    : str  – user turn (task + observation)
    response  : str  – target model response in plan-execute format
    task_type : str  – e.g. "clean", "heat", ...
    switch    : str  – "SWITCH" or "KEEP"
    quality   : float – normalised quality score in [0, 1]

Usage
-----
    from hspo.sft_builder import SFTBuilder
    from hspo.types import EpisodeRecord

    builder = SFTBuilder(quality_threshold=0.3)
    records_format, records_exec, records_plan, records_all = builder.build(episodes)
    builder.write_jsonl(records_all, "data/sft_all.jsonl")
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hspo.types import EpisodeRecord, SegmentRecord, StepRecord
from hspo.segmentation import segment_trajectory, segment_advantages
from hspo.config import HSPOConfig


# ── Output record ─────────────────────────────────────────────────────────────

def _record(
    prompt: str,
    response: str,
    task_type: str,
    switch: str,
    quality: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "prompt":    prompt,
        "response":  response,
        "task_type": task_type,
        "switch":    switch,
        "quality":   round(float(quality), 4),
    }
    if extra:
        r.update(extra)
    return r


# ── SFT record templates ──────────────────────────────────────────────────────

def _format_response(switch: str, subgoal: str, action: str) -> str:
    return f"<switch>{switch}</switch><subgoal>{subgoal}</subgoal><action>{action}</action>"


def _build_prompt(task_desc: str, obs: str, subgoal_context: str = "") -> str:
    """Standard prompt for a plan-execute agent step."""
    lines = [f"Task: {task_desc}", f"Observation: {obs}"]
    if subgoal_context:
        lines.append(f"Current subgoal: {subgoal_context}")
    return "\n".join(lines)


# ── Quality scoring ───────────────────────────────────────────────────────────

def _step_quality(step: StepRecord, advantages: List[float], step_idx_in_seg: int) -> float:
    """
    Quality ∈ [0, 1]:  normalised advantage × validity × format correctness.

    Higher quality → more suitable as SFT target.
    """
    base = 0.5 + (advantages[step_idx_in_seg] / 2.0 if advantages else 0.0)
    base = max(0.0, min(1.0, base))
    if not step.valid_format:
        base *= 0.2
    if not step.action_valid:
        base *= 0.5
    return base


# ── Main builder ─────────────────────────────────────────────────────────────

class SFTBuilder:
    """
    Convert a list of EpisodeRecords into four SFT JSONL datasets.

    Parameters
    ----------
    quality_threshold : float
        Steps with quality < this value are excluded.
    hspo_cfg          : HSPOConfig for λ-return computation.
    """

    def __init__(
        self,
        quality_threshold: float = 0.3,
        hspo_cfg: Optional[HSPOConfig] = None,
    ) -> None:
        self.q_thr = quality_threshold
        self.cfg = hspo_cfg or HSPOConfig()

    # ------------------------------------------------------------------ #

    def build(
        self,
        episodes: List[EpisodeRecord],
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Build the four SFT splits from a list of episodes.

        Returns (format_records, executor_records, planner_records, all_records)
        """
        fmt_recs: List[Dict] = []
        exec_recs: List[Dict] = []
        plan_recs: List[Dict] = []

        for ep in episodes:
            all_steps = [s for seg in ep.segments for s in seg.steps]
            if not all_steps:
                continue
            # Re-segment when: (a) no existing segments, or (b) caller flagged
            # that the episode needs HSPO segmentation applied (e.g. raw
            # trajectories from convert_raw_to_hspo_episodes.py).
            needs_reseg = (not ep.segments) or getattr(ep, '_needs_resegment', False)
            segments = segment_trajectory(all_steps, self.cfg) if needs_reseg else ep.segments

            for seg in segments:
                advs = segment_advantages(seg, self.cfg)

                for j, step in enumerate(seg.steps):
                    q = _step_quality(step, advs, j)
                    if q < self.q_thr:
                        continue

                    sw = step.switch or "KEEP"
                    subgoal = step.subgoal or seg.subgoal
                    action = step.action
                    prompt = _build_prompt(ep.task_desc, step.obs, subgoal)

                    # ── Format record: teach output format ────────────────
                    fmt_recs.append(_record(
                        prompt=prompt,
                        response=_format_response(sw, subgoal, action),
                        task_type=ep.task_type,
                        switch=sw,
                        quality=q,
                    ))

                    # ── Executor record: given fixed subgoal, pick action ─
                    exec_prompt = _build_prompt(ep.task_desc, step.obs, subgoal)
                    exec_recs.append(_record(
                        prompt=exec_prompt,
                        response=_format_response("KEEP", subgoal, action),
                        task_type=ep.task_type,
                        switch="KEEP",
                        quality=q,
                    ))

                    # ── Planner record: at SWITCH boundary only ────────────
                    if sw == "SWITCH":
                        plan_prompt = _build_prompt(ep.task_desc, step.obs)
                        # Teacher-forced subgoal from expert segmentation
                        plan_recs.append(_record(
                            prompt=plan_prompt,
                            response=_format_response("SWITCH", subgoal, action),
                            task_type=ep.task_type,
                            switch="SWITCH",
                            quality=q,
                            extra={"segment_idx": seg.segment_idx},
                        ))

        all_recs = fmt_recs + exec_recs + plan_recs
        return fmt_recs, exec_recs, plan_recs, all_recs

    # ------------------------------------------------------------------ #

    @staticmethod
    def write_jsonl(records: List[Dict], path: str) -> None:
        """Write records to a JSONL file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        """Load records from a JSONL file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def jsonl_to_parquet_messages(
        records: List[Dict],
        train_path: str,
        val_path: Optional[str] = None,
        val_frac: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, int]:
        """
        Convert SFTBuilder JSONL records to verl-compatible parquet files.

        Each record's 'prompt' becomes user message, 'response' becomes
        assistant message. Output parquet has a 'messages' column with
        list-of-dicts format that MultiTurnSFTDataset expects.

        Parameters
        ----------
        records : list of dicts with 'prompt' and 'response' keys.
        train_path : path for train.parquet.
        val_path : path for val.parquet (auto-derived if None).
        val_frac : fraction of records for validation.
        seed : shuffle seed.

        Returns
        -------
        dict with 'train' and 'val' counts.
        """
        import random

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        messages_records = []
        for rec in records:
            messages_records.append({
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["response"]},
                ],
                "task_type": rec.get("task_type", ""),
                "switch": rec.get("switch", ""),
            })

        random.seed(seed)
        random.shuffle(messages_records)

        n_val = max(1, int(len(messages_records) * val_frac))
        val_recs = messages_records[:n_val]
        train_recs = messages_records[n_val:]

        train_path = Path(train_path)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        if val_path is None:
            val_path = train_path.parent / "val.parquet"
        val_path = Path(val_path)
        val_path.parent.mkdir(parents=True, exist_ok=True)

        def _write(rows, path):
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(path))

        _write(train_recs, str(train_path))
        _write(val_recs, str(val_path))

        return {"train": len(train_recs), "val": len(val_recs)}

    @staticmethod
    def jsonl_to_parquet_messages_single(records: List[Dict], path: str) -> int:
        """Convert SFTBuilder JSONL records to a single parquet file."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        messages_records = []
        for rec in records:
            messages_records.append({
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["response"]},
                ],
                "task_type": rec.get("task_type", ""),
                "switch": rec.get("switch", ""),
            })

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(messages_records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(out))
        return len(messages_records)

    def build_and_write(
        self,
        episodes: List[EpisodeRecord],
        output_dir: str,
    ) -> Dict[str, int]:
        """Build all four splits and write to output_dir/sft_{split}.jsonl."""
        fmt_r, exec_r, plan_r, all_r = self.build(episodes)
        counts = {}
        for name, recs in [("format", fmt_r), ("executor", exec_r), ("planner", plan_r), ("all", all_r)]:
            path = str(Path(output_dir) / f"sft_{name}.jsonl")
            self.write_jsonl(recs, path)
            counts[name] = len(recs)
        return counts
