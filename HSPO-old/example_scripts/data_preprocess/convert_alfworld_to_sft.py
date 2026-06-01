from __future__ import annotations
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Convert raw ALFWorld expert trajectories (from collect_alfworld_sft_demos.py)
into multi-turn SFT parquet files for MultiTurnSFTDataset.

Each trajectory becomes one row: a `messages` list of alternating
{role: user, content: <formatted_prompt>} / {role: assistant, content: <3-block response>}
dicts — exactly matching the prompt assembly in
AlfWorldEnvironmentManagerOptions.build_text_obs() / SimpleMemory.fetch_options().

Usage
-----
    python3 example_scripts/data_preprocess/convert_alfworld_to_sft.py \\
        --in_dir  $HSPO_SFT_RAW_DIR \\
        --out_dir $HSPO_SFT_DATA_DIR \\
        --val_frac 0.05 \\
        --history_length 2 \\
        --max_length 4096
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Inline copies of the prompt templates (must stay in sync with
# agent_system/environments/prompts/alfworld.py)
# ---------------------------------------------------------------------------
ALFWORLD_TEMPLATE_OPTIONS_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

Your current observation is:
{current_observation}

Your current sub-goal is:
{current_subgoal}
(If this is the first step of the episode, this will be "None".)

Your admissible actions are:
[{admissible_actions}]

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is allowed before, after, or between these blocks.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If <switch>KEEP</switch>, you MUST copy the EXACT current sub-goal.
    * If <switch>SWITCH</switch>, you MUST write a NEW short sub-goal achievable in a few actions.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>
"""

ALFWORLD_TEMPLATE_OPTIONS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your overall task is: {task_description}

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

You have already taken {step_count} step(s).
Most recent {history_length} observations and actions:
{action_history}

Your current observation is:
{current_observation}

Your current sub-goal is:
{current_subgoal}

Your admissible actions are:
[{admissible_actions}]

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is permitted anywhere in the output.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If you KEEP, copy the EXACT current sub-goal into <subgoal>.
    * If you SWITCH, write a NEW short sub-goal achievable in a few actions and NOT the entire task.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>
"""


def _format_admissible(admissible: list[str]) -> str:
    return "\n ".join(f"'{s}'" for s in admissible if s != "help")


def _format_history(steps: list[dict], t: int, history_length: int) -> tuple[str, int]:
    """Return (action_history_str, valid_len) for step index t."""
    valid_len = min(t, history_length)
    start_idx = t - valid_len
    lines = []
    for j in range(valid_len):
        step_num = start_idx + j + 1
        obs = steps[start_idx + j]["obs"]
        act = steps[start_idx + j]["action"]
        lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
    return "\n".join(lines), valid_len


def trajectory_to_messages(traj: dict, history_length: int) -> list[dict]:
    """Convert one raw trajectory dict to a messages list for the SFT dataset."""
    steps = traj["steps"]
    task_desc = traj["task_desc"]
    messages = []

    for t, step in enumerate(steps):
        admissible_str = _format_admissible(step["admissible"])

        if t == 0:
            user_content = ALFWORLD_TEMPLATE_OPTIONS_NO_HIS.format(
                current_observation=step["obs"],
                current_subgoal="None",
                admissible_actions=admissible_str,
            )
        else:
            action_history, valid_len = _format_history(steps, t, history_length)
            prev_subgoal = steps[t - 1]["subgoal"]
            user_content = ALFWORLD_TEMPLATE_OPTIONS.format(
                task_description=task_desc,
                step_count=t,
                history_length=valid_len,
                action_history=action_history,
                current_step=t + 1,
                current_observation=step["obs"],
                current_subgoal=prev_subgoal,
                admissible_actions=admissible_str,
            )

        assistant_content = (
            f"<switch>{step['switch']}</switch>"
            f"<subgoal>{step['subgoal']}</subgoal>"
            f"<action>{step['action']}</action>"
        )

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


def convert(in_dir: str, out_dir: str, val_frac: float, history_length: int, max_length: int, seed: int) -> None:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(in_path.glob("traj_*.json"))
    if not traj_files:
        raise FileNotFoundError(f"No traj_*.json files found in {in_dir}")

    print(f"Found {len(traj_files)} trajectory files.")

    records = []
    skipped = 0
    for fp in traj_files:
        with open(fp) as f:
            traj = json.load(f)
        if not traj.get("won"):
            skipped += 1
            continue
        try:
            msgs = trajectory_to_messages(traj, history_length)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")
            skipped += 1
            continue
        records.append({
            "messages": msgs,
            "task_type": traj.get("task_type", "unknown"),
            "gamefile": traj.get("gamefile", ""),
        })

    print(f"Converted {len(records)} trajectories ({skipped} skipped).")

    random.seed(seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * val_frac))
    val_records = records[:n_val]
    train_records = records[n_val:]

    def write_parquet(rows: list[dict], out_file: Path) -> None:
        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_file)
        print(f"  Wrote {len(rows)} rows → {out_file}")

    write_parquet(train_records, out_path / "train.parquet")
    write_parquet(val_records, out_path / "val.parquet")
    print(f"\nDone. Train: {len(train_records)}, Val: {len(val_records)}")


def main():
    parser = argparse.ArgumentParser(description="Convert ALFWorld raw trajectories to SFT parquet")
    parser.add_argument("--in_dir", default=os.environ.get("HSPO_SFT_RAW_DIR", os.environ.get("SEPRL_SFT_RAW_DIR", "")))
    parser.add_argument("--out_dir", default=os.environ.get("HSPO_SFT_DATA_DIR", os.environ.get("SEPRL_SFT_DATA_DIR", "")))
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--history_length", type=int, default=2,
                        help="Number of past steps shown in prompt (matches env.history_length in RL training)")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        history_length=args.history_length,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
