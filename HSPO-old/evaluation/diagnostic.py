#!/usr/bin/env python3
# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
HSPO Diagnostic Experiments (Section 5 ablations).

Implements two diagnostic experiments from the paper:

Task A – Subgoal Attribution Test
    Given an observation where the correct subgoal is g*, measure
    P(model issues g* | observation) across 6 task types.
    Compared against baseline that always issues "find object".

Task B – Executor Isolation Test
    Force-feed a fixed subgoal g to the executor and measure
    ExecutorSuccess(g) = fraction of steps that advance toward g,
    regardless of what the planner would have chosen.
    This tests the executor's ability to follow subgoals it was given.

Usage
-----
    python evaluation/diagnostic.py \\
        --model_path /path/to/model \\
        --data_path  /path/to/diagnostic_scenarios.jsonl \\
        [--temperature 0.0] \\
        [--num_samples 5]    # per-scenario samples for stochastic models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HSPO_ROOT = Path(__file__).resolve().parent.parent
if str(_HSPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSPO_ROOT))

import numpy as np


# ── Scenario format ──────────────────────────────────────────────────────────
# A diagnostic_scenarios.jsonl file has one JSON object per line:
# {
#   "scenario_id":   "clean_cup_A_001",
#   "task_type":     "clean",
#   "task_desc":     "clean a cup and put it in cabinet",
#   "observation":   "You are in the kitchen...",
#   "correct_subgoal": "go to the sinkbasin",   # ground-truth subgoal
#   "correct_switch":  "SWITCH",
#   "executor_test":   true,                     # include in Task B?
#   "forced_subgoal":  "go to the sinkbasin",   # subgoal to force-feed
#   "oracle_action":   "go to sinkbasin 1"      # oracle action for exec test
# }


def _parse_output(text: str) -> Tuple[str, str, str]:
    """Return (switch, subgoal, action)."""
    sw_m = re.search(r"<switch>(.*?)</switch>", text, re.DOTALL | re.IGNORECASE)
    sg_m = re.search(r"<subgoal>(.*?)</subgoal>", text, re.DOTALL | re.IGNORECASE)
    ac_m = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    return (
        (sw_m.group(1).strip().upper() if sw_m else "KEEP"),
        (sg_m.group(1).strip() if sg_m else ""),
        (ac_m.group(1).strip() if ac_m else ""),
    )


def _subgoal_match(predicted: str, correct: str) -> bool:
    """Soft match: both contain the same main verb+object."""
    p, c = predicted.lower(), correct.lower()
    # Extract key words (verbs and nouns)
    key_words = set(re.findall(r"\b\w{3,}\b", c)) - {"the", "and", "with", "for", "some"}
    if not key_words:
        return p == c
    return len(key_words & set(re.findall(r"\b\w{3,}\b", p))) / len(key_words) >= 0.5


def _action_advances_subgoal(action: str, subgoal: str, oracle_action: str) -> bool:
    """Check if the action is consistent with advancing toward the subgoal."""
    a_l, o_l = action.lower(), oracle_action.lower()
    # Exact match or high word overlap
    if a_l == o_l:
        return True
    a_words = set(re.findall(r"\b\w{3,}\b", a_l))
    o_words = set(re.findall(r"\b\w{3,}\b", o_l))
    if o_words and len(a_words & o_words) / len(o_words) >= 0.6:
        return True
    return False


# ── Task A – Subgoal Attribution ─────────────────────────────────────────────

def run_task_a(
    scenarios: List[Dict],
    llm,
    params,
    tokenizer,
    num_samples: int = 1,
) -> Dict[str, Any]:
    """
    For each scenario, prompt the model and measure P(correct_subgoal).
    Returns per-task-type accuracy and overall.
    """
    by_type: Dict[str, List[bool]] = defaultdict(list)

    for sc in scenarios:
        if sc.get("executor_test") and not sc.get("correct_subgoal"):
            continue  # skip exec-only scenarios

        prompt = _make_prompt_a(sc, tokenizer)
        outputs = llm.generate([prompt] * num_samples, params)

        hits = []
        for out in outputs:
            text = out.outputs[0].text
            sw, sg, _ = _parse_output(text)
            # Check switch decision
            sw_ok = (sw == sc.get("correct_switch", "SWITCH"))
            # Check subgoal
            sg_ok = _subgoal_match(sg, sc.get("correct_subgoal", ""))
            hits.append(sw_ok and sg_ok)

        by_type[sc["task_type"]].append(float(np.mean(hits)))

    results: Dict[str, Any] = {
        "task_a_per_type": {tt: float(np.mean(v)) for tt, v in by_type.items()},
        "task_a_overall": float(np.mean([v for vs in by_type.values() for v in vs])),
    }
    return results


def _make_prompt_a(sc: Dict, tokenizer) -> str:
    content = f"Task: {sc['task_desc']}\nObservation: {sc['observation']}"
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ── Task B – Executor Isolation ───────────────────────────────────────────────

def run_task_b(
    scenarios: List[Dict],
    llm,
    params,
    tokenizer,
    num_samples: int = 1,
) -> Dict[str, Any]:
    """
    Force-feed forced_subgoal to the model and measure action quality.
    """
    exec_scenarios = [s for s in scenarios if s.get("executor_test")]
    by_type: Dict[str, List[bool]] = defaultdict(list)

    for sc in exec_scenarios:
        forced_sg = sc.get("forced_subgoal", sc.get("correct_subgoal", ""))
        oracle_ac = sc.get("oracle_action", "")
        prompt = _make_prompt_b(sc, forced_sg, tokenizer)
        outputs = llm.generate([prompt] * num_samples, params)

        hits = []
        for out in outputs:
            text = out.outputs[0].text
            _, _, action = _parse_output(text)
            hits.append(_action_advances_subgoal(action, forced_sg, oracle_ac))

        by_type[sc["task_type"]].append(float(np.mean(hits)))

    results: Dict[str, Any] = {
        "task_b_per_type": {tt: float(np.mean(v)) for tt, v in by_type.items()},
        "task_b_overall":  float(np.mean([v for vs in by_type.values() for v in vs])) if by_type else 0.0,
    }
    return results


def _make_prompt_b(sc: Dict, forced_subgoal: str, tokenizer) -> str:
    content = (
        f"Task: {sc['task_desc']}\n"
        f"Observation: {sc['observation']}\n"
        f"Current subgoal: {forced_subgoal}"
    )
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HSPO diagnostic experiments A & B")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True, help="JSONL of diagnostic scenarios")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per scenario (for stochastic models)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=0.8)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    scenarios = []
    with open(args.data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    print(f"Loaded {len(scenarios)} diagnostic scenarios")

    res_a = run_task_a(scenarios, llm, params, tokenizer, args.num_samples)
    res_b = run_task_b(scenarios, llm, params, tokenizer, args.num_samples)

    results = {**res_a, **res_b, "model": args.model_path, "num_scenarios": len(scenarios)}

    print("\n=== Task A: Subgoal Attribution ===")
    print(f"Overall P(correct subgoal): {res_a['task_a_overall']:.3f}")
    for tt, v in sorted(res_a["task_a_per_type"].items()):
        print(f"  {tt:12s}: {v:.3f}")

    print("\n=== Task B: Executor Isolation ===")
    print(f"Overall ExecutorSuccess:    {res_b['task_b_overall']:.3f}")
    for tt, v in sorted(res_b["task_b_per_type"].items()):
        print(f"  {tt:12s}: {v:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
