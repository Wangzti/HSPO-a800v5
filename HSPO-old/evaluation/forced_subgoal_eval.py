#!/usr/bin/env python3
# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
Forced-subgoal executor evaluation for HSPO low-level training.

This is a lightweight P3 diagnostic. It does not replace full ALFWorld
environment rollout, but it lets us compare an Executor SFT checkpoint and a
low-level HSPO checkpoint on the same forced-subgoal scenarios.

Expected JSONL schema:
{
  "scenario_id": "clean_cup_001",
  "task_type": "clean",
  "task_desc": "clean a cup and put it in cabinet",
  "observation": "...",
  "forced_subgoal": "clean the cup using the sinkbasin",
  "oracle_action": "clean cup 1 with sinkbasin 1",
  "valid_actions": ["clean cup 1 with sinkbasin 1", "..."],
  "switch_target": "KEEP"
}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HSPO_ROOT = Path(__file__).resolve().parent.parent
if str(_HSPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSPO_ROOT))

from hspo.parser import PlanExecuteParser


def _word_overlap(a: str, b: str) -> float:
    aw = set(re.findall(r"\b\w{3,}\b", a.lower()))
    bw = set(re.findall(r"\b\w{3,}\b", b.lower()))
    if not bw:
        return float(a.strip().lower() == b.strip().lower())
    return len(aw & bw) / len(bw)


def _action_matches(action: str, oracle_action: str) -> bool:
    if not oracle_action:
        return False
    if action.strip().lower() == oracle_action.strip().lower():
        return True
    return _word_overlap(action, oracle_action) >= 0.6


def _make_prompt(sc: Dict[str, Any], tokenizer) -> str:
    valid_actions = sc.get("valid_actions") or []
    valid_str = "\n ".join(f"'{a}'" for a in valid_actions)
    content = (
        f"Task: {sc['task_desc']}\n"
        f"Observation: {sc['observation']}\n"
        f"Current subgoal: {sc['forced_subgoal']}\n"
    )
    if valid_str:
        content += f"Admissible actions:\n[{valid_str}]\n"
    content += (
        "Output exactly:\n"
        "<switch>KEEP or SWITCH</switch>\n"
        "<subgoal>copy the current subgoal unless switching</subgoal>\n"
        "<action>one admissible action</action>"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _switch_metrics(preds: List[str], targets: List[str]) -> Dict[str, float]:
    if not targets:
        return {"switch_precision": 0.0, "switch_recall": 0.0, "switch_f1": 0.0}
    tp = sum(p == "SWITCH" and t == "SWITCH" for p, t in zip(preds, targets))
    fp = sum(p == "SWITCH" and t != "SWITCH" for p, t in zip(preds, targets))
    fn = sum(p != "SWITCH" and t == "SWITCH" for p, t in zip(preds, targets))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "switch_precision": float(precision),
        "switch_recall": float(recall),
        "switch_f1": float(f1),
    }


def evaluate(args) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    scenarios = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    parser = PlanExecuteParser()

    prompts = [_make_prompt(sc, tokenizer) for sc in scenarios]
    outputs = llm.generate(prompts, params)

    action_hits = []
    invalids = []
    format_valid = []
    missing_switch = []
    missing_subgoal = []
    pred_switches = []
    switch_targets = []
    premature = []
    delayed = []

    rows = []
    for sc, out in zip(scenarios, outputs):
        text = out.outputs[0].text
        parsed = parser.parse(text)
        action = parsed.action or ""
        switch = parsed.switch or "KEEP"
        valid_actions = [a.lower() for a in sc.get("valid_actions", [])]
        target = sc.get("switch_target")

        hit = _action_matches(action, sc.get("oracle_action", ""))
        invalid = bool(valid_actions and action.lower() not in valid_actions)

        action_hits.append(hit)
        invalids.append(invalid)
        format_valid.append(parsed.valid_format)
        missing_switch.append("missing_switch" in parsed.errors)
        missing_subgoal.append("missing_subgoal" in parsed.errors)
        pred_switches.append(switch)
        if target:
            switch_targets.append(target)
            premature.append(switch == "SWITCH" and target == "KEEP")
            delayed.append(switch == "KEEP" and target == "SWITCH")

        rows.append(
            {
                "scenario_id": sc.get("scenario_id", ""),
                "task_type": sc.get("task_type", ""),
                "forced_subgoal": sc.get("forced_subgoal", ""),
                "oracle_action": sc.get("oracle_action", ""),
                "model_output": text,
                "parsed_switch": switch,
                "parsed_action": action,
                "action_match": hit,
                "invalid_action": invalid,
                "valid_format": parsed.valid_format,
                "errors": parsed.errors,
            }
        )

    summary: Dict[str, Any] = {
        "model": args.model_path,
        "num_scenarios": len(scenarios),
        "forced_subgoal_success": float(np.mean(action_hits)) if action_hits else 0.0,
        "invalid_action_rate": float(np.mean(invalids)) if invalids else 0.0,
        "format_valid_rate": float(np.mean(format_valid)) if format_valid else 0.0,
        "missing_switch_rate": float(np.mean(missing_switch)) if missing_switch else 0.0,
        "missing_subgoal_rate": float(np.mean(missing_subgoal)) if missing_subgoal else 0.0,
        "premature_switch_rate": float(np.mean(premature)) if premature else 0.0,
        "delayed_switch_rate": float(np.mean(delayed)) if delayed else 0.0,
    }
    if switch_targets:
        summary.update(_switch_metrics(pred_switches[: len(switch_targets)], switch_targets))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HSPO forced-subgoal executor evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()

    summary = evaluate(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
