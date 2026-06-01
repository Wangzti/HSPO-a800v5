#!/usr/bin/env python3
# Copyright 2025 HSPO Authors
# Licensed under the Apache License, Version 2.0
"""
ALFWorld Evaluation Script for HSPO.

Rolls out a trained HSPO policy (or any HF-compatible model) on all 6 ALFWorld
task types and reports:
  - Overall success rate
  - Per-task-type success rate
  - Mean episode length
  - Mean number of SWITCH events per episode
  - Mean PRM score (optional, requires --score_prm flag)

The script uses the vLLM offline inference engine for fast batch rollout.

Usage
-----
    python evaluation/eval_alfworld.py \\
        --model_path /mnt/nfs/ztwang/checkpoints/hspo/rl/... \\
        --num_episodes 200 \\
        [--max_steps 50] \\
        [--temperature 0.0] \\
        [--score_prm]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure HSPO-agent on path
_HSPO_ROOT = Path(__file__).resolve().parent.parent
if str(_HSPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSPO_ROOT))

import numpy as np


def _load_model(model_path: str, temperature: float, max_new_tokens: int):
    """Load vLLM offline LLM. Returns (llm, sampling_params)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise RuntimeError("vllm not installed. Run: pip install vllm")
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.8)
    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    return llm, params


def _make_prompt(task_desc: str, obs: str, subgoal: str, tokenizer) -> str:
    """Apply chat template."""
    content = f"Task: {task_desc}\nObservation: {obs}"
    if subgoal:
        content += f"\nCurrent subgoal: {subgoal}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _parse_response(text: str):
    """Return (switch, subgoal, action) from plan-execute text."""
    import re
    sw_m = re.search(r"<switch>(.*?)</switch>", text, re.IGNORECASE | re.DOTALL)
    sg_m = re.search(r"<subgoal>(.*?)</subgoal>", text, re.IGNORECASE | re.DOTALL)
    ac_m = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    sw = sw_m.group(1).strip().upper() if sw_m else "KEEP"
    sg = sg_m.group(1).strip() if sg_m else ""
    ac = ac_m.group(1).strip() if ac_m else text.strip()
    return sw, sg, ac


def run_episode(env, llm, params, tokenizer, task_desc: str, max_steps: int) -> Dict[str, Any]:
    """Run one episode and return metrics."""
    from hspo.parser import PlanExecuteParser
    from hspo.prm.alfworld_prm import AlfworldRulePRM
    from hspo.env_state import parse_alfworld_state, build_task_meta

    parser = PlanExecuteParser()
    prm = AlfworldRulePRM()

    obs, info = env.reset()
    obs_text = obs if isinstance(obs, str) else obs.get("text", "")
    task_meta = build_task_meta(task_desc)

    current_subgoal = ""
    num_switches = 0
    prm_scores = []
    prev_state = parse_alfworld_state(obs_text, info.get("admissible_commands"), "")

    for t in range(max_steps):
        prompt = _make_prompt(task_desc, obs_text, current_subgoal, tokenizer)
        out = llm.generate([prompt], params)[0].outputs[0].text

        sw, sg, action = _parse_response(out)

        if sw == "SWITCH" or not current_subgoal:
            current_subgoal = sg or current_subgoal
            num_switches += 1

        next_obs, reward, done, info = env.step(action)
        next_obs_text = next_obs if isinstance(next_obs, str) else next_obs.get("text", "")
        next_state = parse_alfworld_state(
            next_obs_text,
            info.get("admissible_commands"),
            info.get("feedback", ""),
        )

        # Compute PRM score for diagnostics
        try:
            from hspo.segmentation import _infer_subgoal_type
            sg_type = _infer_subgoal_type(current_subgoal)
            prm_out = prm.score(
                subgoal_type=sg_type,
                subgoal_text=current_subgoal,
                state_before=prev_state,
                state_after=next_state,
                action=action,
                task_meta=task_meta,
            )
            prm_scores.append(prm_out.progress_after - prm_out.progress_before)
        except Exception:
            pass

        prev_state = next_state
        obs_text = next_obs_text

        if done:
            return {
                "success": float(reward) > 0.5,
                "steps": t + 1,
                "switches": num_switches,
                "mean_prm_delta": float(np.mean(prm_scores)) if prm_scores else 0.0,
            }

    return {
        "success": False,
        "steps": max_steps,
        "switches": num_switches,
        "mean_prm_delta": float(np.mean(prm_scores)) if prm_scores else 0.0,
    }


def evaluate(args) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    import alfworld.agents.environment as alfenv

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm, params = _load_model(args.model_path, args.temperature, args.max_new_tokens)

    # Load ALFWorld env
    alfworld_data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    env = alfenv.TextWorldEnv(config={
        "data_path": alfworld_data,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
    })

    results_by_type: Dict[str, List[Dict]] = defaultdict(list)

    for ep_idx in range(args.num_episodes):
        task_desc = env.current_task_description()
        task_type = env.current_task_type()
        metrics = run_episode(env, llm, params, tokenizer, task_desc, args.max_steps)
        metrics["task_type"] = task_type
        results_by_type[task_type].append(metrics)
        print(f"[{ep_idx+1}/{args.num_episodes}] {task_type}: success={metrics['success']}, steps={metrics['steps']}")

    # Aggregate
    all_results = [m for recs in results_by_type.values() for m in recs]
    summary = {
        "overall_success": float(np.mean([r["success"] for r in all_results])),
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "mean_switches": float(np.mean([r["switches"] for r in all_results])),
        "mean_prm_delta": float(np.mean([r["mean_prm_delta"] for r in all_results])),
        "per_task_type": {},
    }
    for tt, recs in results_by_type.items():
        summary["per_task_type"][tt] = {
            "success":  float(np.mean([r["success"] for r in recs])),
            "n":        len(recs),
            "mean_steps": float(np.mean([r["steps"] for r in recs])),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="HSPO ALFWorld evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    summary = evaluate(args)

    print("\n=== HSPO ALFWorld Evaluation Results ===")
    print(f"Overall success rate : {summary['overall_success']:.3f}")
    print(f"Mean episode length  : {summary['mean_steps']:.1f}")
    print(f"Mean #SWITCHes       : {summary['mean_switches']:.2f}")
    print(f"Mean PRM Δprogress   : {summary['mean_prm_delta']:.4f}")
    print("\nPer task-type:")
    for tt, s in summary["per_task_type"].items():
        print(f"  {tt:12s}: success={s['success']:.3f} (n={s['n']}, avg_steps={s['mean_steps']:.1f})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
