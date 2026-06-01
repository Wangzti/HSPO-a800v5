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
Collect ALFWorld expert trajectories for SFT warm-up.

Runs HandCodedTWAgent through the TextWorld (TW) environment for each game
file in the ALFWorld training split. Records the observation, admissible
actions, expert action, subgoal text, and SWITCH/KEEP decision at every step.
Saves successful (won=True) trajectories as individual JSON files.

Prerequisites
-------------
- ALFWORLD_DATA env var must point to the ALFWorld data root
  (default: ~/.cache/alfworld).
- Run under the `verl` conda env:
    conda activate verl
    ALFWORLD_DATA=~/.cache/alfworld python3 example_scripts/data_preprocess/collect_alfworld_sft_demos.py

Output
------
One JSON file per successful episode at --out_dir:
    {
        "task_desc": str,           # human task description
        "gamefile": str,            # path to game.tw-pddl
        "task_type": str,
        "steps": [
            {
                "obs": str,                 # raw TextWorld observation
                "admissible": [str, ...],   # valid text commands (filtered, no 'help')
                "action": str,              # action taken by expert
                "subgoal": str,             # NL subgoal for this segment
                "subgoal_idx": int,         # high_pddl index
                "switch": str,              # "SWITCH" or "KEEP"
            },
            ...
        ],
        "won": true
    }
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

# Ensure trajectories package is importable from any working directory
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Subgoal text fallbacks when human annotations are absent
# ---------------------------------------------------------------------------
SUBGOAL_TEMPLATE = {
    "GotoLocation":              "Go to the {0}",
    "PickupObject":              "Pick up the {0}",
    "PutObject":                 "Place the {0} on the {1}",
    "HeatObject":                "Heat the {0} using the microwave",
    "CoolObject":                "Cool the {0} in the fridge",
    "CleanObject":               "Clean the {0} in the sink",
    "SliceObject":               "Slice the {0}",
    "ToggleObject":              "Toggle the {0}",
    "ExamineObject":             "Examine the {0} closely",
    "PickAndPlaceSimplePolicy":  "Pick and place the object",
    "NoOp":                      "Complete the task",
}


def _subgoal_text_from_high_pddl(step: dict) -> str:
    """Map a high_pddl entry to a natural language subgoal string."""
    action = step.get("discrete_action", {}).get("action", "NoOp")
    args   = step.get("discrete_action", {}).get("args", [])
    tpl    = SUBGOAL_TEMPLATE.get(action, "Perform the next step")
    try:
        return tpl.format(*args) if args else tpl
    except (IndexError, KeyError):
        return tpl


def _load_traj_meta(gamefile: str) -> dict:
    """Load traj_data.json from the directory containing game.tw-pddl."""
    traj_path = Path(gamefile).parent / "traj_data.json"
    if not traj_path.exists():
        return {}
    with open(traj_path) as f:
        return json.load(f)


def _extract_task_desc(obs: str) -> str:
    """Extract task description from the initial TextWorld observation."""
    marker = "Your task is to: "
    idx = obs.find(marker)
    if idx != -1:
        # take text up to the next newline
        end = obs.find("\n", idx + len(marker))
        return obs[idx + len(marker): end].strip() if end != -1 else obs[idx + len(marker):].strip()
    return ""


def collect(
    config_path: str,
    num_games: int,
    out_dir: str,
    expert_timeout: int,
    seed: int,
    traj_log_dir: str | None = None,
    model_name: str = "handcoded-expert",
    max_per_task: int = 0,
) -> None:
    import random as _random
    import yaml
    import gym
    import textworld
    import textworld.gym
    from alfworld.agents.environment import get_environment
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from alfworld.agents.expert import HandCodedTWAgent

    from trajectories.traj_logger import TrajLogger, make_meta, make_outcome, make_step

    os.makedirs(out_dir, exist_ok=True)

    _traj_logger: TrajLogger | None = None
    if traj_log_dir is not None:
        _traj_logger = TrajLogger(traj_log_dir)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Collect solvable game files via AlfredTWEnv
    base_env = get_environment(cfg["env"]["type"])(cfg, train_eval="train")
    game_files = base_env.game_files
    print(f"Found {len(game_files)} solvable training games.")

    # Shuffle to avoid task-type ordering bias
    _random.seed(seed)
    _random.shuffle(game_files)

    # Per-game request: needs facts for HandCodedAgent predicates
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        feedback=True,
        facts=True,
        extras=["gamefile"],
    )

    n_success = 0
    n_fail = 0
    per_task_success = {}
    use_stratified = max_per_task > 0

    if use_stratified:
        print(f"Stratified sampling: max {max_per_task} per task type across all {len(game_files)} games")
    else:
        print(f"Collecting up to {num_games} successful trajectories")

    for game_file in game_files:
        if not use_stratified and n_success >= num_games:
            break

        # Pre-check task type for stratified sampling
        meta = _load_traj_meta(game_file)
        task_type = meta.get("task_type", "unknown")

        if use_stratified:
            if per_task_success.get(task_type, 0) >= max_per_task:
                continue

        # Load traj metadata (subgoal descriptions, task type)
        high_pddl  = meta.get("plan", {}).get("high_pddl", [])
        try:
            high_descs = meta["turk_annotations"]["anns"][0]["high_descs"]
        except (KeyError, IndexError):
            high_descs = []

        # Build a single-game gym env with needed wrappers
        env_id = textworld.gym.register_game(
            game_file,
            request_infos,
            max_episode_steps=expert_timeout,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        env = textworld.gym.make(env_id)

        # Create expert, reset it using the game file so it reads traj_data.json
        expert = HandCodedTWAgent(max_steps=expert_timeout)
        expert.reset(game_file)

        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"  [skip] reset failed for {game_file}: {e}")
            env.close()
            n_fail += 1
            continue

        task_desc = _extract_task_desc(obs) or meta.get("turk_annotations", {}).get(
            "anns", [{}])[0].get("task_desc", "")

        steps = []
        prev_subgoal_idx = None
        last_action = ""
        done = False

        while not done:
            admissible = [a for a in info.get("admissible_commands", []) if a != "help"]
            if not admissible:
                break

            # Expert selects next action
            try:
                action = expert.act(info, 0, info.get("won", False), last_action)
            except Exception:
                break

            if action not in admissible:
                # Expert proposed an invalid action — pick closest or skip
                break

            # Subgoal tracking via policy.subgoal_idx
            try:
                sidx = int(expert.policy.subgoal_idx)
            except AttributeError:
                sidx = 0

            if high_descs and sidx < len(high_descs):
                subgoal = high_descs[sidx]
            elif high_pddl and sidx < len(high_pddl):
                subgoal = _subgoal_text_from_high_pddl(high_pddl[sidx])
            else:
                subgoal = "Complete the task"

            switch = "SWITCH" if sidx != prev_subgoal_idx else "KEEP"

            steps.append({
                "obs":         obs,
                "admissible":  admissible,
                "action":      action,
                "subgoal":     subgoal,
                "subgoal_idx": sidx,
                "switch":      switch,
            })

            prev_subgoal_idx = sidx
            last_action = action

            try:
                obs, _reward, done, info = env.step(action)
            except Exception as e:
                print(f"  [skip] step error in {game_file}: {e}")
                break

        env.close()

        won = bool(info.get("won", False))
        if won and len(steps) >= 2:
            record = {
                "task_desc": task_desc,
                "gamefile":  game_file,
                "task_type": task_type,
                "steps":     steps,
                "won":       True,
            }
            out_path = Path(out_dir) / f"traj_{n_success:05d}.json"
            with open(out_path, "w") as f:
                json.dump(record, f, ensure_ascii=False)

            if _traj_logger is not None:
                meta = make_meta(
                    env="alfworld",
                    model=model_name,
                    source="sft_collection",
                    task_desc=task_desc,
                    task_type=task_type,
                    gamefile=game_file,
                    episode_idx=n_success,
                )
                traj_steps = [
                    make_step(
                        t=t,
                        obs=s["obs"],
                        available_actions=s["admissible"],
                        action=s["action"],
                        subgoal=s["subgoal"],
                        switch=s["switch"],
                        reward=0.0,
                        is_valid=True,
                        done=(t == len(steps) - 1),
                    )
                    for t, s in enumerate(steps)
                ]
                outcome = make_outcome(won=True, total_reward=1.0, num_steps=len(steps))
                _traj_logger.log_episode(meta, traj_steps, outcome)

            n_success += 1
            per_task_success[task_type] = per_task_success.get(task_type, 0) + 1
            if n_success % 100 == 0:
                task_summary = " | ".join(f"{t}={c}" for t, c in sorted(per_task_success.items()))
                print(f"  {n_success} saved ({n_success + n_fail} games run) [{task_summary}]")
        else:
            n_fail += 1

    print(f"\nDone. {n_success} successful trajectories → {out_dir}")
    print(f"  Success rate: {n_success}/{n_success + n_fail} = "
          f"{100 * n_success / max(n_success + n_fail, 1):.1f}%")
    print(f"  Per task type:")
    for task, count in sorted(per_task_success.items()):
        print(f"    {task}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Collect ALFWorld SFT demo trajectories")
    parser.add_argument(
        "--config_path",
        default="agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    parser.add_argument("--num_games", type=int, default=2000,
                        help="Max successful trajectories to collect (ignored if --max_per_task is set)")
    parser.add_argument("--max_per_task", type=int, default=250,
                        help="If set, collect up to N successful trajectories per task type (stratified sampling). "
                             "Recommended: 250 for ~1500 total across 6 task types.")
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("HSPO_SFT_RAW_DIR", "/mnt/nfs/ztwang/data/hspo/sft/alfworld_raw/"),
    )
    parser.add_argument("--expert_timeout", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--traj_log_dir",
        default=None,
        help="Directory for TrajLogger output (unified trajectory storage).",
    )
    parser.add_argument("--model_name", default="handcoded-expert",
                        help="Model identifier written into trajectory metadata.")
    args = parser.parse_args()

    # Ensure ALFWORLD_DATA is set
    alfworld_data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = alfworld_data
    print(f"ALFWORLD_DATA = {alfworld_data}")

    traj_log_dir = args.traj_log_dir
    if traj_log_dir is None:
        traj_log_dir = str(
            _REPO_ROOT / "trajectories" / "alfworld" / "sft_collection" / args.model_name
        )

    collect(
        config_path=args.config_path,
        num_games=args.num_games,
        out_dir=args.out_dir,
        expert_timeout=args.expert_timeout,
        seed=args.seed,
        traj_log_dir=traj_log_dir,
        model_name=args.model_name,
        max_per_task=args.max_per_task,
    )


if __name__ == "__main__":
    main()
