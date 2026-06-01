"""
HSPO: Plan-Execute Environment Manager.

Aligned with HiPER's AlfWorldEnvironmentManagerOptions pattern:
1. Uses alfworld_projection_options / webshop_projection_options for tag parsing
2. Projection returns (actions, subgoals, switches, valids) directly
3. Memory with fetch_options for subgoal/switch history
4. Strict no-think prompts (<switch>, <subgoal>, <action> blocks only)
5. Tracks current_subgoal for ABG anchor extraction
"""

import os
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.env_manager import parse_gamefile, set_gamefile
from agent_system.memory import SimpleMemory
from recipe.hspo.prompts import (
    ALFWORLD_HSPO_TEMPLATE,
    ALFWORLD_HSPO_TEMPLATE_NO_HIS,
    WEBSHOP_HSPO_TEMPLATE,
    WEBSHOP_HSPO_TEMPLATE_NO_HIS,
)


# ---------------------------------------------------------------------------
#  HSPO AlfWorld Environment Manager (Options variant)
# ---------------------------------------------------------------------------

class HSPOAlfWorldEnvManager(EnvironmentManagerBase):
    """Plan-Execute environment manager for ALFWorld, aligned with HiPER."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

        self.current_subgoals: List[str] = []
        self.switch_history: List[bool] = []
        self.gamefile: List[str] = []

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()

        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        n = len(text_obs)
        self.current_subgoals = ["" for _ in range(n)]
        self.switch_history = [True for _ in range(n)]

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands,
            init=True,
        )
        return {
            "text": full_text_obs,
            "image": image_obs,
            "anchor": text_obs,
        }, infos

    def step(self, text_actions: List[str]):
        # Projection parses <switch>, <subgoal>, <action> from raw text
        actions, subgoals, switches, valids = self.projection_f(
            text_actions, self.envs.get_admissible_commands,
        )

        # Update subgoal state from parsed switches
        n = len(text_actions)
        for i in range(n):
            sw_str = switches[i].strip().upper() if switches[i] else "KEEP"
            if sw_str == "SWITCH":
                self.current_subgoals[i] = subgoals[i]
                self.switch_history.append(True)
            else:
                if subgoals[i] and subgoals[i] != self.current_subgoals[i]:
                    self.current_subgoals[i] = subgoals[i]
                self.switch_history.append(False)

        # Step environment with parsed actions
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({
            "text_obs": self.pre_text_obs,
            "action": actions,
            "subgoal": subgoals,
            "switch": switches,
        })
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands,
        )

        if infos and infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["switch"] = switches[i] if i < len(switches) else ""
            info["subgoal"] = subgoals[i] if i < len(subgoals) else ""
            info["decoded_action"] = actions[i] if i < len(actions) else ""

        next_observations = {
            "text": full_text_obs,
            "image": image_obs,
            "anchor": text_obs,
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")
            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: "):].strip())
            else:
                self.tasks.append(obs.strip()[:200])

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
    ) -> List[str]:
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens, subgoals, switches = self.memory.fetch_options(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
                subgoal_key="subgoal",
                switch_key="switch",
            )

        for i in range(len(text_obs)):
            reformatted = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != "help"
            )

            if init or self.config.env.history_length <= 0:
                sg = self.current_subgoals[i] if i < len(self.current_subgoals) else ""
                obs = ALFWORLD_HSPO_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    current_subgoal=sg if sg else "None",
                    admissible_actions=reformatted,
                )
            else:
                obs = ALFWORLD_HSPO_TEMPLATE.format(
                    task_description=self.tasks[i] if i < len(self.tasks) else "",
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i] if i < len(valid_lens) else 0,
                    action_history=memory_contexts[i] if i < len(memory_contexts) else "",
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    current_subgoal=subgoals[i] if i < len(subgoals) else self.current_subgoals[i],
                    admissible_actions=reformatted,
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item.get("active_masks", False):
                info = total_infos[batch_idx][i]
                won = float(info.get("won", 0))
                success["success_rate"].append(won)
                gf = info.get("extra.gamefile")
                if gf:
                    _accumulate_gamefile_success(gf, won, success)
                return


# ---------------------------------------------------------------------------
#  HSPO WebShop Environment Manager (Options variant)
# ---------------------------------------------------------------------------

class HSPOWebShopEnvManager(EnvironmentManagerBase):
    """Plan-Execute environment manager for WebShop, aligned with HiPER."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

        self.current_subgoals: List[str] = []
        self.switch_history: List[bool] = []

    def reset(self, kwargs) -> Dict:
        obs, infos = self.envs.reset()
        self.tasks = self._extract_task(obs)
        obs = self._format_obs(obs)

        n = len(obs)
        self.current_subgoals = ["" for _ in range(n)]
        self.switch_history = [True for _ in range(n)]

        observations = {
            "text": self.build_text_obs(obs, infos, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        # Projection parses <switch>, <subgoal>, <action> from raw text
        actions, subgoals, switches, valids = self.projection_f(text_actions)

        n = len(text_actions)
        for i in range(n):
            sw_str = switches[i].strip().upper() if switches[i] else "KEEP"
            if sw_str == "SWITCH":
                self.current_subgoals[i] = subgoals[i]
                self.switch_history.append(True)
            else:
                if subgoals[i] and subgoals[i] != self.current_subgoals[i]:
                    self.current_subgoals[i] = subgoals[i]
                self.switch_history.append(False)

        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_obs = self._format_obs(next_obs)

        self.memory.store({
            "text_obs": self.pre_text_obs,
            "action": actions,
            "subgoal": subgoals,
            "switch": switches,
        })
        self.pre_text_obs = next_obs

        next_observations = {
            "text": self.build_text_obs(next_obs, infos),
            "image": None,
            "anchor": next_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["switch"] = switches[i] if i < len(switches) else ""
            info["subgoal"] = subgoals[i] if i < len(subgoals) else ""
            info["decoded_action"] = actions[i] if i < len(actions) else ""

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _extract_task(self, text_obs: List[str]) -> List[str]:
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            if len(parts) > 2 and parts[1] == "Instruction:":
                tasks.append(parts[2])
            else:
                tasks.append("")
        return tasks

    def _format_obs(self, text_obs: List[str]) -> List[str]:
        formatted = []
        for i, obs in enumerate(text_obs):
            parts = obs.split(" [SEP] ")
            try:
                if i < len(self.tasks) and self.tasks[i] in parts:
                    idx = parts.index(self.tasks[i])
                    formatted.append(
                        " [SEP] ".join(f"'{p}'" for p in parts[idx + 1:])
                    )
                else:
                    formatted.append(obs)
            except Exception:
                formatted.append(obs)
        return formatted

    def _format_avail_actions(self, avail: Dict) -> List[str]:
        actions = []
        if avail.get("has_search_bar", False):
            actions.append("search[<your query>]")
        for txt in avail.get("clickables", []):
            actions.append(f"click[{txt}]")
        return actions

    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[Dict],
        init: bool = False,
    ) -> List[str]:
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens, subgoals, switches = self.memory.fetch_options(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
                subgoal_key="subgoal",
                switch_key="switch",
            )

        for i in range(len(text_obs)):
            avail_acts = self._format_avail_actions(infos[i]["available_actions"])
            reformatted = "\n".join(f"'{s}'," for s in avail_acts)

            if init or self.config.env.history_length <= 0:
                sg = self.current_subgoals[i] if i < len(self.current_subgoals) else ""
                obs = WEBSHOP_HSPO_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i] if i < len(self.tasks) else "",
                    current_observation=text_obs[i],
                    current_subgoal=sg if sg else "None",
                    available_actions=reformatted,
                )
            else:
                obs = WEBSHOP_HSPO_TEMPLATE.format(
                    task_description=self.tasks[i] if i < len(self.tasks) else "",
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i] if i < len(valid_lens) else 0,
                    action_history=memory_contexts[i] if i < len(memory_contexts) else "",
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    current_subgoal=subgoals[i] if i < len(subgoals) else self.current_subgoals[i],
                    available_actions=reformatted,
                )
                if len(obs) > 13000:
                    sg = self.current_subgoals[i] if i < len(self.current_subgoals) else ""
                    obs = WEBSHOP_HSPO_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i] if i < len(self.tasks) else "",
                        current_observation=text_obs[i],
                        current_subgoal=sg if sg else "None",
                        available_actions=reformatted,
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item.get("active_masks", False):
                info = total_infos[batch_idx][i]
                won = float(info.get("won", 0))
                score = float(info.get("task_score", 0))
                success["success_rate"].append(won)
                success["webshop_task_score (not success_rate)"].append(score)
                return


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _accumulate_gamefile_success(gamefile: str, won: float, success: Dict):
    tasks = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]
    for task in tasks:
        if task in (gamefile or ""):
            success.setdefault(f"{task}_success_rate", []).append(won)
            break


# ---------------------------------------------------------------------------
#  Env factory
# ---------------------------------------------------------------------------

def make_hspo_envs(config):
    """Create HSPO environments with Plan-Execute interface (Options variant)."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n must be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True,
    )

    if "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import (
            build_alfworld_envs, alfworld_projection_options,
        )

        alf_cfg = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "agent_system", "environments", "env_package",
            "alfworld", "configs", "config_tw.yaml",
        )
        env_kwargs = {
            "eval_dataset": config.env.alfworld.eval_dataset,
        }

        val_only = config.trainer.get("val_only", False)

        proj = partial(alfworld_projection_options)

        _envs = None if val_only else build_alfworld_envs(
            alf_cfg, config.env.seed,
            config.data.train_batch_size, group_n,
            is_train=True, env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_alfworld_envs(
            alf_cfg, config.env.seed + 1000,
            config.data.val_batch_size, 1,
            is_train=False, env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )

        train_envs = HSPOAlfWorldEnvManager(_envs, proj, config) if _envs is not None else None
        val_envs = HSPOAlfWorldEnvManager(_val_envs, proj, config)

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs, webshop_projection_options,
        )

        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "agent_system", "environments", "env_package",
            "webshop", "webshop", "data",
        )
        if config.env.webshop.use_small:
            fp = os.path.join(data_dir, "items_shuffle_1000.json")
            ap = os.path.join(data_dir, "items_ins_v2_1000.json")
        else:
            fp = os.path.join(data_dir, "items_shuffle.json")
            ap = os.path.join(data_dir, "items_ins_v2.json")

        use_small = bool(config.env.webshop.use_small)
        env_kwargs = {
            "observation_mode": "text",
            "num_products": 1000 if use_small else None,
            "human_goals": config.env.webshop.human_goals,
            "file_path": fp,
            "attr_path": ap,
            "early_terminate_penalty": config.env.webshop.get("early_terminate_penalty", 0.0),
        }

        val_only = config.trainer.get("val_only", False)

        proj = partial(webshop_projection_options)

        _envs = None if val_only else build_webshop_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n, is_train=True,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1, is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )

        train_envs = HSPOWebShopEnvManager(_envs, proj, config) if _envs is not None else None
        val_envs = HSPOWebShopEnvManager(_val_envs, proj, config)

        import time
        total_envs = config.data.train_batch_size * group_n + config.data.val_batch_size
        time.sleep(total_envs * 0.1)
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")

    return train_envs, val_envs
