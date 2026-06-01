# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
ALFWorld Rule-based Process Reward Model.

Covers all 6 ALFWorld task types:
  Pick & Place         – FIND_OBJECT, PICK_OBJECT, GO_TO_RECEPTACLE, PLACE_OBJECT
  Examine in Light     – FIND_OBJECT, PICK_OBJECT, FIND_LIGHT, TOGGLE_LIGHT, EXAMINE_OBJECT
  Clean & Place        – FIND_OBJECT, PICK_OBJECT, GO_TO_TOOL(sink), CLEAN_OBJECT, ...
  Heat & Place         – FIND_OBJECT, PICK_OBJECT, GO_TO_TOOL(microwave), HEAT_OBJECT, ...
  Cool & Place         – FIND_OBJECT, PICK_OBJECT, GO_TO_TOOL(fridge), COOL_OBJECT, ...
  Pick Two & Place     – repeats pick+place twice

State dict keys produced by AlfworldEnvWrapper (see agent_system/environments/…):
  location        : str          current room/container
  inventory       : list[str]    objects held
  visible_objects : list[str]    objects visible from current location
  object_location : dict[str,str] canonical location of known objects
  object_states   : dict[str, dict[str, bool]]  per-object properties
  location_type   : str          coarse type ("sinkbasin", "microwave", "fridge", …)
  valid_actions   : list[str]    admissible action strings

task_meta keys:
  target_obj      : str
  target_obj2     : str (Pick2 only)
  target_receptacle: str
  task_type       : str  e.g. "clean", "heat", "cool", "pick", "look", "pick2"
  requires_clean  : bool
  requires_heat   : bool
  requires_cool   : bool
"""

from __future__ import annotations

from typing import Any, Dict, List

from hspo.prm.base import PRMBase, PRMOutput


# ─────────────────────────────────────────────────────────────────────────────
# Canonical subgoal types
# ─────────────────────────────────────────────────────────────────────────────

SUBGOAL_TYPES = {
    "FIND_OBJECT",
    "PICK_OBJECT",
    "GO_TO_TOOL",
    "APPLY_TOOL",    # generic: covers CLEAN/HEAT/COOL
    "CLEAN_OBJECT",
    "HEAT_OBJECT",
    "COOL_OBJECT",
    "GO_TO_RECEPTACLE",
    "PLACE_OBJECT",
    "FIND_LIGHT",
    "TOGGLE_LIGHT",
    "EXAMINE_OBJECT",
    # Pick2
    "FIND_OBJECT2",
    "PICK_OBJECT2",
    "PLACE_OBJECT2",
}


def _contains(collection: List[str], item: str) -> bool:
    """Case-insensitive membership check."""
    item_l = item.lower()
    return any(item_l in x.lower() or x.lower() in item_l for x in collection)


def _location_type(state: Dict[str, Any]) -> str:
    return state.get("location_type", "").lower()


def _in_inventory(state: Dict[str, Any], obj: str) -> bool:
    return _contains(state.get("inventory", []), obj)


def _is_visible(state: Dict[str, Any], obj: str) -> bool:
    return _contains(state.get("visible_objects", []), obj)


def _obj_state(state: Dict[str, Any], obj: str, prop: str) -> bool:
    obj_l = obj.lower()
    for k, v in state.get("object_states", {}).items():
        if obj_l in k.lower() or k.lower() in obj_l:
            return bool(v.get(prop, False))
    return False


def _obj_location(state: Dict[str, Any], obj: str) -> str:
    obj_l = obj.lower()
    for k, v in state.get("object_location", {}).items():
        if obj_l in k.lower() or k.lower() in obj_l:
            return v.lower()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-subgoal-type progress functions
# ─────────────────────────────────────────────────────────────────────────────

def _progress_find_object(state: Dict, obj: str) -> float:
    if _in_inventory(state, obj):
        return 1.0
    if _is_visible(state, obj):
        return 0.8
    known_loc = _obj_location(state, obj)
    if known_loc:
        return 0.5
    return 0.0


def _progress_pick_object(state: Dict, obj: str) -> float:
    if _in_inventory(state, obj):
        return 1.0
    if _is_visible(state, obj):
        return 0.6
    return 0.0


def _progress_go_to_tool(state: Dict, tool_type: str) -> float:
    """tool_type: 'sinkbasin' | 'microwave' | 'fridge'"""
    lt = _location_type(state)
    if tool_type in lt:
        return 1.0
    # Tool visible but not at it yet
    vis = [x.lower() for x in state.get("visible_objects", [])]
    if any(tool_type in x for x in vis):
        return 0.5
    return 0.0


def _progress_clean(state: Dict, obj: str) -> float:
    if _obj_state(state, obj, "clean"):
        return 1.0
    lt = _location_type(state)
    if _in_inventory(state, obj) and "sinkbasin" in lt:
        return 0.8
    if _in_inventory(state, obj):
        return 0.6
    if _is_visible(state, obj):
        return 0.3
    return 0.0


def _progress_heat(state: Dict, obj: str) -> float:
    if _obj_state(state, obj, "hot") or _obj_state(state, obj, "cooked"):
        return 1.0
    lt = _location_type(state)
    if _in_inventory(state, obj) and "microwave" in lt:
        return 0.8
    if _in_inventory(state, obj):
        return 0.6
    if _is_visible(state, obj):
        return 0.3
    return 0.0


def _progress_cool(state: Dict, obj: str) -> float:
    if _obj_state(state, obj, "cold"):
        return 1.0
    lt = _location_type(state)
    if _in_inventory(state, obj) and "fridge" in lt:
        return 0.8
    if _in_inventory(state, obj):
        return 0.6
    if _is_visible(state, obj):
        return 0.3
    return 0.0


def _progress_go_to_receptacle(state: Dict, receptacle: str) -> float:
    lt = _location_type(state)
    r_l = receptacle.lower()
    if r_l in lt or any(r_l in x.lower() for x in state.get("location", "").lower().split()):
        return 1.0
    if _is_visible(state, receptacle):
        return 0.6
    return 0.0


def _progress_place(state: Dict, obj: str, receptacle: str) -> float:
    obj_loc = _obj_location(state, obj)
    r_l = receptacle.lower()
    if obj_loc and (r_l in obj_loc or obj_loc in r_l):
        return 1.0
    if _in_inventory(state, obj) and _is_visible(state, receptacle):
        return 0.8
    if _in_inventory(state, obj):
        return 0.6
    return 0.0


def _progress_examine(state: Dict, obj: str) -> float:
    # "examine" succeeds if the model explicitly examines; we use inventory + light toggle as proxy
    if _in_inventory(state, obj):
        return 0.8
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Done signals (binary for rules)
# ─────────────────────────────────────────────────────────────────────────────

def _done_for_type(subgoal_type: str, state: Dict, task_meta: Dict) -> float:
    obj  = task_meta.get("target_obj", "")
    obj2 = task_meta.get("target_obj2", "")
    rec  = task_meta.get("target_receptacle", "")

    if subgoal_type == "FIND_OBJECT":
        return 1.0 if (_is_visible(state, obj) or _in_inventory(state, obj)) else 0.0

    if subgoal_type == "PICK_OBJECT":
        return 1.0 if _in_inventory(state, obj) else 0.0

    if subgoal_type in ("GO_TO_TOOL", "APPLY_TOOL"):
        tool = _tool_for_task(task_meta)
        return 1.0 if (tool and tool in _location_type(state)) else 0.0

    if subgoal_type == "CLEAN_OBJECT":
        return 1.0 if _obj_state(state, obj, "clean") else 0.0

    if subgoal_type == "HEAT_OBJECT":
        return 1.0 if (_obj_state(state, obj, "hot") or _obj_state(state, obj, "cooked")) else 0.0

    if subgoal_type == "COOL_OBJECT":
        return 1.0 if _obj_state(state, obj, "cold") else 0.0

    if subgoal_type in ("GO_TO_RECEPTACLE",):
        return 1.0 if _is_visible(state, rec) else 0.0

    if subgoal_type == "PLACE_OBJECT":
        obj_loc = _obj_location(state, obj)
        return 1.0 if (rec.lower() in obj_loc or obj_loc in rec.lower()) else 0.0

    if subgoal_type == "FIND_OBJECT2":
        return 1.0 if (_is_visible(state, obj2) or _in_inventory(state, obj2)) else 0.0

    if subgoal_type == "PICK_OBJECT2":
        return 1.0 if _in_inventory(state, obj2) else 0.0

    if subgoal_type == "PLACE_OBJECT2":
        obj_loc = _obj_location(state, obj2)
        return 1.0 if (rec.lower() in obj_loc or obj_loc in rec.lower()) else 0.0

    if subgoal_type == "EXAMINE_OBJECT":
        return 1.0 if _in_inventory(state, obj) else 0.0

    return 0.0


def _tool_for_task(task_meta: Dict) -> str:
    if task_meta.get("requires_clean"):
        return "sinkbasin"
    if task_meta.get("requires_heat"):
        return "microwave"
    if task_meta.get("requires_cool"):
        return "fridge"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Side-effect (minimal version)
# ─────────────────────────────────────────────────────────────────────────────

def _side_effect(state: Dict, task_meta: Dict) -> float:
    """
    Compute a cumulative side-effect penalty in [0, 1].

    Penalises:
      • target object dropped outside the intended receptacle
      • target object put into a wrong container (e.g. fridge when task=clean)
      • target object state changed contrary to task (e.g. cooled when task=heat)
    """
    obj = task_meta.get("target_obj", "")
    rec = task_meta.get("target_receptacle", "")
    penalty = 0.0

    obj_loc = _obj_location(state, obj)

    # Dropped and not in inventory and not at target receptacle
    if obj and not _in_inventory(state, obj) and obj_loc and rec.lower() not in obj_loc:
        penalty += 0.3

    # Placed in a wrong tool (e.g. fridge during clean task)
    if task_meta.get("requires_clean") and "fridge" in obj_loc:
        penalty += 0.5
    if task_meta.get("requires_heat") and "fridge" in obj_loc:
        penalty += 0.5
    if task_meta.get("requires_cool") and "microwave" in obj_loc:
        penalty += 0.5

    # Object state inconsistent with task goal
    if task_meta.get("requires_clean") and _obj_state(state, obj, "cold"):
        penalty += 0.2
    if task_meta.get("requires_heat") and _obj_state(state, obj, "cold"):
        penalty += 0.2

    return min(penalty, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main PRM class
# ─────────────────────────────────────────────────────────────────────────────

class AlfworldRulePRM(PRMBase):
    """
    Rule-based PRM for ALFWorld. No neural network; all functions are
    deterministic state-machine rules derived from the task specification.

    Usage::

        prm = AlfworldRulePRM()
        out = prm.score(
            subgoal_type="CLEAN_OBJECT",
            subgoal_text="clean the cup in the sinkbasin",
            state_before=state_t,
            state_after=state_t1,
            action="clean cup 1 with sinkbasin 1",
            task_meta={"target_obj": "cup 1", "target_receptacle": "cabinet 1",
                       "requires_clean": True},
        )
        reward = out.low_level_reward()
    """

    def score(
        self,
        subgoal_type: str,
        subgoal_text: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        action: str,
        task_meta: Dict[str, Any],
    ) -> PRMOutput:
        obj  = task_meta.get("target_obj", "")
        obj2 = task_meta.get("target_obj2", "")
        rec  = task_meta.get("target_receptacle", "")
        st   = subgoal_type.upper()

        # ── progress ──────────────────────────────────────────────────── #
        p_before = self._progress(st, state_before, obj, obj2, rec, task_meta)
        p_after  = self._progress(st, state_after,  obj, obj2, rec, task_meta)

        # ── done ──────────────────────────────────────────────────────── #
        done_after = _done_for_type(st, state_after, task_meta)

        # ── validity ──────────────────────────────────────────────────── #
        valid_actions = state_after.get("valid_actions", None)
        if valid_actions is not None:
            # If the environment tells us the set of valid actions after the step,
            # we can compare the action *before* to the set *before*.
            # We use state_before's valid_actions for the validity check.
            va_before = state_before.get("valid_actions", [])
            valid_flag = float(_contains(va_before, action)) if va_before else 1.0
        else:
            # Fallback: check for common ALFWorld failure indicators in the observation
            next_obs = state_after.get("last_feedback", "")
            invalid_phrases = [
                "nothing happens", "you can't", "invalid action",
                "that's not something you can pick", "you need to",
            ]
            next_obs_l = next_obs.lower()
            valid_flag = float(not any(ph in next_obs_l for ph in invalid_phrases))

        # ── side effect ───────────────────────────────────────────────── #
        side_before = _side_effect(state_before, task_meta)
        side_after  = _side_effect(state_after,  task_meta)

        return PRMOutput(
            progress_before=p_before,
            progress_after=p_after,
            done_after=done_after,
            valid=valid_flag,
            side_effect_before=side_before,
            side_effect_after=side_after,
        )

    # ------------------------------------------------------------------ #
    # Dispatcher                                                           #
    # ------------------------------------------------------------------ #

    def _progress(
        self,
        st: str,
        state: Dict,
        obj: str,
        obj2: str,
        rec: str,
        task_meta: Dict,
    ) -> float:
        if st == "FIND_OBJECT":
            return _progress_find_object(state, obj)
        if st == "PICK_OBJECT":
            return _progress_pick_object(state, obj)
        if st == "GO_TO_TOOL":
            return _progress_go_to_tool(state, _tool_for_task(task_meta))
        if st in ("APPLY_TOOL", "CLEAN_OBJECT"):
            return _progress_clean(state, obj)
        if st == "HEAT_OBJECT":
            return _progress_heat(state, obj)
        if st == "COOL_OBJECT":
            return _progress_cool(state, obj)
        if st == "GO_TO_RECEPTACLE":
            return _progress_go_to_receptacle(state, rec)
        if st == "PLACE_OBJECT":
            return _progress_place(state, obj, rec)
        if st == "FIND_OBJECT2":
            return _progress_find_object(state, obj2)
        if st == "PICK_OBJECT2":
            return _progress_pick_object(state, obj2)
        if st == "PLACE_OBJECT2":
            return _progress_place(state, obj2, rec)
        if st in ("FIND_LIGHT", "TOGGLE_LIGHT"):
            # Heuristic: any light source visible
            vis_l = [x.lower() for x in state.get("visible_objects", [])]
            return 1.0 if any("lamp" in x or "light" in x or "candle" in x for x in vis_l) else 0.0
        if st == "EXAMINE_OBJECT":
            return _progress_examine(state, obj)
        # Unknown type – neutral
        return 0.5
