# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
ALFWorld Observation → Structured State Parser

Parses the raw TextWorld observation string into a structured state dict
that the Rule-PRM can consume, since the TextWorld env does not directly
expose object-level state (inventory, object_states, location_type, …).

Typical ALFWorld observation:
    "You are in the kitchen. You see a cup 1, a plate 1 and a sinkbasin 1.
     You are holding nothing.
     Available actions: ..."
or (after a step):
    "-= Kitchen =-
     You are in the kitchen.
     You see a cup 1 and a dirty mug 1.
     You are carrying: nothing."
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── Location type keywords ─────────────────────────────────────────────────
_LOCATION_TYPE_MAP = {
    "sinkbasin": "sinkbasin",
    "sink":      "sinkbasin",
    "microwave": "microwave",
    "fridge":    "fridge",
    "refrigerator": "fridge",
    "countertop": "countertop",
    "cabinet":   "cabinet",
    "shelf":     "shelf",
    "drawer":    "drawer",
    "sofa":      "sofa",
    "armchair":  "armchair",
    "bed":       "bed",
    "desk":      "desk",
    "table":     "table",
    "coffeetable": "table",
    "sidetable": "table",
    "garbage":   "garbagecan",
    "toilet":    "toilet",
    "bathtub":   "bathtub",
    "handtowelholder": "handtowelholder",
    "towelholder": "towelholder",
    "safe":      "safe",
    "dresser":   "dresser",
}

# Object state adjectives that appear in ALFWorld obs text
_STATE_ADJECTIVES = {
    "clean":  "clean",
    "dirty":  None,  # opposite of clean
    "hot":    "hot",
    "warm":   "hot",
    "cold":   "cold",
    "cool":   "cold",
    "open":   "open",
    "closed": None,
    "cooked": "hot",  # microwave-heated
}


def _parse_location(obs: str) -> str:
    """Extract room/location name from obs."""
    # Pattern: "-= Kitchen =-" or "You are in the kitchen."
    m = re.search(r"-=\s*(.+?)\s*=-", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"you are (?:in|at|on|near) (?:the\s+)?(.+?)[\.\,]", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return ""


def _parse_location_type(location: str) -> str:
    """Map location string to coarse location type."""
    loc_l = location.lower()
    for kw, lt in _LOCATION_TYPE_MAP.items():
        if kw in loc_l:
            return lt
    return location


def _parse_inventory(obs: str) -> List[str]:
    """Extract inventory from 'You are carrying: ...' or 'holding nothing'."""
    # "You are carrying: cup 1, plate 1."
    m = re.search(r"(?:carrying|holding):\s*(.+?)[\.\n]", obs, re.IGNORECASE)
    if m:
        items_str = m.group(1).strip().lower()
        if "nothing" in items_str or "empty" in items_str:
            return []
        items = [x.strip() for x in re.split(r",\s*|\s+and\s+", items_str) if x.strip()]
        return items

    # "You are holding nothing." / "You hold nothing."
    if re.search(r"(?:hold|carrying) nothing", obs, re.IGNORECASE):
        return []

    # "You pick up the cup 1" / "You are holding cup 1"
    m = re.search(r"holding (?:the\s+)?(.+?)[\.\,\n]", obs, re.IGNORECASE)
    if m:
        item = m.group(1).strip().lower()
        if "nothing" not in item:
            return [item]
    return []


def _parse_visible_objects(obs: str) -> List[str]:
    """Extract visible objects from 'You see ...' sentences."""
    visible = []
    # "You see a cup 1, a plate 1 and a sinkbasin 1."
    for m in re.finditer(r"you see (.+?)[\.\n]", obs, re.IGNORECASE):
        raw = m.group(1)
        raw = re.sub(r"\b(a|an|the)\b", "", raw, flags=re.IGNORECASE)
        objects = re.split(r",\s*|\s+and\s+", raw)
        for obj in objects:
            obj = obj.strip().lower()
            if obj:
                visible.append(obj)
    return visible


def _parse_object_states(obs: str, visible_objects: List[str]) -> Dict[str, Dict[str, bool]]:
    """
    Infer object state properties from adjectives in the observation.
    e.g. "a dirty cup 1" → cup 1: {clean: False}
         "a hot potato 1" → potato 1: {hot: True}
    """
    states: Dict[str, Dict[str, bool]] = {}

    # Pattern: optional adjective(s) before object name number
    pattern = re.compile(
        r"\b(clean|dirty|hot|warm|cold|cool|open|closed|cooked)(?:\s+\w+)?\s+(\w[\w ]*?\d+)\b",
        re.IGNORECASE,
    )
    for m in pattern.finditer(obs):
        adj = m.group(1).lower()
        obj = m.group(2).strip().lower()
        prop = _STATE_ADJECTIVES.get(adj)
        if obj not in states:
            states[obj] = {}
        if prop is not None:
            states[obj][prop] = True
        else:
            # "dirty" → clean: False
            if adj == "dirty":
                states[obj]["clean"] = False
            elif adj == "closed":
                states[obj]["open"] = False

    return states


def _parse_admissible_type(admissible_commands: Optional[List[str]]) -> str:
    """Infer location type from admissible commands (fallback)."""
    if not admissible_commands:
        return ""
    for cmd in admissible_commands:
        cmd_l = cmd.lower()
        for kw, lt in _LOCATION_TYPE_MAP.items():
            if f"in {kw}" in cmd_l or f"with {kw}" in cmd_l:
                return lt
    return ""


def parse_alfworld_state(
    obs: str,
    admissible_commands: Optional[List[str]] = None,
    last_feedback: str = "",
) -> Dict[str, Any]:
    """
    Parse a raw ALFWorld observation string into a structured state dict.

    Returns
    -------
    dict with keys:
        location        : str  – e.g. "kitchen"
        location_type   : str  – coarse type e.g. "sinkbasin"
        inventory       : list[str]
        visible_objects : list[str]
        object_states   : dict[str, dict[str, bool]]
        object_location : dict[str, str]  – best-effort (empty if unknown)
        valid_actions   : list[str] | None
        last_feedback   : str
    """
    location = _parse_location(obs)
    inventory = _parse_inventory(obs)
    visible_objects = _parse_visible_objects(obs)
    obj_states = _parse_object_states(obs, visible_objects)

    # Infer location_type from location name first, then admissible commands
    location_type = _parse_location_type(location)
    if not location_type and admissible_commands:
        location_type = _parse_admissible_type(admissible_commands)

    # Best-effort object_location: items in inventory → "inventory"
    obj_location: Dict[str, str] = {obj: "inventory" for obj in inventory}

    return {
        "location":        location,
        "location_type":   location_type,
        "inventory":       inventory,
        "visible_objects": visible_objects,
        "object_states":   obj_states,
        "object_location": obj_location,
        "valid_actions":   admissible_commands,
        "last_feedback":   last_feedback,
    }


def _extract_task_type_from_obs(obs: str) -> str:
    """Heuristic: extract task type from initial observation."""
    obs_l = obs.lower()
    if "clean" in obs_l and "put" in obs_l:
        return "clean"
    if "heat" in obs_l and "put" in obs_l:
        return "heat"
    if "cool" in obs_l and "put" in obs_l:
        return "cool"
    if "examine" in obs_l or "look at" in obs_l:
        return "look"
    if "put two" in obs_l or "pick two" in obs_l or "pick up two" in obs_l:
        return "pick2"
    if "put" in obs_l or "place" in obs_l or "move" in obs_l:
        return "pick"
    return "pick"


def _extract_target_objects(obs: str) -> tuple:
    """Best-effort extraction of target object and receptacle from task description."""
    # "clean some cup and put it in cabinet" → object=cup, receptacle=cabinet
    m = re.search(r"clean (?:some |a |an |the )?(\w[\w ]*?) and put it in (?:the )?(\w[\w ]*)", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"heat (?:some |a |an |the )?(\w[\w ]*?) and put it in (?:the )?(\w[\w ]*)", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"cool (?:some |a |an |the )?(\w[\w ]*?) and put it in (?:the )?(\w[\w ]*)", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"put (?:some |a |an |the )?(\w[\w ]*?) in (?:the )?(\w[\w ]*)", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(?:pick up|get|find) (?:some |a |an |the )?(\w[\w ]*?)[\.\,\n]", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip(), ""
    return "", ""


def build_task_meta(task_description: str) -> Dict[str, Any]:
    """Build task_meta dict from the task description string."""
    task_type = _extract_task_type_from_obs(task_description)
    obj, rec = _extract_target_objects(task_description)
    return {
        "target_obj":        obj.lower(),
        "target_obj2":       "",
        "target_receptacle": rec.lower(),
        "task_type":         task_type,
        "requires_clean":    task_type == "clean",
        "requires_heat":     task_type == "heat",
        "requires_cool":     task_type == "cool",
    }
