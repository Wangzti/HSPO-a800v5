# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Unit tests for ALFWorld Rule-PRM."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hspo.prm.alfworld_prm import AlfworldRulePRM
from hspo.prm.base import PRMOutput


@pytest.fixture
def prm():
    return AlfworldRulePRM()


def _state(inventory=(), visible=(), location_type="", obj_states=None, obj_location=None):
    return {
        "inventory": list(inventory),
        "visible_objects": list(visible),
        "location_type": location_type,
        "object_states": obj_states or {},
        "object_location": obj_location or {},
        "valid_actions": None,
        "last_feedback": "",
    }


TASK_META_CLEAN = {
    "target_obj": "cup 1",
    "target_obj2": "",
    "target_receptacle": "cabinet 1",
    "task_type": "clean",
    "requires_clean": True,
    "requires_heat": False,
    "requires_cool": False,
}


class TestAlfworldRulePRM:

    def test_find_object_progress_increases_when_visible(self, prm):
        state_before = _state()
        state_after = _state(visible=["cup 1"])
        out = prm.score("FIND_OBJECT", "find the cup", state_before, state_after, "go to countertop 1", TASK_META_CLEAN)
        assert out.progress_after > out.progress_before

    def test_find_object_progress_1_when_in_inventory(self, prm):
        state_after = _state(inventory=["cup 1"])
        out = prm.score("FIND_OBJECT", "find the cup", _state(), state_after, "pick up cup 1", TASK_META_CLEAN)
        assert out.progress_after == pytest.approx(1.0)

    def test_pick_object_done_when_in_inventory(self, prm):
        state_after = _state(inventory=["cup 1"])
        out = prm.score("PICK_OBJECT", "pick up cup", _state(visible=["cup 1"]), state_after, "pick up cup 1", TASK_META_CLEAN)
        assert out.done_after == 1.0
        assert out.progress_after == 1.0

    def test_clean_object_done_when_clean(self, prm):
        state_after = _state(
            inventory=["cup 1"],
            obj_states={"cup 1": {"clean": True}},
        )
        out = prm.score("CLEAN_OBJECT", "clean cup", _state(inventory=["cup 1"]), state_after, "clean cup 1 with sinkbasin 1", TASK_META_CLEAN)
        assert out.done_after == 1.0
        assert out.progress_after == 1.0

    def test_clean_object_progress_at_sink(self, prm):
        state_after = _state(inventory=["cup 1"], location_type="sinkbasin")
        out = prm.score("CLEAN_OBJECT", "clean cup", _state(inventory=["cup 1"]), state_after, "go to sinkbasin 1", TASK_META_CLEAN)
        assert out.progress_after == pytest.approx(0.8)

    def test_place_object_done(self, prm):
        state_after = _state(obj_location={"cup 1": "cabinet 1"})
        out = prm.score("PLACE_OBJECT", "put cup in cabinet", _state(inventory=["cup 1"]), state_after, "put cup 1 in cabinet 1", TASK_META_CLEAN)
        assert out.done_after == 1.0

    def test_side_effect_wrong_container(self, prm):
        # Cup placed in fridge during clean task → side effect
        state_after = _state(obj_location={"cup 1": "fridge 1"})
        out = prm.score("CLEAN_OBJECT", "clean cup", _state(), state_after, "put cup 1 in fridge 1", TASK_META_CLEAN)
        assert out.side_effect_after > 0.0

    def test_validity_from_feedback(self, prm):
        state_after = _state()
        state_after["last_feedback"] = "nothing happens"
        out = prm.score("FIND_OBJECT", "find cup", _state(), state_after, "invalid command xyz", TASK_META_CLEAN)
        assert out.valid == 0.0

    def test_low_level_reward_positive_on_progress(self, prm):
        state_before = _state()
        state_after = _state(visible=["cup 1"])
        out = prm.score("FIND_OBJECT", "find the cup", state_before, state_after, "go to countertop 1", TASK_META_CLEAN)
        r = out.low_level_reward()
        assert r > 0.0

    def test_low_level_reward_done_bonus(self, prm):
        state_before = _state(inventory=["cup 1"], location_type="sinkbasin")
        state_after = _state(inventory=["cup 1"], obj_states={"cup 1": {"clean": True}})
        out = prm.score("CLEAN_OBJECT", "clean cup", state_before, state_after, "clean cup 1 with sinkbasin 1", TASK_META_CLEAN)
        r = out.low_level_reward(eta_done=1.0, tau_done=0.9)
        assert r > 1.0  # progress delta + done bonus

    def test_heat_object(self, prm):
        task_meta = {**TASK_META_CLEAN, "requires_clean": False, "requires_heat": True, "task_type": "heat"}
        state_after = _state(inventory=["cup 1"], obj_states={"cup 1": {"hot": True}})
        out = prm.score("HEAT_OBJECT", "heat cup", _state(inventory=["cup 1"]), state_after, "heat cup 1 in microwave 1", task_meta)
        assert out.done_after == 1.0

    def test_cool_object(self, prm):
        task_meta = {**TASK_META_CLEAN, "requires_clean": False, "requires_cool": True, "task_type": "cool"}
        state_after = _state(inventory=["cup 1"], obj_states={"cup 1": {"cold": True}})
        out = prm.score("COOL_OBJECT", "cool cup", _state(inventory=["cup 1"]), state_after, "cool cup 1 in fridge 1", task_meta)
        assert out.done_after == 1.0
