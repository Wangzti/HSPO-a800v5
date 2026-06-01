# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Unit tests for PlanExecuteParser."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hspo.parser import PlanExecuteParser


@pytest.fixture
def parser():
    return PlanExecuteParser()


class TestPlanExecuteParser:

    def test_valid_full_output(self, parser):
        text = "<switch>SWITCH</switch><subgoal>clean the cup</subgoal><action>go to sinkbasin 1</action>"
        r = parser.parse(text)
        assert r.valid_format
        assert r.switch == "SWITCH"
        assert r.subgoal == "clean the cup"
        assert r.action == "go to sinkbasin 1"
        assert r.errors == []

    def test_keep_switch(self, parser):
        text = "<switch>KEEP</switch><subgoal>clean the cup</subgoal><action>clean cup 1 with sinkbasin 1</action>"
        r = parser.parse(text)
        assert r.valid_format
        assert r.switch == "KEEP"

    def test_case_insensitive_tags(self, parser):
        text = "<Switch>switch</Switch><SubGoal>find cup</SubGoal><ACTION>go to countertop 1</ACTION>"
        r = parser.parse(text)
        assert r.valid_format
        assert r.switch == "SWITCH"

    def test_missing_switch(self, parser):
        text = "<subgoal>clean the cup</subgoal><action>go to sinkbasin 1</action>"
        r = parser.parse(text)
        assert not r.valid_format
        assert "missing_switch" in r.errors

    def test_missing_subgoal(self, parser):
        text = "<switch>KEEP</switch><action>go to sinkbasin 1</action>"
        r = parser.parse(text)
        assert not r.valid_format
        assert "missing_subgoal" in r.errors

    def test_missing_action(self, parser):
        text = "<switch>KEEP</switch><subgoal>clean the cup</subgoal>"
        r = parser.parse(text)
        assert not r.valid_format
        assert "missing_action" in r.errors

    def test_invalid_switch_value(self, parser):
        text = "<switch>MAYBE</switch><subgoal>do something</subgoal><action>go north</action>"
        r = parser.parse(text)
        assert not r.valid_format
        assert "invalid_switch_value" in r.errors

    def test_empty_action(self, parser):
        text = "<switch>KEEP</switch><subgoal>clean cup</subgoal><action>   </action>"
        r = parser.parse(text)
        assert not r.valid_format
        assert "empty_action" in r.errors

    def test_char_spans_set(self, parser):
        text = "<switch>SWITCH</switch><subgoal>clean cup</subgoal><action>go to sink 1</action>"
        r = parser.parse(text)
        assert r.switch_char_span is not None
        assert r.subgoal_char_span is not None
        assert r.action_char_span is not None
        # Non-overlapping
        sw_start, sw_end = r.switch_char_span
        sg_start, sg_end = r.subgoal_char_span
        ac_start, ac_end = r.action_char_span
        assert sw_end <= sg_start
        assert sg_end <= ac_start

    def test_to_dict(self, parser):
        text = "<switch>KEEP</switch><subgoal>find cup</subgoal><action>look</action>"
        r = parser.parse(text)
        d = r.to_dict()
        assert "switch" in d
        assert "valid_format" in d
        assert isinstance(d["errors"], list)

    def test_batch(self, parser):
        texts = [
            "<switch>SWITCH</switch><subgoal>find cup</subgoal><action>go to table 1</action>",
            "malformed output",
        ]
        results = parser.parse_batch(texts)
        assert len(results) == 2
        assert results[0].valid_format
        assert not results[1].valid_format

    def test_multiline_output(self, parser):
        text = """
<switch>KEEP</switch>
<subgoal>pick up the apple</subgoal>
<action>pick up apple 1</action>
"""
        r = parser.parse(text)
        assert r.valid_format
        assert r.subgoal == "pick up the apple"
        assert r.action == "pick up apple 1"
