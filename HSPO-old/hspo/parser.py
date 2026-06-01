# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
PlanExecuteParser

Extracts <switch>, <subgoal>, <action> spans from raw LLM output strings.
Returns both the parsed text content and the character-level offsets of the
content inside each tag block (used downstream by TokenMaskBuilder to compute
token spans).

Error taxonomy stored in parse_result["errors"]:
  "missing_switch"         – no <switch>...</switch> block found
  "missing_subgoal"        – no <subgoal>...</subgoal> block found
  "missing_action"         – no <action>...</action> block found
  "invalid_switch_value"   – switch text is not "SWITCH" or "KEEP"
  "multiple_action_blocks" – more than one <action> block
  "empty_action"           – action content is whitespace-only
  "empty_subgoal"          – subgoal content is whitespace-only
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Compiled patterns – each returns (content, char_start, char_end) of the full
# tag block (including the opening and closing tags) via a named group.
_SWITCH_RE  = re.compile(r"<switch>(.*?)</switch>",   re.IGNORECASE | re.DOTALL)
_SUBGOAL_RE = re.compile(r"<subgoal>(.*?)</subgoal>", re.IGNORECASE | re.DOTALL)
_ACTION_RE  = re.compile(r"<action>(.*?)</action>",   re.IGNORECASE | re.DOTALL)


def _extract_span(
    pattern: re.Pattern,
    text: str,
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Return (content_stripped, (content_start, content_end)) or (None, None)."""
    m = pattern.search(text)
    if m is None:
        return None, None
    raw = m.group(1)
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    content = raw.strip()
    return content, (m.start(1) + leading, m.start(1) + trailing)


class ParseResult:
    """Structured result of a single parse call."""

    __slots__ = (
        "switch", "subgoal", "action",
        "switch_char_span", "subgoal_char_span", "action_char_span",
        "valid_format", "errors",
    )

    def __init__(self) -> None:
        self.switch: Optional[str] = None
        self.subgoal: Optional[str] = None
        self.action: Optional[str] = None
        self.switch_char_span: Optional[Tuple[int, int]] = None
        self.subgoal_char_span: Optional[Tuple[int, int]] = None
        self.action_char_span: Optional[Tuple[int, int]] = None
        self.valid_format: bool = False
        self.errors: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "switch": self.switch,
            "subgoal": self.subgoal,
            "action": self.action,
            "switch_char_span": list(self.switch_char_span) if self.switch_char_span else None,
            "subgoal_char_span": list(self.subgoal_char_span) if self.subgoal_char_span else None,
            "action_char_span": list(self.action_char_span) if self.action_char_span else None,
            "valid_format": self.valid_format,
            "errors": self.errors,
        }


class PlanExecuteParser:
    """
    Parse HSPO model outputs with the Plan-Execute structured format.

    Expected output format (case-insensitive tags):
        <switch>SWITCH or KEEP</switch>
        <subgoal>natural language subgoal</subgoal>
        <action>environment action string</action>

    Usage::

        parser = PlanExecuteParser()
        result = parser.parse("<switch>KEEP</switch><subgoal>clean cup</subgoal><action>go to sinkbasin 1</action>")
        assert result.valid_format
        assert result.switch == "KEEP"
    """

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()

        # ── switch ────────────────────────────────────────────────────── #
        switch_val, switch_span = _extract_span(_SWITCH_RE, text)
        if switch_val is None:
            result.errors.append("missing_switch")
        else:
            result.switch = switch_val.upper()
            result.switch_char_span = switch_span
            if result.switch not in ("SWITCH", "KEEP"):
                result.errors.append("invalid_switch_value")
                result.switch = None  # treat as missing

        # ── subgoal ───────────────────────────────────────────────────── #
        subgoal_val, subgoal_span = _extract_span(_SUBGOAL_RE, text)
        if subgoal_val is None:
            result.errors.append("missing_subgoal")
        elif not subgoal_val:
            result.errors.append("empty_subgoal")
        else:
            result.subgoal = subgoal_val
            result.subgoal_char_span = subgoal_span

        # ── action (allow multiple, flag if > 1) ──────────────────────── #
        all_action_matches = list(_ACTION_RE.finditer(text))
        if not all_action_matches:
            result.errors.append("missing_action")
        else:
            if len(all_action_matches) > 1:
                result.errors.append("multiple_action_blocks")
            m = all_action_matches[0]
            raw_action = m.group(1)
            leading = len(raw_action) - len(raw_action.lstrip())
            trailing = len(raw_action.rstrip())
            action_val = raw_action.strip().lower()
            if not action_val:
                result.errors.append("empty_action")
            else:
                result.action = action_val
                result.action_char_span = (m.start(1) + leading, m.start(1) + trailing)

        # ── validity ──────────────────────────────────────────────────── #
        result.valid_format = (
            result.switch is not None
            and result.subgoal is not None
            and result.action is not None
            and not result.errors
        )
        return result

    def parse_batch(self, texts: List[str]) -> List[ParseResult]:
        return [self.parse(t) for t in texts]
