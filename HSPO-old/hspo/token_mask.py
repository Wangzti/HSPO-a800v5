# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
TokenMaskBuilder

Maps character-level span offsets (from PlanExecuteParser) to token-level
boolean masks over the response token sequence.

HSPO uses three mutually exclusive masks:
  • switch_mask   – covers the <switch>…</switch> block
  • subgoal_mask  – covers the <subgoal>…</subgoal> block
  • action_mask   – covers the <action>…</action> block

During training:
  • A_H  (macro advantage)  × subgoal_mask  → planner PPO loss
  • A_L  (process return)   × action_mask   → executor PPO loss
  • L_switch (CE)           × switch_mask   → supervised switch loss

Non-overlapping guarantee: tags are non-nested in the output format, so the
three masks are naturally disjoint. An assertion verifies this.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class TokenMaskBuilder:
    """
    Build per-token span masks from character offsets and a tokenizer.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer (must support ``encode`` with
        ``return_offsets_mapping=True`` or the character-map approach).
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(
        self,
        response_text: str,
        switch_char_span: Optional[Tuple[int, int]],
        subgoal_char_span: Optional[Tuple[int, int]],
        action_char_span: Optional[Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenise *response_text* and return three boolean mask tensors.

        Each tensor has shape ``(response_len,)`` where ``response_len`` is
        the number of tokens in the response.  A token at position ``i`` is
        ``True`` iff its character span overlaps with the requested tag block.

        Returns
        -------
        dict with keys "switch_mask", "subgoal_mask", "action_mask",
        plus "token_ids" (LongTensor) and "response_len" (int).
        """
        encoding = self.tokenizer(
            response_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        token_ids: List[int] = encoding["input_ids"]
        offsets: List[Tuple[int, int]] = encoding["offset_mapping"]
        n = len(token_ids)

        switch_mask  = self._span_to_mask(offsets, switch_char_span,  n)
        subgoal_mask = self._span_to_mask(offsets, subgoal_char_span, n)
        action_mask  = self._span_to_mask(offsets, action_char_span,  n)

        # Verify non-overlap (debug guard; negligible overhead)
        overlap_sw_sg = (switch_mask & subgoal_mask).any().item()
        overlap_sw_ac = (switch_mask & action_mask).any().item()
        overlap_sg_ac = (subgoal_mask & action_mask).any().item()
        if overlap_sw_sg or overlap_sw_ac or overlap_sg_ac:
            raise RuntimeError(
                "TokenMaskBuilder: span masks overlap — check parser output.\n"
                f"  switch∩subgoal={overlap_sw_sg}, "
                f"switch∩action={overlap_sw_ac}, "
                f"subgoal∩action={overlap_sg_ac}\n"
                f"  text={response_text!r}"
            )

        return {
            "switch_mask":  switch_mask,
            "subgoal_mask": subgoal_mask,
            "action_mask":  action_mask,
            "token_ids":    torch.tensor(token_ids, dtype=torch.long),
            "response_len": n,
        }

    def build_from_parse_result(self, response_text: str, parse_result) -> Dict[str, torch.Tensor]:
        """Convenience wrapper that accepts a ``ParseResult`` object."""
        return self.build(
            response_text=response_text,
            switch_char_span=parse_result.switch_char_span,
            subgoal_char_span=parse_result.subgoal_char_span,
            action_char_span=parse_result.action_char_span,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _span_to_mask(
        offsets: List[Tuple[int, int]],
        char_span: Optional[Tuple[int, int]],
        n: int,
    ) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool)
        if char_span is None:
            return mask
        span_start, span_end = char_span
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                break
            mask[i] = True
        return mask

    # ------------------------------------------------------------------ #
    # Batch helpers (used during training)                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def pad_masks(
        masks: List[torch.Tensor],
        pad_to: int,
    ) -> torch.Tensor:
        """Stack variable-length masks into a (B, pad_to) bool tensor (right-pad with False)."""
        B = len(masks)
        out = torch.zeros(B, pad_to, dtype=torch.bool)
        for i, m in enumerate(masks):
            L = min(len(m), pad_to)
            out[i, :L] = m[:L]
        return out
