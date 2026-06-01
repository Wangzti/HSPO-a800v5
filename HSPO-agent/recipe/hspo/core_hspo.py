"""
HSPO: Hierarchical Subgoal-conditioned Process Optimization.

Core algorithm implementing:
1. Plan-Execute token masking (<switch>, <subgoal>, <action> spans)
2. Anchored Branch Grouping (ABG) for critic-free low-level advantages
3. Segment-level macro PPO for high-level subgoal selection
4. Token-level credit routing

Reference: HSPO paper
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from verl import DataProto


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def to_hashable(x):
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    if isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    return str(x)


def _group_indices_by_traj(traj_uid: np.ndarray) -> Dict[object, List[int]]:
    groups: Dict[object, List[int]] = defaultdict(list)
    for i, tid in enumerate(traj_uid):
        groups[tid].append(i)
    return groups


def _as_torch_1d(x, *, device, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).view(-1)
    arr = np.asarray(x)
    if arr.dtype == object:
        if dtype == torch.bool:
            arr = arr.astype(np.bool_)
        else:
            arr = arr.astype(np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype).view(-1)


# ---------------------------------------------------------------------------
#  Token-level text parsing (HiPER-style)
# ---------------------------------------------------------------------------

def _build_text_and_offsets(tokenizer, seq_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces = tokenizer.batch_decode(
        [[int(i)] for i in seq_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for p in pieces:
        start = pos
        pos += len(p)
        offsets.append((start, pos))
    return "".join(pieces), offsets


def _find_tag_char_span(
    text: str, open_tag: str, close_tags: List[str],
) -> Optional[Tuple[int, int, int, int]]:
    s_open = text.find(open_tag)
    if s_open < 0:
        return None
    s_content = s_open + len(open_tag)
    s_close = None
    close_len = 0
    for tag in close_tags:
        idx = text.find(tag, s_content)
        if idx >= 0 and (s_close is None or idx < s_close):
            s_close = idx
            close_len = len(tag)
    if s_close is None:
        return None
    return s_open, s_content, s_close, close_len


def _char_to_token_start(offsets: List[Tuple[int, int]], s_char: int) -> Optional[int]:
    for i, (_, e) in enumerate(offsets):
        if e > s_char:
            return i
    return None


def _char_to_token_end(offsets: List[Tuple[int, int]], e_char: int) -> int:
    for i, (s, _) in enumerate(offsets):
        if s >= e_char:
            return i
    return len(offsets)


def _span_from_tags_text(
    text: str,
    offsets: List[Tuple[int, int]],
    open_tag: str,
    close_tags: List[str],
    *,
    include_tags: bool = False,
) -> Optional[Tuple[int, int]]:
    sp = _find_tag_char_span(text, open_tag, close_tags)
    if sp is None:
        return None
    s_open, s_content, s_close, close_len = sp
    if include_tags:
        s_char, e_char = s_open, s_close + close_len
    else:
        s_char, e_char = s_content, s_close
    if s_char >= e_char:
        return None
    t_start = _char_to_token_start(offsets, s_char)
    if t_start is None:
        return None
    t_end = _char_to_token_end(offsets, e_char)
    if t_start >= t_end:
        return None
    return (t_start, t_end)


# Tag variant lists (taken from HiPER)
CLOSE_SWITCH_TAGS = ["</switch>", "</switch>\n"]
CLOSE_SUBGOAL_TAGS = [
    "</subgoal>", "</subgoal>\n", "]</subgoal>", "]</subgoal>\n",
    ".</subgoal>", ".</subgoal>\n", ")</subgoal>", ")</subgoal>\n",
]
CLOSE_ACTION_TAGS = [
    "</action>", "</action>\n", "]</action>", "]</action>\n",
    ".</action>", ".</action>\n", ")</action>", ")</action>\n",
]


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class HSPOConfig:
    """HSPO algorithm configuration."""

    # ---- ABG (low-level) ----
    abg_min_group_size: int = 2
    abg_target_size: int = 4
    abg_max_branch: int = 4
    abg_step_cost: float = -0.1
    abg_completion_bonus: float = 2.0
    remove_std: bool = True       # mean-only normalization in ABG

    # ---- Macro-PPO (high-level) ----
    high_gamma: float = 0.99
    high_lam: float = 0.95
    high_clip_epsilon: float = 0.2
    high_beta: float = 0.01       # KL coefficient (high)

    # ---- Low-level PPO ----
    low_clip_epsilon: float = 0.2
    low_beta: float = 0.01        # KL coefficient (low)

    # ---- Switch / termination ----
    switch_alpha: float = 0.5
    alpha_t: float = 0.5          # termination loss weight
    keep_penalty: float = 0.0     # penalty per KEEP turn

    # ---- Loss weights (token-level routing) ----
    alpha_l: float = 1.0          # low-level (action) loss weight
    alpha_h: float = 1.0          # high-level (subgoal) loss weight
    alpha_sft: float = 0.1        # SFT regularization weight
    sg_kl_beta: float = 0.01      # subgoal KL regularization
    alpha_v: float = 0.5          # value loss weight (if critic used)

    # ---- Normalization ----
    norm_adv: bool = True

    # ---- Misc ----
    assign_high_to_subgoal: bool = True
    assign_high_to_switch: bool = False
    include_tags_mask: bool = False
    fallback_action_to_full_response: bool = True


# ---------------------------------------------------------------------------
#  Token mask construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_token_masks(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    *,
    include_tags: bool = False,
    fallback_action: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Build per-token masks for <switch>, <subgoal>, <action> spans.

    Uses HiPER-style _build_text_and_offsets for robust character↔token mapping.

    Returns:
        action_mask   : (N,L) bool — tokens inside <action>...</action>
        subgoal_mask  : (N,L) bool — tokens inside <subgoal>...</subgoal>
        switch_mask   : (N,L) bool — tokens inside <switch>...</switch> (content only)
        is_new_subgoal: (N,) bool  — True when <switch> contains SWITCH (not KEEP)
    """
    N, L = responses.shape
    device = responses.device
    response_mask_bool = response_mask.to(torch.bool)

    action_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    subgoal_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    switch_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    is_new_subgoal = np.zeros((N,), dtype=np.bool_)

    keep_set = {"KEEP"}

    for i in range(N):
        valid_len = int(response_mask[i].sum().item())
        if valid_len <= 0:
            continue
        seq_ids = responses[i, :valid_len].tolist()
        text, offsets = _build_text_and_offsets(tokenizer, seq_ids)

        # ---- ACTION span ----
        sp = _span_from_tags_text(
            text, offsets, "<action>", CLOSE_ACTION_TAGS,
            include_tags=include_tags,
        )
        if sp is None:
            if fallback_action:
                action_mask[i, :valid_len] = True
        else:
            s, e = sp
            if s < e:
                action_mask[i, s:e] = True

        # ---- SUBGOAL span ----
        sp = _span_from_tags_text(
            text, offsets, "<subgoal>", CLOSE_SUBGOAL_TAGS,
            include_tags=include_tags,
        )
        if sp is not None:
            s, e = sp
            if s < e:
                subgoal_mask[i, s:e] = True

        # ---- SWITCH span ----
        # (1) Extract content for boundary decision
        sp_content = _find_tag_char_span(text, "<switch>", CLOSE_SWITCH_TAGS)
        switch_text = "KEEP"
        if sp_content is not None:
            _, s_c, s_e, _ = sp_content
            if s_c < s_e:
                txt = text[s_c:s_e].strip().upper()
                if txt:
                    switch_text = txt

        # (2) Mask span (content only, not tags)
        sp_mask = _span_from_tags_text(
            text, offsets, "<switch>", CLOSE_SWITCH_TAGS,
            include_tags=include_tags,
        )
        if sp_mask is not None:
            s_m, e_m = sp_mask
            if s_m < e_m:
                switch_mask[i, s_m:e_m] = True

        is_new_subgoal[i] = (switch_text not in keep_set)

    # Restrict to valid response tokens
    action_mask &= response_mask_bool
    subgoal_mask &= response_mask_bool
    switch_mask &= response_mask_bool

    return action_mask, subgoal_mask, switch_mask, is_new_subgoal


# ---------------------------------------------------------------------------
#  Decode subgoal text from responses (per sample)
# ---------------------------------------------------------------------------

def decode_subgoal_texts(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
) -> Tuple[List[str], np.ndarray]:
    """Decode the subgoal text from each sample's response.

    Returns:
        subgoal_texts: List[str] — decoded subgoal content
        is_new_subgoal: (N,) bool
    """
    N = responses.shape[0]
    subgoal_texts = ["" for _ in range(N)]
    is_new_subgoal = np.zeros((N,), dtype=np.bool_)

    keep_set = {"KEEP"}

    for i in range(N):
        valid_len = int(response_mask[i].sum().item())
        if valid_len <= 0:
            continue
        seq_ids = responses[i, :valid_len].tolist()
        text, offsets = _build_text_and_offsets(tokenizer, seq_ids)

        # Extract subgoal content
        sp = _find_tag_char_span(text, "<subgoal>", CLOSE_SUBGOAL_TAGS)
        if sp is not None:
            _, s_c, s_e, _ = sp
            if s_c < s_e:
                subgoal_texts[i] = text[s_c:s_e].strip()

        # Extract switch decision
        sp_sw = _find_tag_char_span(text, "<switch>", CLOSE_SWITCH_TAGS)
        switch_text = "KEEP"
        if sp_sw is not None:
            _, s_c, s_e, _ = sp_sw
            if s_c < s_e:
                txt = text[s_c:s_e].strip().upper()
                if txt:
                    switch_text = txt
        is_new_subgoal[i] = (switch_text not in keep_set)

    return subgoal_texts, is_new_subgoal


# ---------------------------------------------------------------------------
#  ABG: Anchored Branch Grouping
# ---------------------------------------------------------------------------

def build_abg_groups(
    anchor_obs: np.ndarray,
    subgoal_texts: np.ndarray,
    episode_index: np.ndarray,
    milestone_scores: np.ndarray,
    cfg: HSPOConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Group transitions by (episode_index, anchor_obs, subgoal_text) anchor.

    Each group contains transitions that share the same task, state, and subgoal.

    Returns:
        group_indices: List of index arrays (into the batch)
        group_scores:  List of milestone-score arrays (aligned with group_indices)
    """
    N = len(anchor_obs)
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)

    for i in range(N):
        key = (
            int(episode_index[i]),
            to_hashable(anchor_obs[i]),
            to_hashable(subgoal_texts[i]),
        )
        key_to_indices[key].append(i)

    group_indices_list: List[np.ndarray] = []
    group_scores_list: List[np.ndarray] = []

    for key, indices in key_to_indices.items():
        idx_arr = np.array(indices, dtype=np.int64)
        scores = milestone_scores[idx_arr]

        if len(idx_arr) >= cfg.abg_min_group_size:
            group_indices_list.append(idx_arr)
            group_scores_list.append(scores)
        # Single-element groups are excluded (no relative comparison possible)

    return group_indices_list, group_scores_list


def compute_abg_low_level_advantage(
    batch_size: int,
    response_length: int,
    response_mask: torch.Tensor,
    group_indices_list: List[np.ndarray],
    group_scores_list: List[np.ndarray],
    cfg: HSPOConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute critic-free low-level advantages via ABG group-relative normalization.

    For each ABG group, normalize milestone scores within the group.
    Advantages are assigned per-step (broadcast to all tokens of that step).

    Returns:
        advantages : (bs, response_length) — per-token low-level advantages
        per_step   : (bs,) — per-step scalar advantage
    """
    bs = batch_size
    advantages = torch.zeros((bs, response_length), device=device, dtype=torch.float32)
    per_step_adv = torch.zeros(bs, device=device, dtype=torch.float32)

    for indices, scores in zip(group_indices_list, group_scores_list):
        scores_t = torch.as_tensor(scores, device=device, dtype=torch.float32)
        mean = scores_t.mean()

        if cfg.remove_std:
            adv = scores_t - mean
        else:
            std = scores_t.std(unbiased=False) + 1e-6
            adv = (scores_t - mean) / std

        for j, idx in enumerate(indices):
            per_step_adv[idx] = adv[j]

    # Broadcast per-step to all response tokens
    advantages = per_step_adv.unsqueeze(-1).expand(-1, response_length) * response_mask.to(torch.float32)

    return advantages, per_step_adv


# ---------------------------------------------------------------------------
#  High-level macro-PPO advantage (segment-level GAE)
# ---------------------------------------------------------------------------

def compute_high_level_macro_advantage(
    batch: DataProto,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute high-level macro advantages over subgoal segments.

    Uses SMDP-style GAE: segments are defined by SWITCH boundaries.
    Within each segment, returns are discounted; across segments,
    GAE is computed with gamma^d discounting (d = segment length).

    Returns:
        advantages : (bs, response_length) — per-token high-level advantages
        returns    : (bs, response_length) — per-token high-level returns
    """
    bs, L = response_mask.shape
    traj_uid = batch.non_tensor_batch["traj_uid"]
    turn_idx = np.asarray(batch.non_tensor_batch.get("turn_idx", np.arange(bs, dtype=np.int64)))
    dones = np.asarray(batch.non_tensor_batch.get("dones", np.zeros(bs, dtype=bool))).astype(np.bool_)
    switch = np.asarray(batch.non_tensor_batch.get("switch", np.zeros(bs, dtype=bool))).astype(np.bool_)
    rewards = _as_torch_1d(batch.non_tensor_batch["rewards"], device=device, dtype=torch.float32)

    groups = _group_indices_by_traj(traj_uid)

    # Per-step scalars
    adv_step = torch.zeros(bs, device=device, dtype=torch.float32)
    ret_step = torch.zeros(bs, device=device, dtype=torch.float32)

    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        T = len(idxs)
        if T == 0:
            continue

        # Build boundary positions (segment starts)
        boundary_pos = [0]
        for pos in range(1, T):
            if bool(switch[idxs[pos]]):
                boundary_pos.append(pos)
        boundary_pos = sorted(set(boundary_pos))

        # Build segments: (start_pos, end_pos, next_start_or_None)
        segs: List[Tuple[int, int, Optional[int]]] = []
        for k, s_pos in enumerate(boundary_pos):
            e_pos = (boundary_pos[k + 1] - 1) if (k + 1 < len(boundary_pos)) else (T - 1)
            nxt = boundary_pos[k + 1] if (k + 1 < len(boundary_pos)) else None
            segs.append((s_pos, e_pos, nxt))

        # GAE across segments (reverse iteration)
        next_seg_adv = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)

        for k in range(len(segs) - 1, -1, -1):
            s_pos, e_pos, nxt_pos = segs[k]
            start_i = idxs[s_pos]
            end_i = idxs[e_pos]

            # Discounted segment return
            seg_return = torch.tensor(0.0, device=device)
            disc = 1.0
            for pos in range(s_pos, e_pos + 1):
                seg_return = seg_return + disc * rewards[idxs[pos]]
                disc *= gamma

            d_k = e_pos - s_pos + 1  # segment duration
            done_end = bool(dones[end_i])
            not_done = 0.0 if done_end else 1.0
            gd = gamma ** d_k

            # Bootstrap value (simplified: 0 at episode end)
            if not_done > 0:
                boot_v = zero  # No critic; high-level uses terminal reward only
            else:
                boot_v = zero

            target = seg_return + gd * not_done * boot_v
            delta_k = target  # target - V_high(start_i); V_high = 0 in critic-free case
            seg_adv_k = delta_k + gd * lam * not_done * next_seg_adv

            adv_step[start_i] = seg_adv_k
            ret_step[start_i] = seg_adv_k  # ret = adv + V, V=0 here

            next_seg_adv = seg_adv_k

        # Propagate segment advantage to all turns within the segment (for logging)
        for k, (s_pos, e_pos, _nxt) in enumerate(segs):
            start_i = idxs[s_pos]
            for pos in range(s_pos + 1, e_pos + 1):
                i = idxs[pos]
                adv_step[i] = adv_step[start_i] * (gamma ** (pos - s_pos))

    # Broadcast to token level
    advantages = adv_step.unsqueeze(-1).expand(-1, L) * response_mask.to(torch.float32)
    returns = ret_step.unsqueeze(-1).expand(-1, L) * response_mask.to(torch.float32)

    return advantages, returns


# ---------------------------------------------------------------------------
#  Main HSPO advantage computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_hspo_advantage(
    batch: DataProto,
    cfg: HSPOConfig,
    tokenizer,
    milestone_scorer,
    env_name: str,
) -> DataProto:
    """Compute HSPO advantages and store masks in the batch.

    Steps:
    1. Build token masks (<switch>, <subgoal>, <action>)
    2. Decode subgoal texts and switch decisions
    3. Compute milestone scores for each step
    4. Build ABG groups and compute low-level advantages
    5. Compute high-level macro advantages
    6. Store masks, advantages, and returns in batch
    """
    device = batch.batch["response_mask"].device
    response_mask = batch.batch["response_mask"].to(torch.bool)
    responses = batch.batch["responses"]

    bs, L = responses.shape

    # -----------------------------------------------------------
    # 1. Build token masks
    # -----------------------------------------------------------
    action_mask, subgoal_mask, switch_mask, is_new_subgoal = build_token_masks(
        responses, response_mask, tokenizer,
        include_tags=cfg.include_tags_mask,
        fallback_action=cfg.fallback_action_to_full_response,
    )

    # -----------------------------------------------------------
    # 2. Decode subgoal texts
    # -----------------------------------------------------------
    subgoal_texts, is_new_subgoal2 = decode_subgoal_texts(
        responses, response_mask, tokenizer,
    )
    # Prefer env-tracked switch over decoded when available
    if "switch" in batch.non_tensor_batch:
        is_new_subgoal = np.asarray(batch.non_tensor_batch["switch"]).astype(np.bool_)

    subgoal_texts_np = np.array(subgoal_texts, dtype=object)

    # -----------------------------------------------------------
    # 3. Compute milestone scores for each step
    # -----------------------------------------------------------
    milestone_scores = np.zeros(bs, dtype=np.float32)

    # Collect per-step info from batch
    anchor_obs = batch.non_tensor_batch.get(
        "anchor_obs",
        batch.non_tensor_batch.get("anchor",
            np.array([None for _ in range(bs)], dtype=object),
        ),
    )
    prev_obs_list = batch.non_tensor_batch.get("prev_obs", [None] * bs)
    curr_obs_list = batch.non_tensor_batch.get("curr_obs", [None] * bs)
    actions_list = batch.non_tensor_batch.get("decoded_actions", [""] * bs)
    gamefiles = batch.non_tensor_batch.get("gamefile", None)

    for i in range(bs):
        subgoal = subgoal_texts_np[i] if i < len(subgoal_texts_np) else ""
        obs_before = str(prev_obs_list[i]) if i < len(prev_obs_list) and prev_obs_list[i] is not None else ""
        obs_after = str(curr_obs_list[i]) if i < len(curr_obs_list) and curr_obs_list[i] is not None else ""
        action = str(actions_list[i]) if i < len(actions_list) else ""
        gamefile = gamefiles[i] if gamefiles is not None and i < len(gamefiles) else None

        if subgoal or obs_after:
            try:
                milestone_scores[i] = milestone_scorer.score(
                    subgoal=subgoal,
                    state_before=obs_before,
                    action=action,
                    state_after=obs_after,
                    gamefile=gamefile,
                )
            except Exception:
                milestone_scores[i] = 0.0

    # -----------------------------------------------------------
    # 4. ABG groups + low-level advantages
    # -----------------------------------------------------------
    episode_index = batch.non_tensor_batch.get(
        "uid",
        np.array([str(i) for i in range(bs)], dtype=object),
    )
    _, inverse = np.unique(episode_index, return_inverse=True)

    group_indices_list, group_scores_list = build_abg_groups(
        anchor_obs=anchor_obs,
        subgoal_texts=subgoal_texts_np,
        episode_index=inverse,
        milestone_scores=milestone_scores,
        cfg=cfg,
    )

    low_advantages, low_per_step = compute_abg_low_level_advantage(
        bs, L, response_mask,
        group_indices_list, group_scores_list,
        cfg, device,
    )

    # -----------------------------------------------------------
    # 5. High-level macro advantages
    # -----------------------------------------------------------
    high_advantages, high_returns = compute_high_level_macro_advantage(
        batch, response_mask,
        cfg.high_gamma, cfg.high_lam,
        device,
    )

    # -----------------------------------------------------------
    # 6. Normalize advantages (optional)
    # -----------------------------------------------------------
    if cfg.norm_adv:
        lo_active = action_mask.any(dim=1)
        if lo_active.any():
            mu = low_per_step[lo_active].mean()
            sigma = low_per_step[lo_active].std(unbiased=False) + 1e-8
            low_per_step = (low_per_step - mu) / sigma
            low_advantages = low_per_step.unsqueeze(-1).expand(-1, L) * response_mask.to(torch.float32)

        hi_active = subgoal_mask.any(dim=1)
        if hi_active.any():
            hi_per_step = high_advantages.sum(dim=1)
            mu = hi_per_step[hi_active].mean()
            sigma = hi_per_step[hi_active].std(unbiased=False) + 1e-8
            hi_per_step = (hi_per_step - mu) / sigma
            high_advantages = hi_per_step.unsqueeze(-1).expand(-1, L) * response_mask.to(torch.float32)

    # -----------------------------------------------------------
    # 7. Token-level credit routing
    #    low_advantages  → action tokens only
    #    high_advantages → subgoal tokens on boundary turns
    # -----------------------------------------------------------
    # Ensure advantages are masked to their respective token spans
    low_advantages = low_advantages * action_mask.to(torch.float32)
    high_advantages = high_advantages * subgoal_mask.to(torch.float32)

    # -----------------------------------------------------------
    # 8. Store in batch
    # -----------------------------------------------------------
    batch.batch["advantages_low"] = low_advantages
    batch.batch["advantages_high"] = high_advantages
    batch.batch["returns_high"] = high_returns

    batch.batch["action_mask"] = action_mask
    batch.batch["subgoal_mask"] = subgoal_mask
    batch.batch["switch_mask"] = switch_mask

    # Combined advantage (for reference; actual routing uses per-level)
    batch.batch["advantages"] = low_advantages + high_advantages

    # Per-step milestone scores (for logging)
    batch.batch["milestone_scores"] = torch.as_tensor(
        milestone_scores, device=device, dtype=torch.float32,
    )

    return batch


# ---------------------------------------------------------------------------
#  Loss helpers (used by trainer)
# ---------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def normalize_advantages(adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = adv[mask]
    if flat.numel() <= 1:
        return adv
    mu = flat.mean()
    sigma = flat.std(unbiased=False) + 1e-8
    out = adv.clone()
    out[mask] = (flat - mu) / sigma
    return out


# ---------------------------------------------------------------------------
#  HSPO loss function (callable from trainer or worker)
# ---------------------------------------------------------------------------

def compute_hspo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages_low: torch.Tensor,
    advantages_high: torch.Tensor,
    action_mask: torch.Tensor,
    subgoal_mask: torch.Tensor,
    switch_mask: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: Optional[torch.Tensor],
    use_kl: bool,
    cfg: "HSPOConfig",
    phase: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level credit-routed HSPO loss.

    Called inside the worker where log_probs carries gradients.

    Args:
        log_probs: Fresh log-probs with grad (from model forward).
        old_log_probs: Detached log-probs from rollout.
        advantages_low: ABG low-level advantages.
        advantages_high: Macro high-level advantages.
        action_mask: 1 on action-tag token spans.
        subgoal_mask: 1 on subgoal-tag token spans.
        switch_mask: 1 on switch-tag token spans.
        response_mask: 1 on all response tokens.
        ref_log_prob: Reference log-probs for KL, or None.
        use_kl: Whether to apply KL penalties.
        cfg: HSPO hyperparameter config.
        phase: One of "warmup" / "executor" / "joint".

    Returns:
        (total_loss, metrics_dict)
    """
    device = log_probs.device
    response_mask = response_mask.to(torch.bool)

    # Default advantage tensors if not provided
    if advantages_low is None:
        advantages_low = torch.zeros_like(log_probs)
    if advantages_high is None:
        advantages_high = torch.zeros_like(log_probs)
    if action_mask is None:
        action_mask = response_mask
    if subgoal_mask is None:
        subgoal_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=device)
    if switch_mask is None:
        switch_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=device)

    loss_total = torch.tensor(0.0, device=device)
    metrics: Dict[str, float] = {}

    # ---- Low-level ABG loss (action tokens) ----
    if phase in ("warmup", "executor", "joint"):
        ratio = torch.exp(log_probs - old_log_probs)
        clip = cfg.low_clip_epsilon
        ratio_clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)

        pg_loss = -torch.min(ratio * advantages_low, ratio_clipped * advantages_low)
        loss_low = masked_mean(pg_loss, action_mask)
        loss_total = loss_total + cfg.alpha_l * loss_low

        if use_kl and ref_log_prob is not None:
            kl = torch.exp(log_probs - ref_log_prob) - 1 - (log_probs - ref_log_prob)
            kl_low = cfg.low_beta * masked_mean(kl, action_mask)
            loss_total = loss_total + kl_low
            metrics["actor/kl_low"] = kl_low.detach().item()

        metrics["actor/loss_low"] = loss_low.detach().item()

    # ---- High-level macro-PPO loss (subgoal tokens) ----
    if phase == "joint":
        ratio = torch.exp(log_probs - old_log_probs)
        clip = cfg.high_clip_epsilon
        ratio_clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)

        pg_loss = -torch.min(ratio * advantages_high, ratio_clipped * advantages_high)
        loss_high = masked_mean(pg_loss, subgoal_mask)
        loss_total = loss_total + cfg.alpha_h * loss_high

        if use_kl and ref_log_prob is not None:
            kl = torch.exp(log_probs - ref_log_prob) - 1 - (log_probs - ref_log_prob)
            kl_high = cfg.high_beta * masked_mean(kl, subgoal_mask)
            loss_total = loss_total + kl_high
            metrics["actor/kl_high"] = kl_high.detach().item()

        metrics["actor/loss_high"] = loss_high.detach().item()

    # ---- SFT regularization (tag format) ----
    if cfg.alpha_sft > 0:
        sft_mask = switch_mask | subgoal_mask | action_mask
        if sft_mask.any():
            loss_sft = cfg.alpha_sft * masked_mean(-log_probs, sft_mask)
            loss_total = loss_total + loss_sft
            metrics["actor/loss_sft"] = loss_sft.detach().item()

    metrics["actor/loss_total"] = loss_total.detach().item()
    return loss_total, metrics
