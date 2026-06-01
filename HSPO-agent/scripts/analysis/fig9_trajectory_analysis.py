"""
Figure 9: Trajectory Structure Analysis for HSPO paper.

Four subfigures:
  (a) Subgoal Segment Length Distribution — by task type
  (b) ABG Group Size Distribution — natural group sizes with K_min threshold
  (c) Milestone Scorer Alignment — scorer outputs vs subgoal completion
  (d) Trajectory Success/Failure Pattern Mining — heatmap of failure modes

Data sources:
  - ALFWorld: /root/autodl-tmp/data/sft/alfworld_raw/traj_*.json (2012 trajectories)
  - WebShop:  product items only (no trajectory data available);
              generates representative statistics from item metadata.
"""

import json
import os
import re
import sys
import glob
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
ALFWORLD_RAW = "/root/autodl-tmp/data/sft/alfworld_raw"
WEBSHOP_DATA = "/root/autodl-tmp/data/webshop"
OUTPUT_DIR  = "/root/projects/HSPO/HSPO-agent/docs/figures"

# Task-type display names
TASK_DISPLAY = {
    "pick_and_place_simple":          "Pick&Place",
    "pick_two_obj_and_place":         "PickTwo&Place",
    "look_at_obj_in_light":           "Examine",
    "pick_heat_then_place_in_recep":  "Heat&Place",
    "pick_cool_then_place_in_recep":  "Cool&Place",
    "pick_clean_then_place_in_recep": "Clean&Place",
}
TASK_ORDER = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_parse_price(price_str: str) -> float:
    """Safely parse a price string that may contain ranges or be malformed."""
    try:
        s = str(price_str).replace("$", "").replace(",", "").strip()
        if not s:
            return 0.0
        # Handle price ranges: "19.99-27.99" → take max
        if "-" in s and not s.startswith("-"):
            parts = s.split("-")
            nums = []
            for p in parts:
                p = p.strip()
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            return max(nums) if nums else 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0
    except (TypeError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
def load_alfworld_trajectories() -> List[dict]:
    """Load all ALFWorld raw trajectory files."""
    trajs = []
    for fpath in sorted(glob.glob(os.path.join(ALFWORLD_RAW, "traj_*.json"))):
        with open(fpath) as f:
            trajs.append(json.load(f))
    print(f"Loaded {len(trajs)} ALFWorld trajectories")
    return trajs


def load_webshop_items() -> List[dict]:
    """Load WebShop product items."""
    items = []
    for fname in ["items_shuffle_1000.json", "items_ins_v2_1000.json"]:
        fpath = os.path.join(WEBSHOP_DATA, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
                if isinstance(data, list):
                    items.extend(data)
                elif isinstance(data, dict):
                    # items_ins_v2_1000.json is a dict keyed by item ID
                    items.extend(list(data.values()))
    print(f"Loaded {len(items)} WebShop items")
    return items


# ---------------------------------------------------------------------------
#  Analysis: (a) Subgoal Segment Length Distribution
# ---------------------------------------------------------------------------
def analyze_segment_lengths(trajs: List[dict]) -> dict:
    """Compute subgoal segment lengths per task type.

    A segment is a contiguous sequence of steps sharing the same subgoal_idx.
    """
    results: Dict[str, List[int]] = defaultdict(list)
    all_lengths: List[int] = []

    for traj in trajs:
        tt = traj["task_type"]
        steps = traj["steps"]
        if not steps:
            continue
        # Group by subgoal_idx
        current_idx = steps[0]["subgoal_idx"]
        seg_len = 1
        for i in range(1, len(steps)):
            if steps[i]["subgoal_idx"] == current_idx:
                seg_len += 1
            else:
                results[tt].append(seg_len)
                all_lengths.append(seg_len)
                current_idx = steps[i]["subgoal_idx"]
                seg_len = 1
        # Last segment
        results[tt].append(seg_len)
        all_lengths.append(seg_len)

    return {"by_task": dict(results), "all": all_lengths}


# ---------------------------------------------------------------------------
#  Analysis: (b) ABG Group Size Distribution
# ---------------------------------------------------------------------------
def analyze_abg_group_sizes(trajs: List[dict]) -> dict:
    """Simulate ABG anchor-group formation from trajectory data.

    Anchor key = (task_type_signature, canonical_state_hash, subgoal_text).
    We build natural groups and record their sizes.
    """
    anchor_groups: Dict[Tuple, List[int]] = defaultdict(list)

    # Build canonical state hashes (simplified: use object visibility pattern)
    for traj in trajs:
        tt = traj["task_type"]
        for step_idx, step in enumerate(traj["steps"]):
            obs = step.get("obs", "")
            # Build canonical state signature:
            # - visible objects (extracted from obs)
            # - inventory status
            state_sig = _canonical_state(obs)
            subgoal = step.get("subgoal", "").strip().lower()
            anchor = (tt, state_sig, subgoal)
            anchor_groups[anchor].append(step_idx)

    group_sizes = [len(v) for v in anchor_groups.values()]

    # Also track subgoal-instance-level grouping
    subgoal_text_groups: Dict[str, int] = defaultdict(int)
    for (tt, state_sig, sg), indices in anchor_groups.items():
        subgoal_text_groups[sg] += 1

    return {
        "group_sizes": group_sizes,
        "num_groups": len(anchor_groups),
        "num_unique_anchors": len(anchor_groups),
        "singleton_groups": sum(1 for s in group_sizes if s == 1),
        "sparse_groups": sum(1 for s in group_sizes if s < 2),
        "mean_size": np.mean(group_sizes) if group_sizes else 0,
        "median_size": np.median(group_sizes) if group_sizes else 0,
    }


def _canonical_state(obs: str) -> str:
    """Extract a canonical state signature from an observation string."""
    obs_lower = obs.lower()
    # Detect inventory
    has_inventory = bool(re.search(
        r"(?:in your inventory|you are carrying|you have)[:\s]*\w", obs_lower))
    # Count visible interactable objects
    receptacles = re.findall(r"(?:a|an)\s+(\w+\s*\d+)", obs_lower)
    # Detect if at a specific location
    at_match = re.search(r"you arrive at ([\w\s]+)\.", obs_lower)
    at_loc = at_match.group(1).strip() if at_match else "start"

    # Build a compact signature
    sig_parts = [
        f"inv:{1 if has_inventory else 0}",
        f"loc:{at_loc[:20]}" if at_match else "loc:start",
        f"obj:{len(receptacles)}",
    ]
    return "|".join(sig_parts)


# ---------------------------------------------------------------------------
#  Analysis: (c) Milestone Scorer Alignment
# ---------------------------------------------------------------------------
def analyze_scorer_alignment(trajs: List[dict]) -> dict:
    """Evaluate milestone scorer alignment with subgoal completion.

    Uses the deterministic v1 scorer to compute progress per step.
    Records (scorer_output, subgoal_was_completed) pairs.
    """
    sys.path.insert(0, "/root/projects/HSPO/HSPO-agent")
    from recipe.hspo.milestone_scorer import (
        AlfWorldMilestoneScorer, _parse_alfworld_subgoal_type,
    )

    scorer = AlfWorldMilestoneScorer()

    per_type: Dict[str, List[dict]] = defaultdict(list)

    for traj in trajs:
        tt = traj["task_type"]
        steps = traj["steps"]
        gamefile = traj.get("gamefile")

        for i in range(len(steps)):
            step = steps[i]
            subgoal = step.get("subgoal", "")
            obs_before = steps[i - 1]["obs"] if i > 0 else steps[0]["obs"]
            action = step.get("action", "")
            obs_after = step.get("obs", "")

            sg_type = _parse_alfworld_subgoal_type(subgoal)

            # Determine if this step completed the subgoal
            # A subgoal is complete if this is the last step with this subgoal_idx
            # OR if the next step has a different subgoal_idx and switch=SWITCH
            completed = False
            if i < len(steps) - 1:
                if (steps[i + 1]["subgoal_idx"] != step["subgoal_idx"]
                        and steps[i + 1].get("switch") == "SWITCH"):
                    completed = True
            else:
                # Last step of trajectory
                if traj.get("won", False):
                    completed = True

            # Compute milestone rank
            target = _extract_target(subgoal)
            rank = scorer._milestone_rank(sg_type, tt, target, obs_after, subgoal)

            # Compute the scorer value for this step
            score = scorer.score(
                subgoal=subgoal,
                state_before=obs_before,
                action=action,
                state_after=obs_after,
                gamefile=gamefile,
            )

            per_type[sg_type].append({
                "score": score,
                "rank_after": rank,
                "completed": completed,
                "subgoal": subgoal,
                "task_type": tt,
            })

    return dict(per_type)


def _extract_target(subgoal: str) -> str:
    """Extract target object from subgoal."""
    sg = subgoal.lower().strip()
    verbs = ["find", "pick up", "pick", "get", "take", "clean", "wash",
             "heat", "microwave", "warm", "cool", "chill", "place", "put",
             "store", "drop", "examine", "look at", "inspect", "search"]
    for verb in sorted(verbs, key=len, reverse=True):
        idx = sg.find(verb)
        if idx >= 0:
            remainder = sg[idx + len(verb):].strip()
            for art in ["the ", "a ", "an ", "some "]:
                if remainder.startswith(art):
                    remainder = remainder[len(art):]
            for sep in [" and ", ", ", " in ", " on ", " at ", " to ", " from "]:
                stop_idx = remainder.find(sep)
                if stop_idx > 0:
                    remainder = remainder[:stop_idx]
            remainder = remainder.strip().strip('.,!?;:"\'')
            if len(remainder) > 1:
                return remainder
            break
    return sg.strip().strip('.,!?;:"\'')


# ---------------------------------------------------------------------------
#  Analysis: (d) Trajectory Success/Failure Pattern Mining
# ---------------------------------------------------------------------------
def analyze_failure_patterns(trajs: List[dict]) -> dict:
    """Mine failure patterns across trajectories.

    Since SFT data is all-won, we analyze structural patterns:
    - Subgoal sequence position vs completion
    - Subgoal type frequencies
    - Step count to completion per task type
    """
    from recipe.hspo.milestone_scorer import _parse_alfworld_subgoal_type

    # Per task-type: subgoal sequence matrix
    # Row = task_type, Col = subgoal position in sequence, Value = completion rate
    max_seqs = 10  # max subgoal position

    # Also track: for each subgoal position, which subgoal type is most common
    pos_type_counts: Dict[int, Counter] = defaultdict(Counter)
    task_pos_types: Dict[str, Dict[int, Counter]] = defaultdict(
        lambda: defaultdict(Counter))

    # Per-task type: average rank progress per subgoal position
    task_pos_progress: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list))

    from recipe.hspo.milestone_scorer import AlfWorldMilestoneScorer
    scorer = AlfWorldMilestoneScorer()

    for traj in trajs:
        tt = traj["task_type"]
        steps = traj["steps"]
        subgoal_seen = set()

        for i, step in enumerate(steps):
            sg_idx = step["subgoal_idx"]
            subgoal = step.get("subgoal", "")
            sg_type = _parse_alfworld_subgoal_type(subgoal)
            obs = step.get("obs", "")

            # Position in subgoal sequence
            pos = len([s for s in subgoal_seen])
            if sg_idx not in subgoal_seen:
                subgoal_seen.add(sg_idx)
                pos = len(subgoal_seen) - 1

            if pos < max_seqs:
                pos_type_counts[pos][sg_type] += 1
                task_pos_types[tt][pos][sg_type] += 1

            # Track rank progress
            target = _extract_target(subgoal)
            rank = scorer._milestone_rank(sg_type, tt, target, obs, subgoal)
            if pos < max_seqs:
                task_pos_progress[tt][pos].append(rank)

    # Build heatmap: task_type x subgoal_position = avg rank
    task_list = [t for t in TASK_ORDER if t in task_pos_progress]
    heatmap = np.zeros((len(task_list), max_seqs))
    heatmap[:] = np.nan

    for ti, tt in enumerate(task_list):
        for pos in range(max_seqs):
            if task_pos_progress[tt][pos]:
                heatmap[ti, pos] = np.mean(task_pos_progress[tt][pos])

    # Also compute subgoal type frequency per position (for a secondary view)
    type_list = ["FindPick", "Clean", "Heat", "Cool", "Place", "Examine"]
    type_heatmap = np.zeros((len(task_list), len(type_list)))

    for ti, tt in enumerate(task_list):
        all_types = Counter()
        for pos in range(max_seqs):
            for sg_type, count in task_pos_types[tt][pos].items():
                all_types[sg_type] += count
        total = sum(all_types.values()) or 1
        for sj, st in enumerate(type_list):
            type_heatmap[ti, sj] = all_types.get(st, 0) / total

    return {
        "heatmap": heatmap,
        "task_list": task_list,
        "max_seqs": max_seqs,
        "type_heatmap": type_heatmap,
        "type_list": type_list,
        "pos_type_counts": pos_type_counts,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_figure_9(
    seg_data: dict,
    abg_data: dict,
    scorer_data: dict,
    failure_data: dict,
    output_path: str,
):
    """Generate the full Figure 9 with 4 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    # ---- (a) Subgoal Segment Length Distribution ----
    _plot_segment_lengths(ax_a, seg_data)

    # ---- (b) ABG Group Size Distribution ----
    _plot_abg_groups(ax_b, abg_data)

    # ---- (c) Milestone Scorer Alignment ----
    _plot_scorer_alignment(ax_c, scorer_data)

    # ---- (d) Trajectory Pattern Mining ----
    _plot_failure_heatmap(ax_d, failure_data)

    fig.tight_layout(pad=3.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved Figure 9 to {output_path}")


def _plot_segment_lengths(ax, seg_data: dict):
    """(a) Subgoal segment length distribution by task type."""
    by_task = seg_data["by_task"]
    task_list = [t for t in TASK_ORDER if t in by_task]

    # Use boxplot for compact view
    data = [by_task[t] for t in task_list]
    labels = [TASK_DISPLAY.get(t, t) for t in task_list]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.6, showfliers=True, flierprops=dict(
                        marker='o', markersize=2, alpha=0.4))

    for patch, color in zip(bp['boxes'], COLORS[:len(task_list)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay means as diamonds
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, marker='D', color='red',
               s=30, zorder=5, label='Mean')

    ax.set_xlabel("Task Type", fontsize=10)
    ax.set_ylabel("Subgoal Segment Length (steps)", fontsize=10)
    ax.set_title("(a) Subgoal Segment Length Distribution by Task Type",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=25, labelsize=8)

    # Add overall stats text
    all_lens = seg_data.get("all", [])
    if all_lens:
        ax.text(0.98, 0.95,
                f"Overall: μ={np.mean(all_lens):.1f}, σ={np.std(all_lens):.1f}\n"
                f"min={min(all_lens)}, max={max(all_lens)}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))


def _plot_abg_groups(ax, abg_data: dict):
    """(b) ABG natural group size distribution."""
    sizes = abg_data["group_sizes"]
    K_MIN = 2  # abg_min_group_size from config

    bins = np.arange(0.5, max(sizes) + 1.5, 1) if sizes else np.arange(0.5, 5.5, 1)
    counts, edges, patches = ax.hist(
        sizes, bins=bins, edgecolor="white", alpha=0.7, color="#4C72B0",
        align="mid", rwidth=0.8)

    # Color sparse groups differently (size < K_MIN)
    for i, patch in enumerate(patches):
        if edges[i] < K_MIN:
            patch.set_facecolor("#C44E52")
            patch.set_alpha(0.8)

    # Add K_min threshold line
    ax.axvline(x=K_MIN - 0.5, color="#C44E52", linestyle="--", linewidth=2,
               label=f"$K_{{\\min}} = {K_MIN}$ (ABG threshold)")

    # Annotate sparse fraction
    sparse_count = sum(1 for s in sizes if s < K_MIN)
    total_count = len(sizes)
    sparse_pct = 100 * sparse_count / total_count if total_count > 0 else 0
    ax.text(0.98, 0.95,
            f"Total anchors: {abg_data['num_unique_anchors']}\n"
            f"Singletons (size=1): {abg_data['singleton_groups']}\n"
            f"Sparse (<{K_MIN}): {sparse_count} ({sparse_pct:.1f}%)\n"
            f"Median group size: {abg_data['median_size']:.1f}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       alpha=0.7))

    ax.set_xlabel("Natural Group Size |G_nat(κ)|", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("(b) ABG Anchor Group Size Distribution",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))


def _plot_scorer_alignment(ax, scorer_data: dict):
    """(c) Milestone scorer alignment: score distribution by completion status."""
    type_list = sorted(scorer_data.keys())
    color_map = {
        "FindPick": "#4C72B0", "Clean": "#55A868", "Heat": "#C44E52",
        "Cool": "#8172B2", "Place": "#CCB974", "Examine": "#64B5CD",
    }

    # For each subgoal type: compute P(completed | score > threshold)
    x_positions = []
    labels = []
    completion_rates = []
    mean_scores_completed = []
    mean_scores_not_completed = []

    for sg_type in type_list:
        entries = scorer_data[sg_type]
        if len(entries) < 5:
            continue

        completed_entries = [e for e in entries if e["completed"]]
        not_completed = [e for e in entries if not e["completed"]]

        comp_rate = len(completed_entries) / len(entries) if entries else 0

        x_positions.append(len(labels))
        labels.append(sg_type)
        completion_rates.append(comp_rate)
        mean_scores_completed.append(
            np.mean([e["score"] for e in completed_entries])
            if completed_entries else 0)
        mean_scores_not_completed.append(
            np.mean([e["score"] for e in not_completed])
            if not_completed else 0)

    # Grouped bar: mean score for completed vs not-completed
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, mean_scores_completed, width,
                   label="Subgoal Completed", color="#55A868", alpha=0.8)
    bars2 = ax.bar(x + width / 2, mean_scores_not_completed, width,
                   label="Subgoal Not Completed", color="#C44E52", alpha=0.8)

    # Add completion rate as text
    for i, rate in enumerate(completion_rates):
        ax.text(i, max(mean_scores_completed[i], mean_scores_not_completed[i]) + 0.05,
                f"{rate:.0%}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Subgoal Type", fontsize=10)
    ax.set_ylabel("Mean Milestone Score", fontsize=10)
    ax.set_title("(c) Milestone Scorer Alignment by Subgoal Type",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # Add alignment quality note
    ax.text(0.98, 0.12,
            "Higher completed scores → good alignment",
            transform=ax.transAxes, fontsize=7, ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.4))


def _plot_failure_heatmap(ax, failure_data: dict):
    """(d) Trajectory pattern heatmap: task_type x subgoal_position."""
    heatmap = failure_data["heatmap"]
    task_list = failure_data["task_list"]
    display_labels = [TASK_DISPLAY.get(t, t) for t in task_list]
    max_seqs = failure_data["max_seqs"]

    # Determine safe vmax
    valid_vals = heatmap[~np.isnan(heatmap)]
    vmax_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else 1.0
    vmin_val = 0.0

    # Mask NaN values for display
    masked = np.ma.masked_invalid(heatmap)

    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn",
                   vmin=vmin_val, vmax=vmax_val)

    # Annotate cells
    for i in range(len(task_list)):
        for j in range(max_seqs):
            val = heatmap[i, j]
            if not np.isnan(val):
                text_color = "black" if val < vmax_val * 0.6 else "white"
                ax.text(j, i, f"{val:.1f}",
                        ha="center", va="center", fontsize=7,
                        color=text_color)

    ax.set_yticks(range(len(task_list)))
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("Subgoal Position in Sequence", fontsize=10)
    ax.set_ylabel("Task Type", fontsize=10)
    ax.set_title("(d) Avg Milestone Rank by Task & Subgoal Position",
                 fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Avg Milestone Rank", fontsize=9)


# ---------------------------------------------------------------------------
#  Additional standalone plots
# ---------------------------------------------------------------------------
def plot_webshop_category_distribution(items: List[dict], output_path: str):
    """Generate a WebShop product category distribution chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (Left) Category distribution
    cat_counts = Counter(item.get("category", "unknown") for item in items)
    cats = sorted(cat_counts.keys())
    counts = [cat_counts[c] for c in cats]

    colors_cat = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    axes[0].bar(cats, counts, color=colors_cat[:len(cats)], alpha=0.8)
    axes[0].set_xlabel("Product Category", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("WebShop Product Categories", fontsize=11, fontweight="bold")
    for i, (c, v) in enumerate(zip(cats, counts)):
        axes[0].text(i, v + 3, str(v), ha="center", fontsize=9)

    # (Right) Price distribution by category
    for cat, color in zip(cats, colors_cat[:len(cats)]):
        prices = []
        for item in items:
            if item.get("category") == cat:
                price = _safe_parse_price(str(item.get("pricing", "$0")))
                if price > 0:
                    prices.append(price)
        if prices:
            axes[1].hist(prices, bins=30, alpha=0.5, label=cat, color=color)

    axes[1].set_xlabel("Price ($)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].set_title("WebShop Price Distribution by Category", fontsize=11,
                      fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, np.percentile(
        [p for item in items for p_str in [str(item.get("pricing", "$0"))]
         for p in [_safe_parse_price(p_str)]
         if p > 0], 95) if items else 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved WebShop analysis to {output_path}")


def plot_summary_statistics(trajs: List[dict], output_path: str):
    """Generate summary statistics plot for the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (1) Steps per trajectory by task type
    task_list = [t for t in TASK_ORDER if
                 any(tr["task_type"] == t for tr in trajs)]
    steps_data = []
    for tt in task_list:
        steps_data.append([len(tr["steps"]) for tr in trajs if tr["task_type"] == tt])

    bp = axes[0].boxplot(steps_data, tick_labels=[TASK_DISPLAY.get(t, t) for t in task_list],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], COLORS[:len(task_list)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel("Task Type", fontsize=9)
    axes[0].set_ylabel("Total Steps", fontsize=9)
    axes[0].set_title("Steps per Trajectory", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=25, labelsize=7)

    # (2) Subgoal count distribution
    all_sg_counts = []
    for tr in trajs:
        sg_set = set(s["subgoal_idx"] for s in tr["steps"])
        all_sg_counts.append(len(sg_set))
    axes[1].hist(all_sg_counts, bins=range(1, 8), edgecolor="white",
                 color="#55A868", alpha=0.7, rwidth=0.85)
    axes[1].set_xlabel("Number of Subgoals", fontsize=9)
    axes[1].set_ylabel("Frequency", fontsize=9)
    axes[1].set_title("Subgoal Count per Trajectory", fontsize=10, fontweight="bold")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].axvline(x=np.mean(all_sg_counts), color="red", linestyle="--",
                    label=f"μ={np.mean(all_sg_counts):.1f}")
    axes[1].legend(fontsize=7)

    # (3) Subgoal type distribution across all trajectories
    from recipe.hspo.milestone_scorer import _parse_alfworld_subgoal_type
    sg_type_counts = Counter()
    for tr in trajs:
        for step in tr["steps"]:
            sg_type = _parse_alfworld_subgoal_type(step.get("subgoal", ""))
            sg_type_counts[sg_type] += 1

    type_order = ["FindPick", "Clean", "Heat", "Cool", "Place", "Examine"]
    type_values = [sg_type_counts.get(t, 0) for t in type_order]
    type_colors_map = {"FindPick": "#4C72B0", "Clean": "#55A868", "Heat": "#C44E52",
                       "Cool": "#8172B2", "Place": "#CCB974", "Examine": "#64B5CD"}
    bars = axes[2].bar(type_order, type_values,
                       color=[type_colors_map[t] for t in type_order], alpha=0.8)
    axes[2].set_xlabel("Subgoal Type", fontsize=9)
    axes[2].set_ylabel("Count", fontsize=9)
    axes[2].set_title("Subgoal Type Distribution", fontsize=10, fontweight="bold")
    axes[2].tick_params(axis='x', rotation=25, labelsize=7)
    for bar, val in zip(bars, type_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved summary statistics to {output_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HSPO Figure 9: Trajectory Structure Analysis")
    print("=" * 60)

    # Load data
    trajs = load_alfworld_trajectories()
    web_items = load_webshop_items()

    # ---- Analysis (a): Segment Lengths ----
    print("\n[1/4] Computing subgoal segment length distributions...")
    seg_data = analyze_segment_lengths(trajs)
    print(f"  Total segments: {len(seg_data['all'])}")
    print(f"  Mean length: {np.mean(seg_data['all']):.2f} ± {np.std(seg_data['all']):.2f}")

    # ---- Analysis (b): ABG Group Sizes ----
    print("\n[2/4] Computing ABG anchor group size distribution...")
    abg_data = analyze_abg_group_sizes(trajs)
    print(f"  Unique anchors: {abg_data['num_unique_anchors']}")
    print(f"  Mean group size: {abg_data['mean_size']:.2f}")
    print(f"  Median group size: {abg_data['median_size']:.1f}")
    print(f"  Singleton groups: {abg_data['singleton_groups']}")
    print(f"  Sparse groups (<2): {abg_data['sparse_groups']}")

    # ---- Analysis (c): Scorer Alignment ----
    print("\n[3/4] Computing milestone scorer alignment...")
    scorer_data = analyze_scorer_alignment(trajs)
    for sg_type, entries in sorted(scorer_data.items()):
        completed = sum(1 for e in entries if e["completed"])
        mean_score_comp = np.mean([e["score"] for e in entries if e["completed"]]) if completed else 0
        mean_score_ncomp = np.mean([e["score"] for e in entries if not e["completed"]]) if (len(entries) - completed) else 0
        print(f"  {sg_type}: {len(entries)} steps, "
              f"{completed}/{len(entries)} completed, "
              f"μ_score(completed)={mean_score_comp:.2f}, "
              f"μ_score(not)={mean_score_ncomp:.2f}")

    # ---- Analysis (d): Failure Patterns ----
    print("\n[4/4] Mining trajectory success/failure patterns...")
    failure_data = analyze_failure_patterns(trajs)
    print(f"  Task list: {failure_data['task_list']}")
    print(f"  Heatmap shape: {failure_data['heatmap'].shape}")

    # ---- Plot ----
    print("\nGenerating plots...")
    plot_figure_9(
        seg_data, abg_data, scorer_data, failure_data,
        os.path.join(OUTPUT_DIR, "fig9_trajectory_analysis.png"),
    )

    # Additional plots
    plot_summary_statistics(
        trajs,
        os.path.join(OUTPUT_DIR, "fig_summary_statistics.png"),
    )
    plot_webshop_category_distribution(
        web_items,
        os.path.join(OUTPUT_DIR, "fig_webshop_analysis.png"),
    )

    print("\nDone. Output files:")
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "fig*.png"))):
        print(f"  {f}")


if __name__ == "__main__":
    main()
