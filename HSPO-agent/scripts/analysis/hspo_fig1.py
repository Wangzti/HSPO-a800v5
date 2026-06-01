#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-ready HSPO diagnostic figure.

2x2 layout — all data from real ALFWorld SFT demonstrations and the actual
HSPO milestone scorer.  No simulated or placeholder data.

  (a) Trajectory length (steps) by task type — boxplot with jitter overlay.
  (b) Subgoal count per trajectory — dense bar chart, sorted, coloured by task.
  (c) Subgoal-type composition by task — stacked horizontal percentage bars.
  (d) Milestone scorer validation — mean ± SEM by subgoal type,
      grouped by Δrank > 0 (progress) vs. Δrank ≤ 0 (no progress).

The script does NOT add a global suptitle; the paper caption carries
the description instead.

Usage:
  python scripts/analysis/hspo_fig1.py \\
      --alfworld_raw /root/autodl-tmp/data/sft/alfworld_raw \\
      --project_root /root/projects/HSPO/HSPO-agent \\
      --output_dir /root/projects/HSPO/HSPO-agent/docs/figures
"""

from __future__ import annotations

import argparse, glob, json, math, os, re, sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

# ---------------------------------------------------------------------------
#  Font  (register local Arial so matplotlib can resolve "Arial" on Linux)
# ---------------------------------------------------------------------------
ARIAL_TTF = "/root/autodl-tmp/Fonts/arial.ttf"
if os.path.isfile(ARIAL_TTF):
    font_manager.fontManager.addfont(ARIAL_TTF)

# ---------------------------------------------------------------------------
#  Global camera-ready style  (applied once at module level)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#D9D9D9",
    "grid.alpha":        0.45,
    "grid.linewidth":    0.55,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ---------------------------------------------------------------------------
#  Colour palette  (colourblind-safe, restrained, paper-suitable)
# ---------------------------------------------------------------------------
TASK_DISPLAY = {
    "pick_and_place_simple":          "Pick",
    "look_at_obj_in_light":           "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep":  "Heat",
    "pick_cool_then_place_in_recep":  "Cool",
    "pick_two_obj_and_place":         "Pick2",
}

TASK_ORDER = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

TASK_COLORS = {
    "pick_and_place_simple":          "#4E79A7",   # muted blue
    "look_at_obj_in_light":           "#76B7B2",   # muted cyan
    "pick_clean_then_place_in_recep": "#59A14F",   # muted green
    "pick_heat_then_place_in_recep":  "#E15759",   # muted red
    "pick_cool_then_place_in_recep":  "#B07AA1",   # muted purple
    "pick_two_obj_and_place":         "#EDC948",   # muted yellow
}

SUBGOAL_ORDER = ["FindPick", "Navigate", "Clean", "Heat", "Cool", "Place", "Examine", "Other"]

SUBGOAL_COLORS = {
    "FindPick": "#4E79A7",
    "Navigate": "#BDBDBD",
    "Clean":    "#59A14F",
    "Heat":     "#E15759",
    "Cool":     "#B07AA1",
    "Place":    "#C7B66B",
    "Examine":  "#76B7B2",
    "Other":    "#D4C9A8",
}

SG_PROGRESS_COLOR   = "#4C956C"   # green for progress
SG_NOPROGRESS_COLOR = "#C96B6B"   # muted red for no progress

# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
def load_alfworld_trajectories(raw_dir: str) -> List[dict]:
    trajs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "traj_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            trajs.append(json.load(f))
    if not trajs:
        raise FileNotFoundError(f"No trajectories found under {raw_dir}/traj_*.json")
    print(f"[INFO] Loaded {len(trajs)} ALFWorld trajectories")
    return trajs


# ---------------------------------------------------------------------------
#  Subgoal parser
# ---------------------------------------------------------------------------
def _simple_subgoal_type(text: str) -> str:
    """Classify a subgoal string into canonical types.

    Priority: specific actions (Clean/Heat/Cool/Place/Examine) are checked
    *before* generic FindPick to avoid "pick up X and clean it" being
    mis-classified as FindPick.
    """
    s = (text or "").lower().strip()
    if any(k in s for k in ["clean", "wash", "rinse", "scrub"]):
        return "Clean"
    if any(k in s for k in ["heat", "warm", "microwave", "cook", "bake", "toast"]):
        return "Heat"
    if any(k in s for k in ["cool", "chill", "freeze", "fridge", "refrigerate"]):
        return "Cool"
    if any(k in s for k in ["place", "put", "store", "drop", "deliver", "set down", "throw"]):
        return "Place"
    if any(k in s for k in ["examine", "inspect", "look at", "turn on the lamp",
                              "turn on the light", "turn the light on",
                              "turn the lamp on", "use the lamp"]):
        return "Examine"
    if any(k in s for k in ["go to", "walk ", "turn around", "turn left", "turn right",
                              "move to", "move forward", "move the", "back up",
                              "step back", "step forward", "head to", "cross the room",
                              "face the", "stand in front", "bring ", "carry ",
                              "navigate to", "enter the", "return to", "go back",
                              "go forward", "walk over", "head "]):
        return "Navigate"
    if any(k in s for k in ["find", "pick", "take", "get", "grab", "retrieve",
                              "look for", "search for", "locate"]):
        return "FindPick"
    if "complete the task" in s or "task complete" in s:
        return "Other"
    return "Other"


def _extract_target(subgoal: str) -> str:
    sg = (subgoal or "").lower().strip()
    verbs = [
        "pick up", "look at", "find", "pick", "get", "take",
        "clean", "wash", "heat", "microwave", "warm",
        "cool", "chill", "place", "put", "store", "drop",
        "examine", "inspect", "search",
    ]
    for verb in sorted(verbs, key=len, reverse=True):
        idx = sg.find(verb)
        if idx < 0:
            continue
        rem = sg[idx + len(verb):].strip()
        for art in ["the ", "a ", "an ", "some "]:
            if rem.startswith(art):
                rem = rem[len(art):]
        for sep in [" and ", ", ", " in ", " on ", " at ", " to ", " from "]:
            cut = rem.find(sep)
            if cut > 0:
                rem = rem[:cut]
        rem = rem.strip().strip(".,!?;:\"'")
        if len(rem) > 1:
            return rem
    return sg.strip().strip(".,!?;:\"'")


def _extract_segments(traj: dict) -> List[dict]:
    """A segment is a maximal contiguous run sharing the same subgoal_idx."""
    steps = traj.get("steps", [])
    if not steps:
        return []
    segments = []
    start = 0
    cur_idx = steps[0].get("subgoal_idx", 0)
    for i in range(1, len(steps)):
        nxt_idx = steps[i].get("subgoal_idx", cur_idx)
        if nxt_idx != cur_idx:
            seg_steps = steps[start:i]
            sg = seg_steps[0].get("subgoal", "")
            segments.append({
                "subgoal_idx": cur_idx,
                "subgoal":     sg,
                "sg_type":     _simple_subgoal_type(sg),
                "length":      len(seg_steps),
                "steps":       seg_steps,
            })
            start = i
            cur_idx = nxt_idx
    seg_steps = steps[start:]
    sg = seg_steps[0].get("subgoal", "")
    segments.append({
        "subgoal_idx": cur_idx,
        "subgoal":     sg,
        "sg_type":     _simple_subgoal_type(sg),
        "length":      len(seg_steps),
        "steps":       seg_steps,
    })
    return segments


# ---------------------------------------------------------------------------
#  Analysis  (all based on REAL SFT trajectory data)
# ---------------------------------------------------------------------------
def _rank_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Mann-Whitney AUROC: P(score_pos > score_neg), ties = 0.5."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    max_pairs = 1_500_000
    if len(pos) * len(neg) > max_pairs:
        rng = np.random.default_rng(123)
        pos = rng.choice(pos, size=min(len(pos), 1500), replace=False)
        neg = rng.choice(neg, size=min(len(neg), 1500), replace=False)
    comp = pos[:, None] - neg[None, :]
    return float((np.sum(comp > 0) + 0.5 * np.sum(comp == 0)) / comp.size)


def compute_diagnostics(trajs: List[dict], project_root: Optional[str]) -> dict:
    """Compute all diagnostics from SFT trajectories + real HSPO scorer."""
    if project_root:
        sys.path.insert(0, project_root)

    # Try to import the real HSPO scorer
    scorer = None
    milestone_rank = None
    try:
        from recipe.hspo.milestone_scorer import AlfWorldMilestoneScorer
        scorer = AlfWorldMilestoneScorer()
        milestone_rank = scorer._milestone_rank
    except Exception as e:
        print(f"[WARN] milestone scorer not available; panel (d) will be empty. {e}",
              file=sys.stderr)

    traj_lengths: Dict[str, List[int]]      = defaultdict(list)
    subgoal_counts: List[Tuple[int, str, int]] = []  # (idx, task_type, n_subgoals)
    segment_type_counts: Dict[str, Counter] = defaultdict(Counter)
    progress_scores: Dict[str, dict]        = defaultdict(
        lambda: {"pos": [], "nonpos": []})

    for ti, traj in enumerate(trajs):
        tt = traj.get("task_type")
        steps = traj.get("steps", [])
        if not tt or not steps:
            continue

        traj_lengths[tt].append(len(steps))
        segments = _extract_segments(traj)
        subgoal_counts.append((ti, tt, len(segments)))

        # Segment-level composition (not step-level)
        for seg in segments:
            segment_type_counts[tt][seg["sg_type"]] += 1

        # Scorer validation on real transitions
        if scorer is not None and milestone_rank is not None:
            for i, step in enumerate(steps):
                subgoal = step.get("subgoal", "")
                sg_type = _simple_subgoal_type(subgoal)
                if sg_type in ("Other", "Navigate"):
                    continue
                obs_before = steps[i - 1].get("obs", "") if i > 0 else steps[0].get("obs", "")
                obs_after  = step.get("obs", "")
                action     = step.get("action", "")
                target     = _extract_target(subgoal)
                try:
                    rb = milestone_rank(sg_type, tt, target, obs_before, subgoal)
                    ra = milestone_rank(sg_type, tt, target, obs_after,  subgoal)
                    delta_rank = ra - rb
                    score = scorer.score(
                        subgoal=subgoal, state_before=obs_before,
                        action=action, state_after=obs_after,
                        gamefile=traj.get("gamefile"),
                    )
                    if delta_rank > 0:
                        progress_scores[sg_type]["pos"].append(float(score))
                    else:
                        progress_scores[sg_type]["nonpos"].append(float(score))
                except Exception:
                    continue

    # Segment composition as percentages (0–100)
    segment_composition: Dict[str, Dict[str, float]] = {}
    for tt, counter in segment_type_counts.items():
        total = sum(counter.values()) or 1
        segment_composition[tt] = {
            sg: 100.0 * counter.get(sg, 0) / total
            for sg in SUBGOAL_ORDER
        }

    return {
        "traj_lengths":        traj_lengths,
        "subgoal_counts":      subgoal_counts,
        "segment_composition": segment_composition,
        "progress_scores":     progress_scores,
    }


# ---------------------------------------------------------------------------
#  Shared plotting helpers
# ---------------------------------------------------------------------------
def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold")


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", alpha=0.0)


def _light_bbox() -> dict:
    """Camera-ready semi-transparent annotation box."""
    return dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=0.6,
        alpha=0.45,
    )


def _present_tasks(diag: dict) -> List[str]:
    return [t for t in TASK_ORDER
            if t in diag["traj_lengths"] and len(diag["traj_lengths"][t]) > 0]


# ---------------------------------------------------------------------------
#  (a) Trajectory length by task — boxplot + jitter
# ---------------------------------------------------------------------------
def _plot_traj_steps(ax: plt.Axes, diag: dict,
                     present_tasks: List[str]) -> None:
    labels = [TASK_DISPLAY[t] for t in present_tasks]
    colors = [TASK_COLORS[t] for t in present_tasks]
    data   = [diag["traj_lengths"][t] for t in present_tasks]

    bp = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.1),
        whiskerprops=dict(color="#555555", linewidth=0.8),
        capprops=dict(color="#555555", linewidth=0.8),
        boxprops=dict(linewidth=0.9),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(7)
    for i, arr in enumerate(data, start=1):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            continue
        sample_idx = rng.choice(len(arr), size=min(120, len(arr)), replace=False)
        xs = rng.normal(i, 0.035, size=len(sample_idx))
        ax.scatter(xs, arr[sample_idx], s=5, color="#555555", alpha=0.20, linewidths=0)
        ax.scatter(i, np.mean(arr), marker="D", s=24, color="#111111", zorder=4)
        ax.text(i, 0.965, f"n={len(arr)}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#5F5F5F")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("(a) Trajectory complexity by task", fontweight="bold", pad=10)
    _style_axis(ax, "Task type", "Trajectory length (steps)")


# ---------------------------------------------------------------------------
#  (b) Subgoal count per trajectory — DENSE BAR CHART  (kept from fig1)
# ---------------------------------------------------------------------------
def _plot_subgoal_dense_bars(ax: plt.Axes, diag: dict,
                              present_tasks: List[str]) -> None:
    """Dense bar chart: each bar = one trajectory, height = # subgoals.

    Trajectories are sorted by subgoal count, coloured by task type.
    This is the distinctive subfigure of hspo_fig1.
    """
    subgoal_counts = diag["subgoal_counts"]  # [(idx, task_type, n_subgoals)]
    sorted_data = sorted(subgoal_counts, key=lambda x: (x[2], x[1]))
    n = len(sorted_data)
    counts = np.array([x[2] for x in sorted_data], dtype=int)
    task_types = [x[1] for x in sorted_data]

    colors = [TASK_COLORS[tt] if tt in TASK_COLORS else "#BBBBBB"
              for tt in task_types]

    ax.bar(np.arange(n), counts, width=1.0, color=colors,
           edgecolor="none", linewidth=0)

    ax.set_xlim(-n * 0.01, n * 1.01)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("(b) Subgoal count per trajectory", fontweight="bold", pad=10)
    _style_axis(ax, f"Trajectory index  (N={n}, sorted by subgoal count)",
                "Number of subgoals")

    # Task-type legend (needed because x-axis is trajectory index, not task)
    handles = [
        Patch(facecolor=TASK_COLORS[t], edgecolor="white", label=TASK_DISPLAY[t])
        for t in present_tasks
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.50, -0.18), ncol=3,
              frameon=False, columnspacing=1.2, handlelength=1.2)

    # Stats annotation
    mean_sg, median_sg = float(np.mean(counts)), float(np.median(counts))
    ax.axhline(mean_sg, color="#555555", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.text(0.99, 0.91, f"mean = {mean_sg:.1f}\nmedian = {median_sg:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=_light_bbox())


# ---------------------------------------------------------------------------
#  (c) Subgoal-type composition by task — stacked horizontal percentage bars
# ---------------------------------------------------------------------------
def _plot_subgoal_composition(ax: plt.Axes, diag: dict,
                               present_tasks: List[str]) -> None:
    """Segment-level subgoal composition, not step-level."""
    labels = [TASK_DISPLAY[t] for t in present_tasks]
    y = np.arange(len(present_tasks))
    left = np.zeros(len(present_tasks), dtype=float)

    for sg in SUBGOAL_ORDER:
        vals = np.array([
            diag["segment_composition"].get(t, {}).get(sg, 0.0)
            for t in present_tasks
        ])
        if np.sum(vals) <= 0:
            continue
        ax.barh(y, vals, left=left, color=SUBGOAL_COLORS[sg],
                edgecolor="white", linewidth=0.5, label=sg)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("(c) Subgoal-type composition by task", fontweight="bold", pad=10)
    ax.set_xlabel("Share of subgoal segments")
    ax.set_ylabel("Task type")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", alpha=0.0)

    subgoal_handles = [
        Patch(facecolor=SUBGOAL_COLORS[sg], edgecolor="white", label=sg)
        for sg in SUBGOAL_ORDER
        if any(diag["segment_composition"].get(t, {}).get(sg, 0.0) > 0
               for t in present_tasks)
    ]
    ax.legend(handles=subgoal_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, frameon=False, columnspacing=1.0, handlelength=1.1)


# ---------------------------------------------------------------------------
#  (d) Milestone scorer validation — mean ± SEM
# ---------------------------------------------------------------------------
def _plot_scorer_validation(ax: plt.Axes, diag: dict) -> None:
    progress_scores = diag["progress_scores"]
    sg_types = [
        sg for sg in ["FindPick", "Clean", "Heat", "Cool", "Place", "Examine"]
        if sg in progress_scores
        and (len(progress_scores[sg]["pos"]) > 0
             or len(progress_scores[sg]["nonpos"]) > 0)
    ]

    if not sg_types:
        ax.text(0.5, 0.5, "Milestone scorer module not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) Scorer separates progress vs. non-progress steps",
                     fontweight="bold", pad=10)
        _style_axis(ax, "Subgoal type", "Milestone scorer output")
        return

    x = np.arange(len(sg_types))
    width = 0.34
    pos_means, nonpos_means = [], []
    pos_sem, nonpos_sem = [], []
    aucs = []

    for sg in sg_types:
        pos    = np.asarray(progress_scores[sg]["pos"],    dtype=float)
        nonpos = np.asarray(progress_scores[sg]["nonpos"], dtype=float)

        pos_means.append(float(np.mean(pos)) if len(pos) else np.nan)
        nonpos_means.append(float(np.mean(nonpos)) if len(nonpos) else np.nan)
        pos_sem.append(
            float(np.std(pos, ddof=1) / np.sqrt(len(pos))) if len(pos) > 1 else 0.0)
        nonpos_sem.append(
            float(np.std(nonpos, ddof=1) / np.sqrt(len(nonpos))) if len(nonpos) > 1 else 0.0)
        aucs.append(_rank_auc(pos.tolist(), nonpos.tolist()))

    # Progress bars
    ax.bar(x - width / 2, pos_means, width, yerr=pos_sem,
           color=SG_PROGRESS_COLOR, edgecolor="white", linewidth=0.4,
           capsize=2.2, label=r"$\Delta$rank $>$ 0")
    # No-progress bars
    ax.bar(x + width / 2, nonpos_means, width, yerr=nonpos_sem,
           color=SG_NOPROGRESS_COLOR, edgecolor="white", linewidth=0.4,
           capsize=2.2, label=r"$\Delta$rank $\leq$ 0")

    ax.axhline(0, color="#555555", linestyle=":", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(sg_types)
    ax.set_title("(d) Scorer separates progress vs. non-progress steps",
                 fontweight="bold", pad=10)
    _style_axis(ax, "Subgoal type", "Milestone scorer output")

    # AUROC annotation
    valid_auc = [a for a in aucs if np.isfinite(a)]
    if valid_auc:
        ax.text(0.98, 0.94,
                f"mean AUROC = {np.mean(valid_auc):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=_light_bbox())

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, columnspacing=1.2, handlelength=1.2)


# ---------------------------------------------------------------------------
#  Main figure
# ---------------------------------------------------------------------------
def plot_main_figure(diag: dict, output_prefix: str) -> None:
    present_tasks = _present_tasks(diag)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # (a)
    _plot_traj_steps(ax_a, diag, present_tasks)
    _add_panel_label(ax_a, "a")

    # (b) — dense bar chart (kept from original hspo_fig1)
    _plot_subgoal_dense_bars(ax_b, diag, present_tasks)
    _add_panel_label(ax_b, "b")

    # (c)
    _plot_subgoal_composition(ax_c, diag, present_tasks)
    _add_panel_label(ax_c, "c")

    # (d)
    _plot_scorer_validation(ax_d, diag)
    _add_panel_label(ax_d, "d")

    # No global suptitle — caption carries the description
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.93, bottom=0.12,
        wspace=0.26, hspace=0.55,
    )

    fig.savefig(output_prefix + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------
    #  Paper figure caption  (for direct inclusion in HSPO.tex)
    # ------------------------------------------------------------------
    n_trajs = len(diag["subgoal_counts"])
    # Count scorer steps
    total_scorer_steps = sum(
        len(v["pos"]) + len(v["nonpos"])
        for v in diag["progress_scores"].values()
    )
    task_names = [TASK_DISPLAY[t] for t in present_tasks]

    caption = (
        f"Figure X: Trajectory structure and milestone-scorer analysis on "
        f"ALFWorld expert demonstrations.\n"
        f"\\textbf{{(a)}} Trajectory length (steps) by task type "
        f"({', '.join(task_names)}).  "
        f"Boxplots show median, quartiles, and range; black diamonds mark the mean.\n"
        f"\\textbf{{(b)}} Number of subgoals per trajectory (N={n_trajs}), "
        f"sorted by subgoal count and coloured by task type.  "
        f"The dashed line indicates the overall mean.\n"
        f"\\textbf{{(c)}} Subgoal-type composition by task, expressed as the "
        f"share of subgoal-level segments (not low-level steps), coloured by "
        f"canonical subgoal type.\n"
        f"\\textbf{{(d)}} Mean milestone scorer output per subgoal type, "
        f"grouped by whether the step made forward progress "
        f"($\\Delta\\mathrm{{rank}}>0$, green) or not "
        f"($\\Delta\\mathrm{{rank}}\\leq 0$, red).  "
        f"Bars show mean $\\pm$ SEM over {total_scorer_steps:,} individual "
        f"step transitions aggregated from {n_trajs:,} trajectories; "
        f"higher scores for progress steps indicate correct scorer alignment.\n"
        f"The mean AUROC across subgoal types is reported in the bottom-right "
        f"corner of panel (d)."
    )

    print("\n" + "=" * 72)
    print("PAPER FIGURE CAPTION  (copy into HSPO.tex)")
    print("=" * 72)
    print(caption)
    print("=" * 72)


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------
def write_summary(diag: dict, output_prefix: str) -> None:
    path = output_prefix + "_summary.txt"
    lines = [f"num_trajectories={len(diag['subgoal_counts'])}"]
    for task in TASK_ORDER:
        if task in diag["traj_lengths"]:
            counts = [x[2] for x in diag["subgoal_counts"] if x[1] == task]
            lines.append(
                f"{TASK_DISPLAY[task]}: "
                f"n={len(diag['traj_lengths'][task])}, "
                f"mean_steps={np.mean(diag['traj_lengths'][task]):.2f}, "
                f"mean_subgoals={np.mean(counts):.2f}"
            )
    for sg in SUBGOAL_ORDER:
        entry = diag["progress_scores"].get(sg, {})
        n_pos = len(entry.get("pos", []))
        n_non = len(entry.get("nonpos", []))
        if n_pos + n_non > 0:
            auc = _rank_auc(entry.get("pos", []), entry.get("nonpos", []))
            lines.append(
                f"scorer_{sg}: pos={n_pos}, nonpos={n_non}, AUROC={auc:.4f}"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] saved {path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld_raw", required=True)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="fig_hspo_diagnostics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trajs = load_alfworld_trajectories(args.alfworld_raw)
    diag  = compute_diagnostics(trajs, args.project_root)

    output_prefix = os.path.join(args.output_dir, args.output_name)
    plot_main_figure(diag, output_prefix)
    write_summary(diag, output_prefix)

    print(f"\nSaved: {output_prefix}.pdf")
    print(f"Saved: {output_prefix}.png")


if __name__ == "__main__":
    main()
