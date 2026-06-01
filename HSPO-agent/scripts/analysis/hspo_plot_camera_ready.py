#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-ready diagnostic plots for HSPO.

Main figure: 2x2 ALFWorld diagnostic analysis.
  (a) Trajectory length by task type.
  (b) Subgoal count per trajectory by task type.
  (c) Subgoal-type composition by task type, computed over subgoal segments.
  (d) Intrinsic milestone-scorer sanity check on progress vs. non-progress transitions.

The script intentionally does NOT add a global suptitle, because paper figures
usually carry the global description in the caption rather than inside the figure.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


# -----------------------------
# Camera-ready style
# -----------------------------
ARIAL_TTF = "/root/autodl-tmp/Fonts/arial.ttf"
if os.path.isfile(ARIAL_TTF):
    # Register local Arial ttf so matplotlib can resolve "Arial" on Linux.
    font_manager.fontManager.addfont(ARIAL_TTF)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.alpha": 0.45,
    "grid.linewidth": 0.55,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
})

# Colorblind-safe, restrained scientific palette.
TASK_DISPLAY = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
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
    "pick_and_place_simple": "#4E79A7",          # muted blue
    "look_at_obj_in_light": "#76B7B2",           # muted cyan
    "pick_clean_then_place_in_recep": "#59A14F", # muted green
    "pick_heat_then_place_in_recep": "#E15759",  # muted red
    "pick_cool_then_place_in_recep": "#B07AA1",  # muted purple
    "pick_two_obj_and_place": "#EDC948",         # muted yellow
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


# -----------------------------
# Loading and parsing
# -----------------------------
def load_alfworld_trajectories(raw_dir: str) -> List[dict]:
    trajs = []
    pattern = os.path.join(raw_dir, "traj_*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            trajs.append(json.load(f))
    if not trajs:
        raise FileNotFoundError(f"No trajectory files found under: {pattern}")
    return trajs


def simple_subgoal_type(text: str) -> str:
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


def extract_target(subgoal: str) -> str:
    """Heuristic target extraction for scorer calls."""
    sg = (subgoal or "").lower().strip()
    verbs = [
        "pick up", "look at", "find", "pick", "get", "take",
        "clean", "wash", "heat", "microwave", "warm",
        "cool", "chill", "place", "put", "store", "drop",
        "examine", "inspect", "search",
    ]
    for verb in sorted(verbs, key=len, reverse=True):
        idx = sg.find(verb)
        if idx >= 0:
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


def extract_segments(traj: dict) -> List[dict]:
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
                "subgoal": sg,
                "sg_type": simple_subgoal_type(sg),
                "length": len(seg_steps),
                "steps": seg_steps,
            })
            start = i
            cur_idx = nxt_idx

    seg_steps = steps[start:]
    sg = seg_steps[0].get("subgoal", "")
    segments.append({
        "subgoal_idx": cur_idx,
        "subgoal": sg,
        "sg_type": simple_subgoal_type(sg),
        "length": len(seg_steps),
        "steps": seg_steps,
    })
    return segments


def rank_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Mann-Whitney AUROC: P(score_pos > score_neg), ties count as 0.5."""
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


# -----------------------------
# Diagnostics
# -----------------------------
def compute_diagnostics(trajs: List[dict]) -> dict:
    traj_lengths = defaultdict(list)
    subgoal_counts = defaultdict(list)
    segment_type_counts = defaultdict(Counter)
    progress_scores = defaultdict(lambda: {"pos": [], "nonpos": []})

    scorer = None
    milestone_rank = None
    try:
        from recipe.hspo.milestone_scorer import AlfWorldMilestoneScorer
        scorer = AlfWorldMilestoneScorer()
        milestone_rank = scorer._milestone_rank
    except Exception as e:
        print(f"[WARN] milestone scorer not available; panel (d) will be empty. Reason: {e}", file=sys.stderr)

    for traj in trajs:
        tt = traj.get("task_type")
        steps = traj.get("steps", [])
        if not tt or not steps:
            continue

        traj_lengths[tt].append(len(steps))
        segments = extract_segments(traj)
        subgoal_counts[tt].append(len(segments))

        # Segment-level composition, not low-level step composition.
        for seg in segments:
            segment_type_counts[tt][seg["sg_type"]] += 1

        # Intrinsic scorer sanity check over real transitions.
        if scorer is not None and milestone_rank is not None:
            for i, step in enumerate(steps):
                subgoal = step.get("subgoal", "")
                sg_type = simple_subgoal_type(subgoal)
                if sg_type in ("Other", "Navigate"):
                    continue

                obs_before = steps[i - 1].get("obs", "") if i > 0 else steps[0].get("obs", "")
                obs_after = step.get("obs", "")
                action = step.get("action", "")
                target = extract_target(subgoal)

                try:
                    rb = milestone_rank(sg_type, tt, target, obs_before, subgoal)
                    ra = milestone_rank(sg_type, tt, target, obs_after, subgoal)
                    delta_rank = ra - rb
                    score = scorer.score(
                        subgoal=subgoal,
                        state_before=obs_before,
                        action=action,
                        state_after=obs_after,
                        gamefile=traj.get("gamefile"),
                    )
                    if delta_rank > 0:
                        progress_scores[sg_type]["pos"].append(float(score))
                    else:
                        progress_scores[sg_type]["nonpos"].append(float(score))
                except Exception:
                    continue

    segment_composition = {}
    for tt, counter in segment_type_counts.items():
        total = sum(counter.values()) or 1
        segment_composition[tt] = {
            sg: 100.0 * counter.get(sg, 0) / total
            for sg in SUBGOAL_ORDER
        }

    return {
        "traj_lengths": traj_lengths,
        "subgoal_counts": subgoal_counts,
        "segment_composition": segment_composition,
        "progress_scores": progress_scores,
    }


# -----------------------------
# Plotting
# -----------------------------
def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10, 1.08, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12, fontweight="bold"
    )


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", alpha=0.0)


def plot_box_with_points(ax, data, labels, colors, ylabel, title):
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
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
        ax.text(
            i, 0.965, f"n={len(arr)}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#5F5F5F"
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold", pad=10)
    style_axis(ax, "Task type", ylabel)


def plot_main_figure(diag: dict, output_prefix: str) -> None:
    present_tasks = [
        t for t in TASK_ORDER
        if t in diag["traj_lengths"] and len(diag["traj_lengths"][t]) > 0
    ]
    labels = [TASK_DISPLAY[t] for t in present_tasks]
    task_colors = [TASK_COLORS[t] for t in present_tasks]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # (a)
    traj_data = [diag["traj_lengths"][t] for t in present_tasks]
    plot_box_with_points(
        ax_a, traj_data, labels, task_colors,
        ylabel="Trajectory length (steps)",
        title="(a) Trajectory complexity by task"
    )
    add_panel_label(ax_a, "a")

    # (b)
    sg_data = [diag["subgoal_counts"][t] for t in present_tasks]
    plot_box_with_points(
        ax_b, sg_data, labels, task_colors,
        ylabel="Number of subgoals",
        title="(b) Subgoal count per trajectory"
    )
    all_counts = np.concatenate([np.asarray(x) for x in sg_data if len(x) > 0])
    med_all = float(np.median(all_counts)) if len(all_counts) else np.nan
    mean_all = float(np.mean(all_counts)) if len(all_counts) else np.nan
    if np.isfinite(med_all):
        ax_b.axhline(med_all, color="#555555", linestyle="--", linewidth=0.9, alpha=0.65)
        ax_b.text(
            0.98, 0.93,
            f"mean={mean_all:.1f}\nmedian={med_all:.0f}",
            transform=ax_b.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#BBBBBB",
                linewidth=0.6,
                alpha=0.45,
            )
        )
    add_panel_label(ax_b, "b")

    # (c)
    y = np.arange(len(present_tasks))
    left = np.zeros(len(present_tasks), dtype=float)
    for sg in SUBGOAL_ORDER:
        vals = np.array([
            diag["segment_composition"].get(t, {}).get(sg, 0.0)
            for t in present_tasks
        ])
        if np.sum(vals) <= 0:
            continue
        ax_c.barh(
            y, vals, left=left,
            color=SUBGOAL_COLORS[sg],
            edgecolor="white",
            linewidth=0.5,
            label=sg,
        )
        left += vals

    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_c.set_title("(c) Subgoal-type composition by task", fontweight="bold", pad=10)
    ax_c.set_xlabel("Share of subgoal segments")
    ax_c.set_ylabel("Task type")
    ax_c.grid(axis="x", alpha=0.35)
    ax_c.grid(axis="y", alpha=0.0)
    subgoal_handles = [
        Patch(facecolor=SUBGOAL_COLORS[sg], edgecolor="white", label=sg)
        for sg in SUBGOAL_ORDER
        if any(diag["segment_composition"].get(t, {}).get(sg, 0.0) > 0 for t in present_tasks)
    ]
    ax_c.legend(
        handles=subgoal_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.1,
    )
    add_panel_label(ax_c, "c")

    # (d)
    progress_scores = diag["progress_scores"]
    sg_types = [
        sg for sg in ["FindPick", "Clean", "Heat", "Cool", "Place", "Examine"]
        if sg in progress_scores and
        (len(progress_scores[sg]["pos"]) > 0 or len(progress_scores[sg]["nonpos"]) > 0)
    ]

    if sg_types:
        x = np.arange(len(sg_types))
        width = 0.34

        pos_means, nonpos_means = [], []
        pos_sem, nonpos_sem = [], []
        aucs = []

        for sg in sg_types:
            pos = np.asarray(progress_scores[sg]["pos"], dtype=float)
            nonpos = np.asarray(progress_scores[sg]["nonpos"], dtype=float)

            pos_means.append(float(np.mean(pos)) if len(pos) else np.nan)
            nonpos_means.append(float(np.mean(nonpos)) if len(nonpos) else np.nan)

            pos_sem.append(float(np.std(pos, ddof=1) / np.sqrt(len(pos))) if len(pos) > 1 else 0.0)
            nonpos_sem.append(float(np.std(nonpos, ddof=1) / np.sqrt(len(nonpos))) if len(nonpos) > 1 else 0.0)

            aucs.append(rank_auc(pos.tolist(), nonpos.tolist()))

        ax_d.bar(
            x - width / 2, pos_means, width,
            yerr=pos_sem,
            color="#4C956C",
            edgecolor="white",
            linewidth=0.4,
            capsize=2.2,
            label=r"$\Delta$rank $>$ 0",
        )
        ax_d.bar(
            x + width / 2, nonpos_means, width,
            yerr=nonpos_sem,
            color="#C96B6B",
            edgecolor="white",
            linewidth=0.4,
            capsize=2.2,
            label=r"$\Delta$rank $\leq$ 0",
        )

        ax_d.axhline(0, color="#555555", linestyle=":", linewidth=0.9)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(sg_types)
        ax_d.set_title("(d) Scorer separates progress vs. non-progress steps", fontweight="bold", pad=10)
        ax_d.set_xlabel("Subgoal type")
        ax_d.set_ylabel("Milestone scorer output")
        ax_d.grid(axis="y", alpha=0.35)
        ax_d.grid(axis="x", alpha=0.0)

        valid_auc = [a for a in aucs if np.isfinite(a)]
        if valid_auc:
            ax_d.text(
                0.98, 0.94,
                f"mean AUROC={np.mean(valid_auc):.2f}",
                transform=ax_d.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="#BBBBBB",
                    linewidth=0.6,
                    alpha=0.45,
                )
            )

        ax_d.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.2,
        )
    else:
        ax_d.text(
            0.5, 0.5,
            "Milestone scorer module not available",
            ha="center", va="center",
            transform=ax_d.transAxes,
        )
        ax_d.set_title("(d) Scorer separates progress vs. non-progress steps", fontweight="bold", pad=10)
        ax_d.set_xlabel("Subgoal type")
        ax_d.set_ylabel("Milestone scorer output")

    add_panel_label(ax_d, "d")

    # Give legends enough room; no global suptitle.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.12, wspace=0.26, hspace=0.55)

    fig.savefig(output_prefix + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld_raw", required=True)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="fig_hspo_alfworld_diagnostics_camera_ready")
    args = parser.parse_args()

    if args.project_root:
        sys.path.insert(0, args.project_root)

    os.makedirs(args.output_dir, exist_ok=True)
    trajs = load_alfworld_trajectories(args.alfworld_raw)
    diag = compute_diagnostics(trajs)

    output_prefix = os.path.join(args.output_dir, args.output_name)
    plot_main_figure(diag, output_prefix)

    # Save a small text summary for reproducibility/caption writing.
    summary_path = output_prefix + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_trajectories={len(trajs)}\n")
        for task in TASK_ORDER:
            if task in diag["traj_lengths"]:
                f.write(
                    f"{TASK_DISPLAY[task]}: "
                    f"n={len(diag['traj_lengths'][task])}, "
                    f"mean_steps={np.mean(diag['traj_lengths'][task]):.2f}, "
                    f"mean_subgoals={np.mean(diag['subgoal_counts'][task]):.2f}\n"
                )

    print(f"Saved: {output_prefix}.pdf")
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
