import json, os, re, glob, argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, LogLocator
from matplotlib.patches import Patch

# -----------------------------
# Styling
# -----------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linestyle': '-',
})

TASK_DISPLAY = {
    'pick_and_place_simple': 'Pick',
    'pick_two_obj_and_place': 'Pick2',
    'look_at_obj_in_light': 'Look',
    'pick_heat_then_place_in_recep': 'Heat',
    'pick_cool_then_place_in_recep': 'Cool',
    'pick_clean_then_place_in_recep': 'Clean',
}
TASK_ORDER = [
    'pick_and_place_simple',
    'look_at_obj_in_light',
    'pick_clean_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_two_obj_and_place',
]
TASK_COLORS = {
    'pick_and_place_simple': '#4C72B0',
    'look_at_obj_in_light': '#64B5CD',
    'pick_clean_then_place_in_recep': '#55A868',
    'pick_heat_then_place_in_recep': '#C44E52',
    'pick_cool_then_place_in_recep': '#8172B2',
    'pick_two_obj_and_place': '#CCB974',
}
SUBGOAL_ORDER = ['FindPick', 'Clean', 'Heat', 'Cool', 'Place', 'Examine', 'Other']
SUBGOAL_COLORS = {
    'FindPick': '#4C72B0',
    'Clean': '#55A868',
    'Heat': '#C44E52',
    'Cool': '#8172B2',
    'Place': '#CCB974',
    'Examine': '#64B5CD',
    'Other': '#B9B9B9',
}

# -----------------------------
# Helpers
# -----------------------------
def safe_parse_price(s: str) -> float:
    try:
        s = str(s).replace('$', '').replace(',', '').strip()
        if not s:
            return 0.0
        if '-' in s and not s.startswith('-'):
            vals = [float(x.strip()) for x in s.split('-') if x.strip()]
            return max(vals) if vals else 0.0
        return float(s)
    except Exception:
        return 0.0


def load_alfworld_trajectories(raw_dir: str) -> List[dict]:
    trajs = []
    for fpath in sorted(glob.glob(os.path.join(raw_dir, 'traj_*.json'))):
        with open(fpath, 'r', encoding='utf-8') as f:
            trajs.append(json.load(f))
    return trajs


def canonical_state(obs: str) -> str:
    obs_lower = (obs or '').lower()
    has_inventory = bool(re.search(r'(?:in your inventory|you are carrying|you have)[:\s]*\w', obs_lower))
    receptacles = re.findall(r'(?:a|an)\s+(\w+\s*\d+)', obs_lower)
    at_match = re.search(r'you arrive at ([\w\s]+)\.', obs_lower)
    at_loc = at_match.group(1).strip() if at_match else 'start'
    return f"inv:{int(has_inventory)}|loc:{at_loc[:24]}|obj:{len(receptacles)}"


def simple_subgoal_type(text: str) -> str:
    s = (text or '').lower().strip()
    if any(k in s for k in ['find', 'pick', 'take', 'get']):
        return 'FindPick'
    if any(k in s for k in ['clean', 'wash']):
        return 'Clean'
    if any(k in s for k in ['heat', 'warm', 'microwave']):
        return 'Heat'
    if any(k in s for k in ['cool', 'chill', 'freeze']):
        return 'Cool'
    if any(k in s for k in ['place', 'put', 'store', 'drop']):
        return 'Place'
    if any(k in s for k in ['examine', 'inspect', 'look at']):
        return 'Examine'
    return 'Other'


def extract_segments(traj: dict) -> List[dict]:
    steps = traj.get('steps', [])
    if not steps:
        return []
    segments = []
    start = 0
    cur_idx = steps[0].get('subgoal_idx', 0)
    for i in range(1, len(steps)):
        nxt = steps[i].get('subgoal_idx', cur_idx)
        if nxt != cur_idx:
            seg_steps = steps[start:i]
            sg = seg_steps[0].get('subgoal', '')
            segments.append({
                'subgoal_idx': cur_idx,
                'subgoal': sg,
                'sg_type': simple_subgoal_type(sg),
                'length': len(seg_steps),
                'steps': seg_steps,
            })
            start = i
            cur_idx = nxt
    seg_steps = steps[start:]
    sg = seg_steps[0].get('subgoal', '')
    segments.append({
        'subgoal_idx': cur_idx,
        'subgoal': sg,
        'sg_type': simple_subgoal_type(sg),
        'length': len(seg_steps),
        'steps': seg_steps,
    })
    return segments

# -----------------------------
# Data analysis
# -----------------------------
def compute_diagnostics(trajs: List[dict], k_min: int = 2, m_target: int = 8, b_max: int = 4):
    # (a) trajectory lengths
    traj_lengths = defaultdict(list)
    subgoal_counts = defaultdict(list)
    segment_lengths = defaultdict(list)
    segment_type_comp = defaultdict(Counter)

    # (d) ABG natural groups
    anchor_groups = defaultdict(int)

    # (f) scorer sanity check (intrinsic): classify by delta progress > 0 vs <= 0
    # NOTE: this is a sanity check, not independent external validation.
    progress_scores = defaultdict(lambda: {'pos': [], 'nonpos': []})

    # optional milestone scorer import
    scorer = None
    milestone_rank = None
    try:
        import sys
        # caller should already have inserted project_root if needed.
        from recipe.hspo.milestone_scorer import AlfWorldMilestoneScorer
        scorer = AlfWorldMilestoneScorer()
        milestone_rank = scorer._milestone_rank
    except Exception:
        pass

    for tr in trajs:
        tt = tr.get('task_type')
        steps = tr.get('steps', [])
        if not steps:
            continue
        traj_lengths[tt].append(len(steps))
        segs = extract_segments(tr)
        subgoal_counts[tt].append(len(segs))
        for seg in segs:
            segment_lengths[tt].append(seg['length'])
            segment_type_comp[tt][seg['sg_type']] += 1

        for st in steps:
            anchor = (tt, canonical_state(st.get('obs', '')), st.get('subgoal', '').strip().lower())
            anchor_groups[anchor] += 1

        # scorer sanity check, if scorer module is available
        if scorer is not None and milestone_rank is not None:
            for i, st in enumerate(steps):
                sg = st.get('subgoal', '')
                sg_type = simple_subgoal_type(sg)
                if sg_type == 'Other':
                    continue
                obs_before = steps[i-1].get('obs', '') if i > 0 else steps[0].get('obs', '')
                obs_after = st.get('obs', '')
                action = st.get('action', '')
                try:
                    # heuristic target extraction: mild fallback
                    target = sg.split()[-1] if sg else ''
                    rank_before = milestone_rank(sg_type, tt, target, obs_before, sg)
                    rank_after = milestone_rank(sg_type, tt, target, obs_after, sg)
                    delta = rank_after - rank_before
                    score = scorer.score(subgoal=sg, state_before=obs_before, action=action, state_after=obs_after, gamefile=tr.get('gamefile'))
                    if delta > 0:
                        progress_scores[sg_type]['pos'].append(score)
                    else:
                        progress_scores[sg_type]['nonpos'].append(score)
                except Exception:
                    continue

    # build composition matrix using SEGMENTS, not steps
    comp_pct = {}
    for tt, counter in segment_type_comp.items():
        total = sum(counter.values()) or 1
        comp_pct[tt] = {sg: counter.get(sg, 0) / total for sg in SUBGOAL_ORDER}

    group_sizes = list(anchor_groups.values())
    eligible_before = [g for g in group_sizes if g >= k_min]
    covered_trans_before = sum(eligible_before)
    total_trans = sum(group_sizes)
    anchor_cov_before = (sum(1 for g in group_sizes if g >= k_min) / len(group_sizes) * 100) if group_sizes else 0.0
    trans_cov_before = (covered_trans_before / total_trans * 100) if total_trans else 0.0

    extra_branches = sum(min(max(m_target - g, 0), b_max) for g in group_sizes if g < k_min)
    # after supplementation, every anchor becomes eligible by construction if at least one branch is added to groups below threshold
    anchor_cov_after = 100.0 if group_sizes else 0.0
    trans_cov_after = 100.0 if total_trans else 0.0

    return {
        'traj_lengths': traj_lengths,
        'subgoal_counts': subgoal_counts,
        'segment_lengths': segment_lengths,
        'segment_type_comp': comp_pct,
        'anchor_group_sizes': group_sizes,
        'anchor_cov_before': anchor_cov_before,
        'trans_cov_before': trans_cov_before,
        'anchor_cov_after': anchor_cov_after,
        'trans_cov_after': trans_cov_after,
        'extra_branches': extra_branches,
        'num_anchors': len(group_sizes),
        'singleton_pct': (sum(1 for g in group_sizes if g == 1) / len(group_sizes) * 100) if group_sizes else 0.0,
        'progress_scores': progress_scores,
        'k_min': k_min,
        'm_target': m_target,
        'b_max': b_max,
    }

# -----------------------------
# Plot utilities
# -----------------------------
def add_panel_label(ax, label: str):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


def styled_boxplot(ax, data, labels, colors, ylabel, title, annotate_n=True):
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    # means + light jittered outliers sample
    for i, arr in enumerate(data, 1):
        arr = np.asarray(arr)
        if len(arr) == 0:
            continue
        ax.scatter(i, arr.mean(), marker='D', color='black', s=24, zorder=4)
        # small random sample of points for texture
        if len(arr) > 0:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(arr), size=min(120, len(arr)), replace=False)
            xs = rng.normal(loc=i, scale=0.04, size=len(idx))
            ax.scatter(xs, arr[idx], s=5, color='gray', alpha=0.25, zorder=2, linewidths=0)
        if annotate_n:
            ax.text(i, ax.get_ylim()[1]*0.965 if ax.get_ylim()[1] > 0 else 0.95,
                    f'n={len(arr)}', ha='center', va='bottom', fontsize=8, color='dimgray')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=0)
    ax.set_axisbelow(True)


def plot_diagnostics(diag: dict, output_prefix: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=False)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    present_tasks = [t for t in TASK_ORDER if t in diag['traj_lengths'] and len(diag['traj_lengths'][t]) > 0]
    labels = [TASK_DISPLAY[t] for t in present_tasks]
    colors = [TASK_COLORS[t] for t in present_tasks]

    # (a) trajectory length
    traj_data = [diag['traj_lengths'][t] for t in present_tasks]
    styled_boxplot(ax_a, traj_data, labels, colors,
                   ylabel='Trajectory length (steps)',
                   title='Trajectory complexity')
    add_panel_label(ax_a, 'a')

    # (b) subgoals per trajectory (replaces ambiguous segment-length-only main panel)
    sg_data = [diag['subgoal_counts'][t] for t in present_tasks]
    styled_boxplot(ax_b, sg_data, labels, colors,
                   ylabel='# subgoals per trajectory',
                   title='Subgoal count by task')
    med_all = np.median(np.concatenate([np.asarray(x) for x in sg_data if len(x) > 0])) if sg_data else 0
    ax_b.axhline(med_all, color='#D55E5E', linestyle='--', linewidth=1.2, alpha=0.9)
    ax_b.text(0.02, 0.92, f'overall median={med_all:.1f}', transform=ax_b.transAxes,
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='lightgray'), fontsize=9)
    add_panel_label(ax_b, 'b')

    # (c) subgoal composition by task type, COUNTING SEGMENTS not steps
    y = np.arange(len(present_tasks))
    left = np.zeros(len(present_tasks))
    for sg in SUBGOAL_ORDER:
        vals = [diag['segment_type_comp'][t].get(sg, 0) * 100 for t in present_tasks]
        if np.sum(vals) == 0:
            continue
        ax_c.barh(y, vals, left=left, color=SUBGOAL_COLORS[sg], edgecolor='white', linewidth=0.5, label=sg)
        left += np.array(vals)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.xaxis.set_major_formatter(PercentFormatter(100))
    ax_c.set_xlabel('Share of subgoal segments')
    ax_c.set_title('Subgoal-type composition')
    ax_c.grid(axis='x', alpha=0.18)
    ax_c.grid(axis='y', visible=False)
    handles = [Patch(facecolor=SUBGOAL_COLORS[sg], edgecolor='white', label=sg) for sg in SUBGOAL_ORDER if any(diag['segment_type_comp'][t].get(sg, 0) > 0 for t in present_tasks)]
    ax_c.legend(handles=handles, ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, columnspacing=0.8, handlelength=1.0)
    add_panel_label(ax_c, 'c')

    # (d) natural anchor sparsity, log scale reasonable
    sizes = diag['anchor_group_sizes']
    bins_labels = ['1', '2', '3', '4', '5-8', '9-16', '>16']
    binned = [0] * len(bins_labels)
    for g in sizes:
        if g == 1:
            binned[0] += 1
        elif g == 2:
            binned[1] += 1
        elif g == 3:
            binned[2] += 1
        elif g == 4:
            binned[3] += 1
        elif 5 <= g <= 8:
            binned[4] += 1
        elif 9 <= g <= 16:
            binned[5] += 1
        else:
            binned[6] += 1
    bar_colors = ['#D97A7A'] + ['#6C8EBF'] * (len(binned) - 1)
    bars = ax_d.bar(np.arange(len(binned)), binned, color=bar_colors)
    ax_d.set_yscale('log')
    ax_d.yaxis.set_major_locator(LogLocator(base=10))
    ax_d.set_xticks(np.arange(len(bins_labels)))
    ax_d.set_xticklabels(bins_labels)
    ax_d.set_xlabel(r'Natural group size $|G_{nat}(\kappa)|$')
    ax_d.set_ylabel('# anchors (log scale)')
    ax_d.set_title('ABG natural-anchor sparsity')
    for rect, val in zip(bars, binned):
        if val > 0:
            ax_d.text(rect.get_x() + rect.get_width()/2, val*1.08, f'{val:,}', ha='center', va='bottom', fontsize=8)
    ax_d.text(0.98, 0.94,
              f"anchors={diag['num_anchors']:,}\nsingleton={diag['singleton_pct']:.1f}%\neligible (|G|≥{diag['k_min']}): {diag['anchor_cov_before']:.1f}%",
              transform=ax_d.transAxes, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='lightgray'))
    add_panel_label(ax_d, 'd')

    # (e) branch supplementation effect: clarify what is being measured
    names = ['Eligible anchors', 'Covered natural\ntransitions']
    before = [diag['anchor_cov_before'], diag['trans_cov_before']]
    after = [diag['anchor_cov_after'], diag['trans_cov_after']]
    x = np.arange(len(names))
    w = 0.34
    ax_e.bar(x - w/2, before, width=w, color='#6C8EBF', label='Natural only')
    ax_e.bar(x + w/2, after, width=w, color='#72B37E', label='+ one-step branches')
    for xi, b, a in zip(x, before, after):
        ax_e.text(xi - w/2, b + 2.2, f'{b:.0f}%', ha='center', fontsize=9)
        ax_e.text(xi + w/2, a + 2.2, f'{a:.0f}%', ha='center', fontsize=9)
    ax_e.set_ylim(0, 108)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(names)
    ax_e.yaxis.set_major_formatter(PercentFormatter())
    ax_e.set_ylabel('Coverage for low-level ABG update')
    ax_e.set_title('Branch supplementation effect')
    ax_e.legend(frameon=False, loc='lower right')
    ax_e.text(0.02, 0.95,
              f"K_min={diag['k_min']}, M_target={diag['m_target']}, B_max={diag['b_max']}\nextra one-step branches={diag['extra_branches']:,}\n(after-supplementation 100% is by design)",
              transform=ax_e.transAxes, ha='left', va='top',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='lightgray'))
    add_panel_label(ax_e, 'e')

    # (f) scorer sanity check
    prog = diag['progress_scores']
    sg_types = [sg for sg in ['FindPick', 'Clean', 'Heat', 'Cool', 'Place', 'Examine'] if sg in prog and (len(prog[sg]['pos']) + len(prog[sg]['nonpos']) > 0)]
    if len(sg_types) > 0:
        x = np.arange(len(sg_types))
        w = 0.34
        pos_means = [np.mean(prog[sg]['pos']) if len(prog[sg]['pos']) else np.nan for sg in sg_types]
        neg_means = [np.mean(prog[sg]['nonpos']) if len(prog[sg]['nonpos']) else np.nan for sg in sg_types]
        pos_err = [np.std(prog[sg]['pos']) / max(np.sqrt(len(prog[sg]['pos'])), 1) if len(prog[sg]['pos']) else 0 for sg in sg_types]
        neg_err = [np.std(prog[sg]['nonpos']) / max(np.sqrt(len(prog[sg]['nonpos'])), 1) if len(prog[sg]['nonpos']) else 0 for sg in sg_types]
        ax_f.bar(x - w/2, pos_means, width=w, yerr=pos_err, color='#72B37E', capsize=3, label=r'$\Delta$rank > 0')
        ax_f.bar(x + w/2, neg_means, width=w, yerr=neg_err, color='#D97A7A', capsize=3, label=r'$\Delta$rank ≤ 0')
        ax_f.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(sg_types, rotation=20)
        ax_f.set_ylabel('Mean milestone score')
        ax_f.set_title('Scorer sanity check on progress transitions')
        ax_f.legend(frameon=False, loc='upper right')
        ax_f.text(0.98, 0.95,
                  'Intrinsic sanity check:\nlabels come from milestone-rank delta,\nnot an external annotation set.',
                  transform=ax_f.transAxes, ha='right', va='top',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='lightgray'), fontsize=8)
    else:
        ax_f.text(0.5, 0.5, 'Milestone scorer module not available', ha='center', va='center')
        ax_f.set_title('Scorer sanity check')
    add_panel_label(ax_f, 'f')

    fig.suptitle('HSPO diagnostic analysis on ALFWorld demonstrations', y=0.98, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.96], w_pad=2.2, h_pad=2.2)
    fig.savefig(output_prefix + '.png', dpi=260, bbox_inches='tight', facecolor='white')
    fig.savefig(output_prefix + '.pdf', dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_webshop_metadata(items: List[dict], output_prefix: str):
    # appendix-only helper
    cats = []
    prices_by = defaultdict(list)
    for item in items:
        cat = str(item.get('category', 'unknown')).lower().strip()
        cat = {'beauty': 'makeup', 'grocery': 'food', 'garden': 'furniture'}.get(cat, cat)
        if cat in {'makeup', 'electronics', 'fashion', 'furniture', 'food'}:
            cats.append(cat)
            p = safe_parse_price(item.get('pricing', '$0'))
            if p > 0:
                prices_by[cat].append(p)
    cat_counts = Counter(cats)
    ordered = ['fashion', 'makeup', 'electronics', 'furniture', 'food']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    vals = [cat_counts.get(c, 0) for c in ordered]
    cols = ['#C44E52', '#4C72B0', '#55A868', '#8172B2', '#CCB974']
    bars = axes[0].bar(ordered, vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width()/2, v + max(vals)*0.01, str(v), ha='center', va='bottom', fontsize=9)
    axes[0].set_title('WebShop item metadata: category counts')
    axes[0].set_ylabel('# items')
    axes[0].tick_params(axis='x', rotation=20)

    box_data = [prices_by[c] for c in ordered if len(prices_by[c])]
    box_labels = [c for c in ordered if len(prices_by[c])]
    bp = axes[1].boxplot(box_data, vert=False, patch_artist=True, labels=box_labels, showfliers=False)
    for patch, c in zip(bp['boxes'], cols[:len(box_data)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    axes[1].set_title('WebShop item metadata: price distribution')
    axes[1].set_xlabel('Price ($)')
    fig.tight_layout()
    fig.savefig(output_prefix + '.png', dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(output_prefix + '.pdf', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_webshop_items(webshop_dir: str):
    items = []
    # only load product files, not instruction dictionaries that create misleading 'unknown'
    for fname in ['items_shuffle_1000.json']:
        fpath = os.path.join(webshop_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    items.extend(data)
                elif isinstance(data, dict):
                    items.extend(list(data.values()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alfworld_raw', required=True)
    ap.add_argument('--project_root', default=None)
    ap.add_argument('--webshop_data', default=None)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--skip_webshop', action='store_true')
    ap.add_argument('--k_min', type=int, default=2)
    ap.add_argument('--m_target', type=int, default=8)
    ap.add_argument('--b_max', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.project_root:
        import sys
        sys.path.insert(0, args.project_root)

    trajs = load_alfworld_trajectories(args.alfworld_raw)
    diag = compute_diagnostics(trajs, k_min=args.k_min, m_target=args.m_target, b_max=args.b_max)
    plot_diagnostics(diag, os.path.join(args.output_dir, 'fig_hspo_diagnostics_refined'))

    # save compact summary csv-like text for captioning / sanity check
    with open(os.path.join(args.output_dir, 'fig_hspo_diagnostics_refined_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"#anchors={diag['num_anchors']}\n")
        f.write(f"singleton_pct={diag['singleton_pct']:.2f}\n")
        f.write(f"anchor_cov_before={diag['anchor_cov_before']:.2f}\n")
        f.write(f"trans_cov_before={diag['trans_cov_before']:.2f}\n")
        f.write(f"extra_branches={diag['extra_branches']}\n")

    if not args.skip_webshop and args.webshop_data:
        items = load_webshop_items(args.webshop_data)
        if items:
            plot_webshop_metadata(items, os.path.join(args.output_dir, 'fig_webshop_metadata_refined'))


if __name__ == '__main__':
    main()
