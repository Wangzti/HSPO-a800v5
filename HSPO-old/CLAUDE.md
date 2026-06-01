# HSPO-agent CLAUDE.md

## Project Overview

HSPO (Hierarchical Sub-goal conditioned Process Optimization) is a hierarchical RL framework for LLM agents. Built on top of HiPER-agent (at `../HiPER-agent/`) and verl.

**Core innovation**: Token-level credit assignment separating planner (subgoal tokens) and executor (action tokens) with a rule-based PRM for dense low-level reward.

## Directory Structure

```
HSPO-agent/
├── hspo/                          # HSPO core package (NEW – main contribution)
│   ├── config.py                  # HSPOConfig dataclass
│   ├── types.py                   # StepRecord, SegmentRecord, MacroTransition
│   ├── parser.py                  # PlanExecuteParser (<switch>/<subgoal>/<action>)
│   ├── token_mask.py              # TokenMaskBuilder (per-token span masks)
│   ├── advantages.py              # compute_process_return, compute_macro_gae
│   └── prm/
│       ├── base.py                # PRMBase, PRMOutput
│       └── alfworld_prm.py        # AlfworldRulePRM (6 task types, rule-based)
├── agent_system/
│   ├── environments/              # COPIED from HiPER-agent (unmodified)
│   ├── multi_turn_rollout/
│   │   ├── rollout_loop_hspo.py   # HSPO rollout (PRM + token masks + λ-return)
│   │   └── utils.py               # COPIED from HiPER-agent
│   └── reward_manager/
│       ├── hspo_reward_manager.py # HSPO reward placement (low + macro)
│       └── __init__.py            # Exposes HSPORewardManager
├── verl/                          # SYMLINK → ../HiPER-agent/verl/
│   └── trainer/
│       └── main_ppo_hspo.py       # HSPO training entry point
├── example_scripts/
│   ├── HSPO_trainer/
│   │   └── run_alfworld_hspo.sh   # Main training script
│   ├── SFT_warmup/                # COPIED from HiPER-agent
│   └── data_preprocess/           # COPIED from HiPER-agent
└── tests/sanity/                  # Unit tests (34 tests, all green)
    ├── test_hspo_parser.py
    ├── test_hspo_prm.py
    └── test_hspo_advantages.py
```

## Key Design Decisions

1. **Token-level masks**: `switch_mask`, `subgoal_mask`, `action_mask` are non-overlapping. During training, A_H × subgoal_mask (planner PPO), A_L × action_mask (executor PPO), CE × switch_mask (SWITCH supervision).

2. **No low-level critic**: λ-return `A_t^L = Σ (γλ)^{t-l} r_l^L` replaces GAE. Batch-normalised within segment.

3. **Two-phase training**: `phase=low_level` trains executor only; `phase=high_level` trains planner only with macro-PPO + critic V_H; `phase=joint` alternates.

4. **Rule-PRM**: `AlfworldRulePRM` covers all 6 ALFWorld task types. No neural PRM in phase 0–2.

## Training Phases

```
Phase low_level  → run_alfworld_hspo.sh low_level
Phase high_level → run_alfworld_hspo.sh high_level
Phase joint      → run_alfworld_hspo.sh joint
```

Prerequisites: SFT checkpoint at `/mnt/nfs/ztwang/checkpoints/hspo/sft/alfworld_qwen2.5_0.5b_instruct`.

## Tests

```bash
cd /mnt/nfs/ztwang/projects/demos/HSPO/HSPO-agent
/mnt/nfs/ztwang/conda_envs/verl-webshop/bin/pytest tests/sanity/ -v
```

## Conda Envs

- `verl` (Python 3.12) – ALFWorld RL training
- `verl-webshop` (Python 3.10) – WebShop / unit tests

## Collaboration Protocol

- `hspo/` is the write zone for HSPO-specific logic.
- `verl/` is a symlink to HiPER-agent – do not modify it here.
- `agent_system/environments/` is a copy of HiPER's – do not modify unless needed.
- All new subgoal types should be added to `hspo/prm/alfworld_prm.py:SUBGOAL_TYPES` and `_progress` dispatcher.
- **Session log**: At the end of every session, append a summary of actions, results, blockers, and next steps to `docs/progress_log.md`.
