#!/usr/bin/env bash
# ============================================================
# HSPO Setup Check — verify all paths and dependencies
# Usage: bash scripts/check_setup.sh [1.5b|7b]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSPO_AGENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$HSPO_AGENT_ROOT/config/paths.sh"

MODEL_SIZE="${1:-1.5b}"
PASS=0
FAIL=0

check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  [OK] $desc"
        ((PASS++))
    else
        echo "  [FAIL] $desc"
        ((FAIL++))
    fi
}

check_file() {
    local desc="$1" path="$2"
    if [ -f "$path" ]; then
        echo "  [OK] $desc : $path"
        ((PASS++))
    elif [ -d "$path" ]; then
        echo "  [OK] $desc (dir): $path"
        ((PASS++))
    else
        echo "  [FAIL] $desc MISSING: $path"
        ((FAIL++))
    fi
}

echo "=== HSPO Setup Check ==="
echo "  HSPO_AGENT_ROOT:  $HSPO_AGENT_ROOT"
echo "  HIPER_AGENT_ROOT: $HIPER_AGENT_ROOT"
echo "  HSPO_DATA_ROOT:   $HSPO_DATA_ROOT"
echo "  HSPO_CKPT_ROOT:   $HSPO_CKPT_ROOT"
echo "  HSPO_MODEL_ROOT:  $HSPO_MODEL_ROOT"
echo "  MODEL_SIZE:       $MODEL_SIZE"
echo ""

# ── 1. Directories ──────────────────────────────────────────
echo "--- Required Directories ---"
check_file "HSPO-agent"            "$HSPO_AGENT_ROOT"
check_file "HiPER-agent"           "$HIPER_AGENT_ROOT"
check_file "SFT data"              "$HSPO_SFT_DATA_DIR"
check_file "SFT raw"               "$HSPO_SFT_RAW_DIR"

# ── 2. HiPER-agent critical files ────────────────────────────
echo ""
echo "--- HiPER-agent Critical Files ---"
check_file "main_ppo_hspo.py"      "$HIPER_AGENT_ROOT/verl/trainer/main_ppo_hspo.py"
check_file "ray_trainer.py"        "$HIPER_AGENT_ROOT/verl/trainer/ppo/ray_trainer.py"
check_file "dp_actor.py"           "$HIPER_AGENT_ROOT/verl/workers/actor/dp_actor.py"
check_file "ALFWorld env"          "$HIPER_AGENT_ROOT/agent_system/environments/env_package/alfworld"

# ── 3. Model ─────────────────────────────────────────────────
echo ""
echo "--- Model ---"
BASE_MODEL=$(hspo_model_path "$MODEL_SIZE")
check_file "Base model ($MODEL_SIZE)" "$BASE_MODEL"

# ── 4. SFT Checkpoint ────────────────────────────────────────
echo ""
echo "--- SFT Checkpoint ---"
SFT_CKPT=$(hspo_sft_ckpt_dir "$MODEL_SIZE")
if [ -f "$SFT_CKPT/tokenizer_config.json" ] || [ -f "$SFT_CKPT/tokenizer.json" ]; then
    echo "  [OK] SFT ckpt: $SFT_CKPT"
    ((PASS++))
elif [ -L "$SFT_CKPT" ] && [ -d "$SFT_CKPT" ]; then
    TARGET=$(readlink -f "$SFT_CKPT")
    if [ -f "$TARGET/tokenizer_config.json" ]; then
        echo "  [OK] SFT ckpt (symlink): $SFT_CKPT -> $TARGET"
        ((PASS++))
    else
        echo "  [FAIL] SFT symlink target has no tokenizer: $TARGET"
        ((FAIL++))
    fi
else
    echo "  [FAIL] SFT ckpt not found: $SFT_CKPT"
    echo "         Run run_sft_pipeline.sh first."
    ((FAIL++))
fi

# ── 5. RL Data ───────────────────────────────────────────────
echo ""
echo "--- RL Training Data ---"
check_file "RL train data" "$HSPO_RL_TRAIN_DATA"
check_file "RL val data"   "$HSPO_RL_VAL_DATA"

# ── 6. Python env ────────────────────────────────────────────
echo ""
echo "--- Python Environment ---"
check "python3 available"     which python3
check "hspo package"          python3 -c "import hspo"
check "verl package"          python3 -c "import verl"
check "ALFWorld installed"    python3 -c "import alfworld"
check "torch available"       python3 -c "import torch; print(torch.cuda.is_available())"

# ── 7. HSPO sanity tests ─────────────────────────────────────
echo ""
echo "--- HSPO Sanity Tests ---"
if python3 -m pytest "$HSPO_AGENT_ROOT/tests/sanity/" -q 2>&1; then
    echo "  [OK] HSPO tests pass"
    ((PASS++))
else
    echo "  [FAIL] HSPO tests failed"
    ((FAIL++))
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "==============================="
echo "  Result: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo "  Fix the FAIL items above before running training."
    exit 1
else
    echo "  All checks passed! Ready to run:"
    echo ""
    echo "    # Phase 5: Low-level executor training"
    echo "    bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh low_level"
    echo ""
    echo "    # Phase 6: High-level planner training"
    echo "    bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh high_level"
    echo ""
    echo "    # Phase 7: Joint training"
    echo "    bash example_scripts/HSPO_trainer/run_alfworld_hspo.sh joint"
fi
