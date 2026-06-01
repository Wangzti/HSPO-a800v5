#!/usr/bin/env bash
# ============================================================
# HSPO SFT Warm-up for ALFWorld — Qwen2.5-1.5B-Instruct
# Shortcut: run the full 4-step SFT pipeline
# ============================================================
set -x
cd "$(dirname "$0")"
bash run_sft_pipeline.sh 1.5b "$@"
