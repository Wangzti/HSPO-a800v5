#!/usr/bin/env bash
# ============================================================
# HSPO 项目打包脚本
# 将代码 + SFT 数据打包，上传到 GitHub 后在新服务器上运行 SFT 预热训练。
#
# 用法:
#   bash package_for_transfer.sh
#
# 输出:
#   $PACK_DIR/
#   ├── HSPO-agent/         # 项目代码（不含 WebShop 数据）
#   │   └── data/sft/       # SFT parquet + raw trajectories
#   └── hspo_transfer.tar.gz  # 压缩包 (~50MB)
#
# 目标服务器操作:
#   tar -xzf hspo_transfer.tar.gz
#   cd HSPO-agent
#   pip install -e .
#   # 确保 ALFWorld 环境已安装 (pip install alfworld textworld)
#   # 确保 Qwen2.5-0.5B-Instruct 模型已下载或可访问
#   bash example_scripts/SFT_warmup/run_sft_pipeline.sh 0.5b
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config/paths.sh"

PACK_DIR="${PACK_DIR:-$HSPO_DATA_ROOT/../hspo_transfer}"
SFT_DATA="$HSPO_DATA_ROOT/sft"

echo "=== HSPO 项目打包 ==="
echo "源目录: $SCRIPT_DIR"
echo "输出目录: $PACK_DIR"

# Clean previous
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR/HSPO-agent"

# ── 1. 拷贝项目代码（排除不需要的大文件） ──────────────────────────
echo ""
echo "[1/4] 拷贝项目代码..."

rsync -a \
    --exclude='agent_system/environments/env_package/webshop' \
    --exclude='agent_system/environments/env_package/gym_cards' \
    --exclude='agent_system/environments/env_package/sokoban' \
    --exclude='agent_system/environments/env_package/appworld' \
    --exclude='agent_system/environments/env_package/search' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.claude' \
    --exclude='wandb' \
    --exclude='outputs' \
    --exclude='trajectories' \
    --exclude='verl' \
    --exclude='.git' \
    --exclude='index.html.*' \
    "$SCRIPT_DIR/" "$PACK_DIR/HSPO-agent/"

# ── 2. 拷贝 SFT parquet 数据到项目内的 data/sft/ ─────────────────
echo "[2/4] 拷贝 SFT parquet 数据..."
mkdir -p "$PACK_DIR/HSPO-agent/data/sft"
cp -r "$SFT_DATA/alfworld_sft" "$PACK_DIR/HSPO-agent/data/sft/"

# ── 3. 拷贝 raw 轨迹（可选，用于重新分段） ──────────────────────
echo "[3/4] 拷贝 raw trajectories..."
cp -r "$SFT_DATA/alfworld_raw" "$PACK_DIR/HSPO-agent/data/sft/"

# ── 4. 修改 SFT pipeline 脚本中数据路径指向项目内 ────────────────
echo "[4/4] 修正 SFT pipeline 数据路径..."
SED_EXPR='s|DATA_DIR="$HSPO_SFT_DATA_DIR"|DATA_DIR="$SCRIPT_DIR/data/sft/alfworld_sft"|'
sed -i "$SED_EXPR" "$PACK_DIR/HSPO-agent/example_scripts/SFT_warmup/run_sft_pipeline.sh"

# 目标服务器用户只需编辑 config/paths.sh 中的 HSPO_MODEL_ROOT 等变量即可

# ── 生成 README ──────────────────────────────────────────────────
cat > "$PACK_DIR/HSPO-agent/README_TRANSFER.md" << 'READMEEOF'
# HSPO Transfer 包使用说明

本包包含 HSPO 项目代码 + ALFWorld SFT 数据，可直接在新服务器上运行 SFT 预热训练。

## 前置条件

```bash
conda create -n verl python=3.12
conda activate verl
pip install alfworld textworld
pip install -e .
pip install verl  # 或从源码安装 veRL
```

## 修改路径配置

编辑 **`config/paths.sh`**（这是唯一需要修改的路径文件），修改以下变量：

```bash
HSPO_PROJECT_ROOT=/your/path/to/HSPO        # 项目根目录
HSPO_DATA_ROOT=/your/path/to/data           # 数据目录
HSPO_CKPT_ROOT=/your/path/to/checkpoints    # checkpoint 目录
HSPO_MODEL_ROOT=/your/path/to/models        # 模型目录
HSPO_OUTPUT_ROOT=/your/path/to/outputs      # 输出目录
```

如使用 HuggingFace Hub 下载模型（而非本地路径），修改 `config/paths.sh` 中：
```bash
export HSPO_MODEL_05B="Qwen/Qwen2.5-0.5B-Instruct"
```

## 运行 SFT 预热训练

```bash
source config/paths.sh
export ALFWORLD_DATA=~/.cache/alfworld
export HF_HUB_OFFLINE=0   # 新服务器上首次需要下载模型
bash example_scripts/SFT_warmup/run_sft_pipeline.sh 0.5b
```

## 数据说明

- `data/sft/alfworld_sft/` — 已生成的 SFT parquet 数据（4 种 split）
- `data/sft/alfworld_raw/` — 原始 expert 轨迹 JSON（如需重新分段）

## Wandb

如需 wandb 记录，先 `wandb login` 或设置 `WANDB_API_KEY` 环境变量。
READMEEOF

# ── 打包 ────────────────────────────────────────────────────────
echo ""
echo "打包为 hspo_transfer.tar.gz ..."
cd "$PACK_DIR"
tar -czf hspo_transfer.tar.gz HSPO-agent/

echo ""
echo "============================================================"
echo "  打包完成!"
echo "  位置: $PACK_DIR/"
echo "  大小: $(du -sh hspo_transfer.tar.gz | cut -f1)"
echo "============================================================"
echo ""
echo "传输到目标服务器:"
echo "  scp $PACK_DIR/hspo_transfer.tar.gz user@server:/path/"
echo ""
echo "目标服务器上解压:"
echo "  tar -xzf hspo_transfer.tar.gz"
echo "  cd HSPO-agent"
echo "  # 修改 model path + 安装依赖后:"
echo "  bash example_scripts/SFT_warmup/run_sft_pipeline.sh 0.5b"
