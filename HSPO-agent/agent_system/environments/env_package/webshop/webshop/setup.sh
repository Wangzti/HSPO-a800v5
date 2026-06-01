#!/bin/bash

# Repo-relative dir for code imports (web_agent_site/utils.py expects ../data/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Large assets go on data disk; override with WEBSHOP_DATA_ROOT=/your/path
# Default: /root/autodl-tmp/data/webshop (under /root/autodl-tmp/data)
WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-/root/autodl-tmp/data/webshop}"

# Prefer conda env Python if conda is installed but not initialized in this shell
for _conda in "${HOME}/miniconda3/etc/profile.d/conda.sh" "/root/miniconda3/etc/profile.d/conda.sh"; do
  if [ -f "$_conda" ]; then
    # shellcheck source=/dev/null
    . "$_conda"
    break
  fi
done
# Run after: conda activate your-env  OR  export PYTHON_CMD=/path/to/python
PYTHON_CMD="${PYTHON_CMD:-$(command -v python3 || command -v python || true)}"
if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
  echo "[ERROR] No python found. Activate env first, e.g.: conda activate verl-agent-webshop"
  exit 1
fi
export PATH="$(dirname "$(readlink -f "$PYTHON_CMD" 2>/dev/null || echo "$PYTHON_CMD")"):${PATH}"

pip_install_requirements() {
  local req="$SCRIPT_DIR/requirements.txt"
  if "$PYTHON_CMD" -m pip install -r "$req"; then
    return 0
  fi
  echo "[WARN] pip install failed (mirror/index). Retrying with PyPI root..."
  PIP_INDEX_URL=https://pypi.org/simple "$PYTHON_CMD" -m pip install -r "$req"
}

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python Dependencies (use same interpreter as this shell / conda env)
if ! pip_install_requirements; then
  echo "[ERROR] pip install failed. Retry manually:"
  echo "  PIP_INDEX_URL=https://pypi.org/simple $PYTHON_CMD -m pip install -r \"$SCRIPT_DIR/requirements.txt\""
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  conda install -y mkl
  conda install -y -c conda-forge faiss-cpu
  conda install -y -c conda-forge openjdk=11
else
  echo "[WARN] conda not on PATH — skipped mkl, faiss-cpu, openjdk. Pyserini needs JDK 11 (JAVA_HOME)."
  echo "       Fix: source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>, then re-run this script."
fi

# gdown via module so no standalone executable is required
"$PYTHON_CMD" -m pip install -q gdown

# Download dataset into WEBSHOP_DATA_ROOT, then link ./data -> there (keeps ../data paths valid)
mkdir -p "$WEBSHOP_DATA_ROOT"
cd "$WEBSHOP_DATA_ROOT" || exit 1
if [ "$data" == "small" ]; then
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib # items_shuffle_1000
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu # items_ins_v2_1000
elif [ "$data" == "all" ]; then
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB # items_shuffle
  "$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi # items_ins_v2
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
"$PYTHON_CMD" -m gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd "$SCRIPT_DIR" || exit 1
if [ -e "$SCRIPT_DIR/data" ] && [ ! -L "$SCRIPT_DIR/data" ]; then
  echo "[WARN] $SCRIPT_DIR/data exists and is not a symlink; renaming to data.bak.$$"
  mv "$SCRIPT_DIR/data" "$SCRIPT_DIR/data.bak.$$"
fi
ln -sfn "$WEBSHOP_DATA_ROOT" "$SCRIPT_DIR/data"
echo "[INFO] Dataset stored under $WEBSHOP_DATA_ROOT ; ./data -> symlink"

# Download spaCy large NLP model
"$PYTHON_CMD" -m spacy download en_core_web_lg
"$PYTHON_CMD" -m spacy download en_core_web_sm

# Build search engine index
cd "$SCRIPT_DIR/search_engine" || exit 1
mkdir -p resources resources_100 resources_1k resources_100k
"$PYTHON_CMD" convert_product_file_format.py
mkdir -p indexes
bash ./run_indexing.sh
cd "$SCRIPT_DIR" || exit 1

# Create logging folder + samples of log data
# get_human_trajs () {
#   PYCMD=$(cat <<EOF
# import gdown
# url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
# gdown.download_folder(url, quiet=True, remaining_ok=True)
# EOF
#   )
#   python -c "$PYCMD"
# }
# mkdir -p user_session_logs/
# cd user_session_logs/
# echo "Downloading 50 example human trajectories..."
# get_human_trajs
# echo "Downloading example trajectories complete"
# cd ..