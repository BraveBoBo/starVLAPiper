#!/usr/bin/env bash
set -euo pipefail

# Use Hugging Face mirror
# export HF_ENDPOINT="https://hf-mirror.com"

# Avoid deprecated hf_transfer
unset HF_HUB_ENABLE_HF_TRANSFER

# More stable for hf-mirror large files
export HF_HUB_DISABLE_XET=1

# Longer timeout for slow network
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120

# Put cache on /data, not system disk
export HF_HOME=/data/huggingface
export HF_HUB_CACHE=/data/huggingface/hub

DEST=/data/datasets
mkdir -p "$DEST" "$HF_HOME" "$HF_HUB_CACHE"

# Install hf CLI if missing
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install -U "huggingface_hub[cli]"
fi

dl() {
  local repo="$1"
  local dir="$2"
  shift 2

  mkdir -p "$dir"

  for i in $(seq 1 50); do
    echo "==== Downloading: $repo -> $dir | try $i/50 ===="

    if command -v hf >/dev/null 2>&1; then
      if hf download "$repo" --local-dir "$dir" "$@"; then
        return 0
      fi
    else
      if huggingface-cli download "$repo" --local-dir "$dir" "$@"; then
        return 0
      fi
    fi

    echo "[retry $i] $repo failed, retry after 10s..."
    sleep 10
  done

  echo "FAILED: $repo"
  return 1
}

# 1. Download ALL files in model repo
# dl StarVLA/Qwen3VL-PI_v3-Bridge-RT_1 \
#   "$DEST/Qwen3VL-PI_v3-Bridge-RT_1"

# # 2. Download ALL Bridge dataset
# dl IPEC-COMMUNITY/bridge_orig_lerobot \
#   "$DEST/bridge_orig_lerobot" \
#   --repo-type dataset

# 3. Download ALL Fractal dataset
dl IPEC-COMMUNITY/fractal20220817_data_lerobot \
  "$DEST/fractal20220817_data_lerobot" \
  --repo-type dataset

echo "==== ALL DONE ==== Files are in $DEST"