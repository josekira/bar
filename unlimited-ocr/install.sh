#!/usr/bin/env bash
# Install Unlimited-OCR 2 (baidu/Unlimited-OCR) with its SGLang server.
# Needs: Linux, an NVIDIA GPU with CUDA drivers (Baidu tested on an 80 GB A800),
#        git, git-lfs and uv (https://docs.astral.sh/uv/).
# Usage: bash install.sh            (installs into ./Unlimited-OCR and ./.venv next to this script)
set -euo pipefail
cd "$(dirname "$0")"

for tool in git git-lfs uv; do
  command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }
done
command -v nvidia-smi >/dev/null || echo "WARNING: nvidia-smi not found; the server needs an NVIDIA GPU."

# 1. Get the code (the custom SGLang wheel is stored with Git LFS).
if [ ! -d Unlimited-OCR ]; then
  git lfs install --skip-repo
  git clone https://github.com/baidu/Unlimited-OCR.git
fi
(cd Unlimited-OCR && git lfs pull)
WHEEL=$(ls Unlimited-OCR/wheel/sglang-*.whl)
if [ "$(stat -c %s "$WHEEL")" -lt 1000000 ]; then
  echo "The SGLang wheel is still an LFS pointer; run 'git lfs pull' inside Unlimited-OCR." >&2
  exit 1
fi

# 2. Python 3.10 environment with the customized SGLang build.
[ -d .venv ] || uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install "$WHEEL" pymupdf==1.27.2.2 fastokens==0.3.2 requests

echo
echo "Installed. Next:"
echo "  bash start_server.sh                     # terminal 1 (downloads the model on first run)"
echo "  .venv/bin/python ocr.py my.pdf -o my.md  # terminal 2"
