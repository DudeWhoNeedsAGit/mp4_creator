#!/usr/bin/env bash
set -e

echo "== MP4 Karaoke Environment Setup =="

# System deps
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libmp3lame0 \
    fonts-dejavu-core \
    libasound2-dev

# Create venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade tooling
pip install --upgrade pip setuptools wheel

# Install CUDA torch FIRST (important)
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
pip install -r requirements.txt

# Sanity check
python - <<'EOF'
import torch, numpy
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF

echo "== Setup complete =="
