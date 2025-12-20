#!/usr/bin/env bash
set -e

echo "== Aeneas Alignment Environment Setup =="

# System dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev build-essential ffmpeg libxml2-dev libxslt1-dev

# Create virtual environment
python3.10 -m venv venv-aeneas
source venv-aeneas/bin/activate

# Upgrade core Python packaging tools
pip install --upgrade pip "setuptools<60" "wheel<0.38"


# Install numpy (version compatible with aeneas)
pip install "numpy<2"

# Install aeneas without build isolation to use our setuptools/wheel versions
pip install aeneas --no-build-isolation

# Sanity check
python - <<'EOF'
from aeneas.executetask import ExecuteTask
print("Aeneas OK")
EOF

echo "== Aeneas setup complete =="
