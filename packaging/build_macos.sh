#!/usr/bin/env bash
# Build the standalone macOS app: dist/FuzzFace.app
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-build.txt

pyinstaller --noconfirm --clean --windowed \
  --name FuzzFace \
  --collect-all soundfile \
  --exclude-module matplotlib \
  gui.py

echo
echo "Built: dist/FuzzFace.app"
echo "Run with:  open dist/FuzzFace.app"
