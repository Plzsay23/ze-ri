#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

grep -RIn \
  --binary-files=without-match \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=build \
  --exclude-dir=install \
  --exclude-dir=log \
  --exclude-dir=__pycache__ \
  --exclude-dir=.pytest_cache \
  --exclude-dir=target \
  --exclude='find_local_absolute_paths.sh' \
  --exclude='*.pyc' \
  --exclude='*.onnx' \
  --exclude='*.pgm' \
  --exclude='*.rlib' \
  --exclude='*.rmeta' \
  --exclude='*.bak' \
  --exclude='*.bak_*' \
  "/home/hansungai\|~/NBYtics\|~/tools\|~/venv/ze-ri" \
  NBYtics tools src dashboard scripts source_zeri_vlm.sh source_zeri_nbytics.sh 2>/dev/null || true
