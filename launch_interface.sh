#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' was not found. Install Python 3 and try again."
  exit 1
fi

if ! "$PYTHON_BIN" scripts/preflight_check.py; then
  echo
  echo "Preflight failed. Run repair with:"
  echo "  $PYTHON_BIN scripts/preflight_check.py --repair"
  exit 1
fi

export MPLBACKEND=TkAgg

echo "Launching Mars Colony interface..."
exec "$PYTHON_BIN" interface.py
