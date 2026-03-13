#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' was not found. Install Python 3 and try again."
  exit 1
fi

"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || {
import importlib
for module in ("tkinter", "matplotlib"):
    importlib.import_module(module)
PY
  echo "Error: required packages are missing (tkinter and/or matplotlib)."
  echo "Install dependencies, for example: pip install matplotlib"
  exit 1
}

echo "Launching Mars Colony interface..."
exec "$PYTHON_BIN" interface.py
