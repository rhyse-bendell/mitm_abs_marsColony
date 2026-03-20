#!/usr/bin/env python3
"""Environment preflight and optional repair flow for Mars Colony GUI launch."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

MIN_PYTHON = (3, 9)
REQUIRED_MODULES = ("tkinter", "numpy", "matplotlib", "pathfinding")
OPTIONAL_QT_BINDINGS = ("PySide6", "PyQt6", "PyQt5", "PySide2")
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


@dataclass
class CheckMessage:
    level: str
    message: str
    repairable: bool = False


@dataclass
class PreflightReport:
    messages: List[CheckMessage]

    @property
    def has_errors(self) -> bool:
        return any(msg.level == "error" for msg in self.messages)

    @property
    def has_repairable_issue(self) -> bool:
        return any(msg.repairable for msg in self.messages)


def _safe_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - behavior validated through check_environment
        return None, exc


def check_environment() -> PreflightReport:
    messages: List[CheckMessage] = []

    if sys.version_info < MIN_PYTHON:
        messages.append(
            CheckMessage(
                "error",
                f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (found {sys.version.split()[0]}).",
                repairable=False,
            )
        )
    else:
        messages.append(CheckMessage("ok", f"Python version is {sys.version.split()[0]}."))

    loaded_modules = {}
    for module_name in REQUIRED_MODULES:
        module, exc = _safe_import(module_name)
        if exc is not None:
            messages.append(
                CheckMessage(
                    "error",
                    f"Missing or broken module '{module_name}': {exc}",
                    repairable=True,
                )
            )
        else:
            loaded_modules[module_name] = module
            messages.append(CheckMessage("ok", f"Module import succeeded: {module_name}."))

    matplotlib_mod = loaded_modules.get("matplotlib")
    if matplotlib_mod is not None:
        forced_backend = (os.environ.get("MPLBACKEND") or "").strip().lower()
        if forced_backend and ("qt" in forced_backend or "pyside" in forced_backend or "pyqt" in forced_backend):
            messages.append(
                CheckMessage(
                    "error",
                    f"MPLBACKEND is forcing '{forced_backend}', which conflicts with this Tk GUI.",
                    repairable=False,
                )
            )

    installed_qt = []
    for qt_name in OPTIONAL_QT_BINDINGS:
        qt_mod, _ = _safe_import(qt_name)
        if qt_mod is not None:
            installed_qt.append(qt_name)
    if installed_qt:
        messages.append(
            CheckMessage(
                "warning",
                f"Qt bindings installed ({', '.join(installed_qt)}); avoid forcing Qt backends for this Tk app.",
                repairable=False,
            )
        )

    numpy_mod = loaded_modules.get("numpy")
    if numpy_mod is not None:
        version_text = str(getattr(numpy_mod, "__version__", "unknown"))
        major = version_text.split(".", 1)[0]
        if major.isdigit() and int(major) >= 2:
            messages.append(
                CheckMessage(
                    "warning",
                    "NumPy 2.x detected; mismatched compiled wheels can cause import/runtime failures.",
                    repairable=True,
                )
            )

    if not REQUIREMENTS_PATH.exists():
        messages.append(CheckMessage("error", f"Missing dependency file: {REQUIREMENTS_PATH}", repairable=False))
    else:
        messages.append(CheckMessage("ok", f"Dependency file found: {REQUIREMENTS_PATH.name}."))

    return PreflightReport(messages)


def build_repair_command(python_executable: str = sys.executable, requirements_path: Path = REQUIREMENTS_PATH) -> List[str]:
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(requirements_path),
    ]


def run_repair(python_executable: str = sys.executable) -> int:
    if not REQUIREMENTS_PATH.exists():
        print(f"Cannot repair: missing {REQUIREMENTS_PATH}")
        return 2
    command = build_repair_command(python_executable=python_executable, requirements_path=REQUIREMENTS_PATH)
    print("Running repair command:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_fresh_preflight_check(python_executable: str = sys.executable) -> int:
    command = [python_executable, str(Path(__file__).resolve())]
    print("Re-running preflight checks in a fresh Python process:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _print_report(report: PreflightReport) -> None:
    prefix_map = {"ok": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}
    print("Mars Colony preflight report")
    for msg in report.messages:
        print(f"{prefix_map.get(msg.level, '[INFO]')} {msg.message}")

    if report.has_errors:
        print("\nPreflight status: FAILED")
    elif any(msg.level == "warning" for msg in report.messages):
        print("\nPreflight status: PASSED with warnings")
    else:
        print("\nPreflight status: PASSED")

    print("\nNotes:")
    print("- This GUI is Tk-based and does not require PySide/Qt.")
    print("- If you see NumPy/Matplotlib binary mismatch errors, run '--repair'.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Mars Colony runtime prerequisites.")
    parser.add_argument("--repair", action="store_true", help="Attempt controlled pip repair using requirements.txt")
    args = parser.parse_args(argv)

    report = check_environment()
    _print_report(report)

    if args.repair:
        rc = run_repair()
        if rc != 0:
            return rc
        return run_fresh_preflight_check()

    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
