#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.task_model import TaskModelLoader, validate_task_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate internal consistency for a task package.")
    parser.add_argument("--task-id", default="mars_colony", help="Task package id under config/tasks (default: mars_colony)")
    parser.add_argument("--config-root", default="config/tasks", help="Path to task packages root (default: config/tasks)")
    args = parser.parse_args()

    model = TaskModelLoader(config_root=args.config_root).load(task_id=args.task_id, validate=False)
    report = validate_task_model(model)

    print(f"Task package validation: {args.task_id}")
    print(f"Errors: {len(report.errors)}")
    for issue in report.errors:
        print(f"  - [{issue.code}] {issue.message}")
    print(f"Warnings: {len(report.warnings)}")
    for issue in report.warnings:
        print(f"  - [{issue.code}] {issue.message}")

    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
