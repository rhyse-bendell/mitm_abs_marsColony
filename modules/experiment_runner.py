"""Headless runner for one experimental condition (no GUI). Used by scripts/run_experiment.py and the proof matrix.

A condition is: seed, duration, optional backend, and `experimental_settings`:
    {"constructs": {"teamwork_potential": x, "taskwork_potential": y},
     "constituent_overrides": {<any of the seven constituents>: value},
     "per_agent": {"Architect": {"constructs": {...}, "constituent_overrides": {...}}}}
Explicit constituent overrides replace gestalt-generated values (see modules/experimental_profile.py).
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from modules.simulation import SimulationState


def run_condition(
    *,
    seed: int,
    duration_s: int = 600,
    experimental_settings: Optional[Dict[str, Any]] = None,
    experiment_name: str = "experiment",
    project_root: Optional[str | Path] = None,
    brain_backend: str = "rule_brain",
    step_s: float = 1.0,
) -> Dict[str, Any]:
    """Run one seeded condition to ``duration_s`` and return a JSON-serializable summary (also written to the session)."""
    random.seed(int(seed))
    own_tmp = None
    if project_root is None:
        own_tmp = tempfile.TemporaryDirectory()
        project_root = own_tmp.name
    state = SimulationState(
        phases=None,
        project_root=str(project_root),
        flash_mode=True,
        experiment_name=experiment_name,
        seed=int(seed),
        brain_backend=brain_backend,
        experimental_settings=dict(experimental_settings or {}),
    )
    trajectory = []
    try:
        for i in range(int(duration_s / step_s)):
            state.update(step_s)
            if (i + 1) % int(100 / step_s) == 0:
                cm = state.environment.construction
                by_type = {}
                for p in cm.projects.values():
                    row = by_type.setdefault(p.get("type"), {"created": 0, "validated": 0})
                    row["created"] += 1
                    row["validated"] += int(bool(p.get("validated_complete")))
                trajectory.append({"t": round(state.time), "projects": by_type, "links": len(cm.connectors), "bridge": cm.bridges["bridge_bc"].status})
        measures = state._final_measures()
        try:
            measures["structural_support"] = state.metrics._structural_support_report()
        except Exception as exc:  # reporting must not break a run
            measures["structural_support"] = {"error": repr(exc)}
        summary = {
            "seed": int(seed),
            "duration_s": duration_s,
            "experimental_settings": dict(experimental_settings or {}),
            "manifest_condition": state._experimental_manifest(),
            "measures": measures,
            "trajectory": trajectory,
        }
        summary["session_folder"] = str(getattr(state.logger.output_session, "session_folder", ""))
    finally:
        state.stop()
        if own_tmp is not None:
            own_tmp.cleanup()
    return summary


def parse_overrides(pairs) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pair in pairs or []:
        name, _, value = str(pair).partition("=")
        if not value:
            raise ValueError(f"override must be name=value, got '{pair}'")
        out[name.strip()] = float(value)
    return out


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one experimental condition headlessly.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--teamwork", type=float, default=None, help="Teamwork Potential (0..1)")
    parser.add_argument("--taskwork", type=float, default=None, help="Taskwork Potential (0..1)")
    parser.add_argument("--set", action="append", default=[], metavar="CONSTITUENT=VALUE", help="explicit constituent override (repeatable)")
    parser.add_argument("--condition-json", default=None, help="JSON file with experimental_settings (CLI flags are applied on top)")
    parser.add_argument("--backend", default="rule_brain")
    parser.add_argument("--out", default=None, help="write the summary JSON here")
    parser.add_argument("--project-root", default=None, help="session output root (default: temporary)")
    args = parser.parse_args(argv)
    settings: Dict[str, Any] = json.loads(Path(args.condition_json).read_text(encoding="utf-8")) if args.condition_json else {}
    settings.setdefault("constructs", {})
    settings.setdefault("constituent_overrides", {})
    if args.teamwork is not None:
        settings["constructs"]["teamwork_potential"] = args.teamwork
    if args.taskwork is not None:
        settings["constructs"]["taskwork_potential"] = args.taskwork
    settings["constituent_overrides"].update(parse_overrides(args.set))
    summary = run_condition(seed=args.seed, duration_s=args.duration, experimental_settings=settings, project_root=args.project_root, brain_backend=args.backend, experiment_name="headless_condition")
    text = json.dumps(summary, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(json.dumps({"manifest_condition": summary["manifest_condition"]["per_agent"], "experimental": summary["measures"]["experimental"]["team"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())