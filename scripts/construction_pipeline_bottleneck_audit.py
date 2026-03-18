#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.phase_definitions import MISSION_PHASES
from modules.simulation import SimulationState
from modules.task_model import load_task_model

CONDITIONS = {
    "high_task_high_team": {"taskwork_potential": 0.9, "teamwork_potential": 0.9},
    "high_task_low_team": {"taskwork_potential": 0.9, "teamwork_potential": 0.1},
    "low_task_high_team": {"taskwork_potential": 0.1, "teamwork_potential": 0.9},
    "low_task_low_team": {"taskwork_potential": 0.1, "teamwork_potential": 0.1},
}

MILESTONES = [
    "created",
    "any_progress",
    "resource_complete",
    "ready_for_validation",
    "validate_attempted",
    "validated_complete",
    "needs_repair",
    "repair_attempted",
    "repair_succeeded",
]


def _build_agent_configs(task, taskwork: float, teamwork: float):
    configs = []
    for default in task.agent_defaults:
        configs.append(
            {
                "name": default.agent_name,
                "display_name": default.display_name or default.agent_name,
                "agent_id": default.agent_id or default.agent_name,
                "role": default.role_id,
                "label": default.agent_label or default.role_id,
                "template_id": default.template_id,
                "constructs": {
                    "taskwork_potential": float(taskwork),
                    "teamwork_potential": float(teamwork),
                },
                "mechanism_overrides": dict(default.mechanism_overrides or {}),
                "packet_access": list(default.source_access_override or []),
                "initial_goal_seeds": list(default.initial_goal_seeds or []),
                "communication_params": dict(default.communication_params or {}),
                "planner_config": dict(default.planner_config or {}),
                "brain_config": {"backend": "rule_brain"},
            }
        )
    return configs


def _event_rows(session_folder: Path):
    event_path = session_folder / "logs" / "events.csv"
    rows = []
    with event_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = {}
            try:
                payload = json.loads(row.get("payload") or "{}")
            except json.JSONDecodeError:
                payload = {}
            rows.append(
                {
                    "time": float(row.get("time") or 0.0),
                    "event_type": row.get("event_type") or "",
                    "payload": payload,
                }
            )
    return rows


def _blank_type_counter():
    return {t: {m: 0 for m in MILESTONES} for t in ["house", "greenhouse", "water_generator"]}


def _blank_first_times():
    return {t: {m: None for m in MILESTONES} for t in ["house", "greenhouse", "water_generator"]}


def _set_first(first_map: dict, structure_type: str, milestone: str, t: float):
    cur = first_map[structure_type][milestone]
    if cur is None or t < cur:
        first_map[structure_type][milestone] = t


def _run_once(task, condition_name: str, profile: dict, steps: int, dt: float, seed: int, project_root: Path):
    random.seed(seed)
    sim = SimulationState(
        agent_configs=_build_agent_configs(task, profile["taskwork_potential"], profile["teamwork_potential"]),
        phases=MISSION_PHASES,
        flash_mode=True,
        experiment_name=f"pipeline_{condition_name}",
        project_root=project_root,
        brain_backend="rule_brain",
    )
    for _ in range(steps):
        sim.update(dt)
    sim.stop()

    session = sim.logger.output_session.session_folder
    run_summary = json.loads((session / "measures" / "run_summary.json").read_text(encoding="utf-8"))
    events = _event_rows(session)

    project_type_by_id = {
        p.get("id"): p.get("type", "unknown")
        for p in sim.environment.construction.projects.values()
        if isinstance(p, dict)
    }

    counts = _blank_type_counter()
    first_times = _blank_first_times()

    for project in sim.environment.construction.projects.values():
        ptype = project.get("type", "unknown")
        if ptype not in counts:
            continue
        counts[ptype]["created"] += 1
        _set_first(first_times, ptype, "created", 0.0)
        delivered = int(project.get("delivered_resources", {}).get("bricks", 0) or 0)
        if delivered > 0:
            counts[ptype]["any_progress"] += 1
        if project.get("resource_complete", False):
            counts[ptype]["resource_complete"] += 1
            counts[ptype]["ready_for_validation"] += 1
        if project.get("validated_complete", False):
            counts[ptype]["validated_complete"] += 1

    validate_selected_count = 0
    repair_selected_count = 0
    validate_executed_count = 0
    repair_executed_count = 0
    selected_action_counts = Counter()

    for e in events:
        etype = e["event_type"]
        payload = e["payload"]
        t = e["time"]
        project_id = payload.get("project_id")
        stype = payload.get("structure_type") or project_type_by_id.get(project_id)

        if etype == "brain_decision_outcome":
            selected = payload.get("selected_action")
            if selected:
                selected_action_counts[str(selected)] += 1
            if selected == "validate_construction":
                validate_selected_count += 1
            if selected == "repair_or_correct_construction":
                repair_selected_count += 1

        if stype not in counts:
            continue

        if etype == "construction_progress_updated":
            _set_first(first_times, stype, "any_progress", t)
        if etype == "construction_ready_for_validation":
            _set_first(first_times, stype, "resource_complete", t)
            _set_first(first_times, stype, "ready_for_validation", t)
        if etype in {"construction_validated_correct", "construction_validated_incorrect"}:
            counts[stype]["validate_attempted"] += 1
            validate_executed_count += 1
            _set_first(first_times, stype, "validate_attempted", t)
        if etype == "construction_validated_correct":
            _set_first(first_times, stype, "validated_complete", t)
        if etype == "construction_validated_incorrect":
            counts[stype]["needs_repair"] += 1
            _set_first(first_times, stype, "needs_repair", t)
        if etype == "construction_build_episode" and payload.get("decision_action") == "repair_or_correct_construction":
            counts[stype]["repair_attempted"] += 1
            repair_executed_count += 1
            _set_first(first_times, stype, "repair_attempted", t)
        if etype == "construction_repair_episode":
            counts[stype]["repair_succeeded"] += 1
            _set_first(first_times, stype, "repair_succeeded", t)

    support_proxy = run_summary.get("outcomes", {}).get("colony_support_capacity_proxy", {})
    validated_mix = support_proxy.get("validated_structures_used", {})

    return {
        "condition": condition_name,
        "seed": seed,
        "steps": steps,
        "dt": dt,
        "counts_by_type": counts,
        "first_times_by_type": first_times,
        "validate_selected_count": validate_selected_count,
        "validate_executed_count": validate_executed_count,
        "repair_selected_count": repair_selected_count,
        "repair_executed_count": repair_executed_count,
        "selected_action_counts": dict(selected_action_counts),
        "final_project_states": {
            p.get("id"): {
                "type": p.get("type"),
                "status": p.get("status"),
                "resource_complete": bool(p.get("resource_complete", False)),
                "validated_complete": bool(p.get("validated_complete", False)),
                "correct": bool(p.get("correct", True)),
                "delivered": int(p.get("delivered_resources", {}).get("bricks", 0) or 0),
                "required": int(p.get("required_resources", {}).get("bricks", 0) or 0),
            }
            for p in sim.environment.construction.projects.values()
            if isinstance(p, dict)
        },
        "support_proxy": support_proxy,
        "effective_support_ratio": float(support_proxy.get("effective_colony_support_ratio_current_phase", 0.0) or 0.0),
        "validated_mix": {
            "house": int(validated_mix.get("house", 0) or 0),
            "greenhouse": int(validated_mix.get("greenhouse", 0) or 0),
            "water_generator": int(validated_mix.get("water_generator", 0) or 0),
        },
        "summary_path": str(session / "measures" / "run_summary.json"),
        "events_path": str(session / "logs" / "events.csv"),
    }


def _aggregate(results: list[dict]):
    n = len(results)
    by_type = _blank_type_counter()
    first_times = _blank_first_times()

    for result in results:
        for stype in by_type:
            for milestone in MILESTONES:
                by_type[stype][milestone] += int(result["counts_by_type"][stype][milestone])
                t = result["first_times_by_type"][stype][milestone]
                if t is not None:
                    current = first_times[stype][milestone]
                    if current is None or t < current:
                        first_times[stype][milestone] = t

    avg_by_type = {
        stype: {milestone: round(by_type[stype][milestone] / max(1, n), 3) for milestone in MILESTONES}
        for stype in by_type
    }

    support_ratios = [float(r.get("effective_support_ratio", 0.0)) for r in results]
    selected_actions = Counter()
    for result in results:
        selected_actions.update(result.get("selected_action_counts", {}))
    validated_mix_avg = {
        k: round(mean(float(r.get("validated_mix", {}).get(k, 0)) for r in results), 3)
        for k in ["house", "greenhouse", "water_generator"]
    }

    return {
        "run_count": n,
        "average_counts_by_type": avg_by_type,
        "earliest_first_times_by_type": first_times,
        "avg_validate_selected": round(mean(r["validate_selected_count"] for r in results), 3) if results else 0.0,
        "avg_validate_executed": round(mean(r["validate_executed_count"] for r in results), 3) if results else 0.0,
        "avg_repair_selected": round(mean(r["repair_selected_count"] for r in results), 3) if results else 0.0,
        "avg_repair_executed": round(mean(r["repair_executed_count"] for r in results), 3) if results else 0.0,
        "avg_effective_support_ratio": round(mean(support_ratios), 4) if support_ratios else 0.0,
        "nonzero_support_ratio_runs": int(sum(1 for v in support_ratios if v > 0.0)),
        "avg_validated_mix": validated_mix_avg,
        "selected_action_counts_total": dict(selected_actions),
        "selected_action_counts_avg_per_run": {
            action: round(total / max(1, n), 3) for action, total in selected_actions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Mars construction pipeline bottlenecks.")
    parser.add_argument("--runs-per-condition", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=20260318)
    parser.add_argument("--out", type=Path, default=Path("artifacts/construction_pipeline_bottleneck_audit.json"))
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    task = load_task_model("mars_colony")
    out = {
        "config": {
            "runs_per_condition": args.runs_per_condition,
            "steps": args.steps,
            "dt": args.dt,
            "seed_base": args.seed_base,
            "conditions": CONDITIONS,
        },
        "runs": [],
        "condition_aggregates": {},
        "global_aggregate": {},
    }

    if args.project_root is not None:
        args.project_root.mkdir(parents=True, exist_ok=True)
        project_root_ctx = None
        project_root = args.project_root
    else:
        project_root_ctx = TemporaryDirectory()
        project_root = Path(project_root_ctx.name)

    try:
        seed_cursor = args.seed_base
        grouped = defaultdict(list)
        for condition_name, profile in CONDITIONS.items():
            for _ in range(args.runs_per_condition):
                seed_cursor += 1
                result = _run_once(
                    task=task,
                    condition_name=condition_name,
                    profile=profile,
                    steps=args.steps,
                    dt=args.dt,
                    seed=seed_cursor,
                    project_root=project_root,
                )
                out["runs"].append(result)
                grouped[condition_name].append(result)

        for condition_name, runs in grouped.items():
            out["condition_aggregates"][condition_name] = _aggregate(runs)
        out["global_aggregate"] = _aggregate(out["runs"])
    finally:
        if project_root_ctx is not None:
            project_root_ctx.cleanup()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out["global_aggregate"], indent=2))
    print(f"Saved audit: {args.out}")


if __name__ == "__main__":
    main()
