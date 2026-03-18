#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


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


def _extract_metrics(run_summary: dict) -> dict:
    process = run_summary.get("process", {})
    outcomes = run_summary.get("outcomes", {})
    audit = process.get("behavioral_sanity_audit", {})

    packet_attempts = int(audit.get("packet_absorption_attempted_count", 0))
    packet_success = int(audit.get("packet_absorption_succeeded_count", 0))

    deriv_attempts = int(audit.get("derivation_attempted_count", 0))
    deriv_success = int(audit.get("derivation_succeeded_count", 0))

    d2i_attempts = int(audit.get("data_to_information_attempted_count", 0))
    d2i_success = int(audit.get("data_to_information_succeeded_count", 0))
    i2k_attempts = int(audit.get("information_to_knowledge_attempted_count", 0))
    i2k_success = int(audit.get("information_to_knowledge_succeeded_count", 0))

    colony_proxy = outcomes.get("colony_survivability_proxy", {})

    return {
        "packet_absorption_attempts": packet_attempts,
        "packet_absorption_successes": packet_success,
        "packet_absorption_failures": int(audit.get("packet_absorption_failed_count", 0)),
        "packet_absorption_success_rate": _safe_rate(packet_success, packet_attempts),
        "derivation_attempts": deriv_attempts,
        "derivation_successes": deriv_success,
        "derivation_failures": int(audit.get("derivation_failed_count", 0)),
        "derivation_success_rate": _safe_rate(deriv_success, deriv_attempts),
        "d_to_i_attempts": d2i_attempts,
        "d_to_i_successes": d2i_success,
        "d_to_i_failures": int(audit.get("data_to_information_failed_count", 0)),
        "d_to_i_success_rate": _safe_rate(d2i_success, d2i_attempts),
        "i_to_k_attempts": i2k_attempts,
        "i_to_k_successes": i2k_success,
        "i_to_k_failures": int(audit.get("information_to_knowledge_failed_count", 0)),
        "i_to_k_success_rate": _safe_rate(i2k_success, i2k_attempts),
        "communication_attempts": int(audit.get("communication_attempt_count", 0)),
        "communication_successes": int(audit.get("communication_success_count", 0)),
        "externalization_attempts": int(audit.get("artifact_externalization_attempt_count", 0)),
        "externalization_created": int(audit.get("artifact_externalization_created_count", 0)),
        "artifact_consult_attempts": int(audit.get("artifact_consult_attempt_count", 0)),
        "artifact_consults": int(audit.get("artifact_consult_success_count", 0)),
        "artifact_adoptions": int(process.get("team_knowledge", {}).get("artifact_adoption_count", 0)),
        "mismatch_detections": int(audit.get("mismatch_detected_count", 0)),
        "repair_attempts": int(audit.get("construction_repair_attempt_count", 0)),
        "repair_successes": int(audit.get("construction_repair_success_count", 0)),
        "validated_structures": int(outcomes.get("total_structures_validated_correct", 0)),
        "completed_structures": int(outcomes.get("total_structures_completed", 0)),
        "structures_repaired": int(outcomes.get("total_structures_repaired_or_corrected", 0)),
        "colony_validated_ratio": float(colony_proxy.get("validated_structure_ratio", 0.0) or 0.0),
        "phase_objectives_completed": int(sum(1 for ok in outcomes.get("phase_objective_completion", {}).values() if ok)),
    }


def _condition_means(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: mean(float(r[k]) for r in rows) for k in keys}


def _directional_delta(condition_means: dict[str, dict], high_keys: tuple[str, str], low_keys: tuple[str, str], metric: str) -> float:
    high = mean(condition_means[c][metric] for c in high_keys)
    low = mean(condition_means[c][metric] for c in low_keys)
    return high - low


def run_audit(runs_per_condition: int, steps: int, dt: float, seed_base: int, project_root: Path) -> dict:
    task = load_task_model("mars_colony")
    per_condition_runs: dict[str, list[dict]] = defaultdict(list)

    for condition_name, profile in CONDITIONS.items():
        for rep in range(runs_per_condition):
            random.seed(seed_base + (rep * 100) + hash(condition_name) % 97)
            sim = SimulationState(
                agent_configs=_build_agent_configs(task, profile["taskwork_potential"], profile["teamwork_potential"]),
                phases=MISSION_PHASES,
                flash_mode=True,
                experiment_name=f"audit_{condition_name}",
                project_root=project_root,
                brain_backend="rule_brain",
            )
            for _ in range(steps):
                sim.update(dt)
            sim.stop()
            summary = json.loads((sim.logger.output_session.measures_dir / "run_summary.json").read_text(encoding="utf-8"))
            per_condition_runs[condition_name].append(_extract_metrics(summary))

    condition_means = {name: _condition_means(rows) for name, rows in per_condition_runs.items()}

    directional = {
        "task_high_minus_low": {
            "packet_absorption_success_rate": _directional_delta(
                condition_means,
                ("high_task_high_team", "high_task_low_team"),
                ("low_task_high_team", "low_task_low_team"),
                "packet_absorption_success_rate",
            ),
            "derivation_success_rate": _directional_delta(
                condition_means,
                ("high_task_high_team", "high_task_low_team"),
                ("low_task_high_team", "low_task_low_team"),
                "derivation_success_rate",
            ),
            "colony_validated_ratio": _directional_delta(
                condition_means,
                ("high_task_high_team", "high_task_low_team"),
                ("low_task_high_team", "low_task_low_team"),
                "colony_validated_ratio",
            ),
        },
        "team_high_minus_low": {
            "communication_successes": _directional_delta(
                condition_means,
                ("high_task_high_team", "low_task_high_team"),
                ("high_task_low_team", "low_task_low_team"),
                "communication_successes",
            ),
            "externalization_created": _directional_delta(
                condition_means,
                ("high_task_high_team", "low_task_high_team"),
                ("high_task_low_team", "low_task_low_team"),
                "externalization_created",
            ),
            "artifact_adoptions": _directional_delta(
                condition_means,
                ("high_task_high_team", "low_task_high_team"),
                ("high_task_low_team", "low_task_low_team"),
                "artifact_adoptions",
            ),
        },
    }

    return {
        "config": {
            "runs_per_condition": runs_per_condition,
            "steps": steps,
            "dt": dt,
            "seed_base": seed_base,
            "conditions": CONDITIONS,
        },
        "condition_means": condition_means,
        "condition_runs": per_condition_runs,
        "directional_deltas": directional,
    }


def main():
    parser = argparse.ArgumentParser(description="Run small profile/construct behavioral sanity audit.")
    parser.add_argument("--runs-per-condition", type=int, default=6)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=20260318)
    parser.add_argument("--out", type=Path, default=Path("artifacts/profile_behavior_sanity_audit.json"))
    parser.add_argument("--project-root", type=Path, default=None, help="Optional explicit project root for simulation Outputs.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.project_root is not None:
        args.project_root.mkdir(parents=True, exist_ok=True)
        results = run_audit(args.runs_per_condition, args.steps, args.dt, args.seed_base, args.project_root)
    else:
        with TemporaryDirectory() as tmpdir:
            results = run_audit(args.runs_per_condition, args.steps, args.dt, args.seed_base, Path(tmpdir))

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Profile Behavior Sanity Audit ===")
    print(json.dumps(results["config"], indent=2))
    print("\nCondition means:")
    for name, metrics in results["condition_means"].items():
        print(
            f"- {name}: "
            f"packet_sr={metrics['packet_absorption_success_rate']:.3f}, "
            f"deriv_sr={metrics['derivation_success_rate']:.3f}, "
            f"comm={metrics['communication_successes']:.2f}, "
            f"ext={metrics['externalization_created']:.2f}, "
            f"adopt={metrics['artifact_adoptions']:.2f}, "
            f"validated={metrics['validated_structures']:.2f}, "
            f"survival_ratio={metrics['colony_validated_ratio']:.3f}"
        )

    print("\nDirectional deltas (high - low):")
    print(json.dumps(results["directional_deltas"], indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
