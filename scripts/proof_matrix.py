"""Proof matrix: eight conditions x seeds, run in parallel; writes one JSON per run and prints a compact table.

python scripts/proof_matrix.py --seeds 0 1 2 --duration 600 --out <dir>
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONDITIONS = {
    "A_neutral": {"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}},
    "B_high_teamwork": {"constructs": {"teamwork_potential": 0.9, "taskwork_potential": 0.5}},
    "C_low_teamwork": {"constructs": {"teamwork_potential": 0.1, "taskwork_potential": 0.5}},
    "D_high_taskwork": {"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.9}},
    "E_low_taskwork": {"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.1}},
    "F_high_push_only": {"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "constituent_overrides": {"dik_push_propensity": 0.95}},
    "G_high_extern_low_memory": {"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "constituent_overrides": {"externalization_propensity": 0.95, "memory_reliability": 0.2}},
    "H_fast_tasks_low_teamwork": {"constructs": {"teamwork_potential": 0.1, "taskwork_potential": 0.5}, "constituent_overrides": {"task_execution_efficiency": 0.95}},
}


def _settings(name, taskwork):
    import copy

    settings = copy.deepcopy(CONDITIONS[name])
    if taskwork is not None and name not in {"D_high_taskwork", "E_low_taskwork"}:
        settings["constructs"]["taskwork_potential"] = taskwork  # hold taskwork fixed so outcomes are not floored by low rule accuracy
    return settings


def _run(job):
    name, seed, duration, out, taskwork = job
    from modules.experiment_runner import run_condition

    path = Path(out) / f"{name}__seed{seed}.json"
    if path.exists():
        return name, seed, str(path)
    summary = run_condition(seed=seed, duration_s=duration, experimental_settings=_settings(name, taskwork), experiment_name=f"proof_{name}")
    summary["condition"] = name
    path.write_text(json.dumps(summary, default=str), encoding="utf-8")
    return name, seed, str(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--taskwork", type=float, default=None, help="fix taskwork_potential for all conditions except D/E")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, args.duration, str(out), args.taskwork) for n in CONDITIONS if not args.only or n in args.only for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, seed, path in pool.map(_run, jobs):
            print("done", name, seed, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())