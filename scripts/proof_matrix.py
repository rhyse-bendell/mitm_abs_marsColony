"""Proof matrix: eight conditions x seeds, run in parallel; persists raw sessions and one summary JSON per run.

python scripts/proof_matrix.py --seeds 0 1 2 --duration 600 --out <dir> [--profile-mode compact_panel] [--existing skip|overwrite|error]

Layout (nothing is written to a temporary directory, so raw logs always survive the run):
    <out>/runs/<condition>/seed_<n>/Outputs/<session>/...   raw session (manifest, event logs, measures)
    <out>/summaries/<condition>__seed<n>.json               run summary; ``session_folder`` is the real persistent path
Each (condition, seed) has its own root, so concurrent workers can never share or overwrite a session.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_PROFILE_MODE = "compact_panel"  # the proof matrix is a compact-panel study; it is passed explicitly, never inherited

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


def _settings(name, taskwork, profile_mode=DEFAULT_PROFILE_MODE):
    import copy

    settings = copy.deepcopy(CONDITIONS[name])
    if taskwork is not None and name not in {"D_high_taskwork", "E_low_taskwork"}:
        settings["constructs"]["taskwork_potential"] = taskwork  # hold taskwork fixed so outcomes are not floored by low rule accuracy
    settings["profile_mode"] = profile_mode
    return settings


def run_paths(out, name, seed):
    out = Path(out)
    return out / "runs" / name / f"seed_{seed}", out / "summaries" / f"{name}__seed{seed}.json"


def _run(job):
    name, seed, duration, out, taskwork, profile_mode, existing = job
    from modules.experiment_runner import run_condition

    run_root, summary_path = run_paths(out, name, seed)
    if run_root.exists() or summary_path.exists():
        if existing == "skip":
            return name, seed, str(summary_path), "skipped"
        if existing == "error":
            raise FileExistsError(f"{run_root} or {summary_path} already exists (use --existing skip|overwrite)")
        shutil.rmtree(run_root, ignore_errors=True)  # overwrite: only this run's own directory
        summary_path.unlink(missing_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = run_condition(seed=seed, duration_s=duration, experimental_settings=_settings(name, taskwork, profile_mode), experiment_name=f"proof_{name}_seed{seed}", project_root=run_root)
    summary["condition"] = name
    summary["profile_mode"] = profile_mode
    summary["run_root"] = str(run_root)
    summary_path.write_text(json.dumps(summary, default=str), encoding="utf-8")
    return name, seed, str(summary_path), "ran"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--taskwork", type=float, default=None, help="fix taskwork_potential for all conditions except D/E")
    parser.add_argument("--profile-mode", default=DEFAULT_PROFILE_MODE, help="broad_legacy or compact_panel (default compact_panel)")
    parser.add_argument("--existing", choices=["skip", "overwrite", "error"], default="skip", help="what to do when a (condition, seed) already has output")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, args.duration, str(out), args.taskwork, args.profile_mode, args.existing) for n in CONDITIONS if not args.only or n in args.only for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, seed, path, status in pool.map(_run, jobs):
            print(status, name, seed, path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())