"""Procedural Baseline characterization batch: 19 one-factor-at-a-time conditions x seeds, compact_panel, 600 sim-s.

python scripts/characterization_batch.py --out Outputs/characterization_<stamp> [--seeds 0 1 2 3 4] [--workers 10] [--duration 600]
        [--only COND ...] [--conditions-file my_design.json] [--list-conditions]

The default design is the 19-condition one-factor-at-a-time characterization around the neutral compact profile (A..S).
--conditions-file replaces it with {"name": {"constructs": {...}, "constituent_overrides": {...}, "factor": ..., "level": ...}, ...}
(constructs default to the neutral 0.5/0.5). Outputs go under --out only; nothing under Outputs/ is tracked by git.

Layout (raw sessions are persistent; every (condition, seed) has its own project root):
    <out>/runs/<condition>/seed_<n>/Outputs/<session>/...   raw session (manifest, events.csv, interaction trace, measures)
    <out>/summaries/<condition>__seed<n>.json               extracted per-run record (config, process, world)
    <out>/progress.log                                      one line per finished run (X/N, elapsed, ETA, failures)

Instrumentation is observational only (event listener, per-tick reads, a pass-through wrapper counting Pilot->middleware rewrites).
No simulator mechanism is altered.
"""
import argparse
import collections
import contextlib
import io
import json
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEUTRAL = {"teamwork_potential": 0.5, "taskwork_potential": 0.5}


def _cond(constructs=None, overrides=None, factor=None, level=None):
    return {"constructs": {**NEUTRAL, **(constructs or {})}, "constituent_overrides": dict(overrides or {}), "factor": factor, "level": level}


CONDITIONS = {"A_neutral": _cond(factor="neutral", level=0.5)}
for letter, key, val in (("B", "teamwork_potential", 0.1), ("C", "teamwork_potential", 0.9), ("D", "taskwork_potential", 0.1), ("E", "taskwork_potential", 0.9)):
    CONDITIONS[letter + "_" + key.split("_")[0] + "_" + ("low" if val < 0.5 else "high")] = _cond({key: val}, factor=key, level=val)
_OVR = [("F", "G", "dik_push_propensity", "push"), ("H", "I", "dik_request_propensity", "request"), ("J", "K", "externalization_propensity", "extern"),
        ("L", "M", "coordination_propensity", "coord"), ("N", "O", "backup_propensity", "backup"), ("P", "Q", "task_execution_efficiency", "taskeff")]
for lo, hi, key, short in _OVR:
    CONDITIONS[lo + "_" + short + "_low"] = _cond(overrides={key: 0.1}, factor=key, level=0.1)
    CONDITIONS[hi + "_" + short + "_high"] = _cond(overrides={key: 0.9}, factor=key, level=0.9)
CONDITIONS["R_memory_0.2"] = _cond(overrides={"memory_reliability": 0.2}, factor="memory_reliability", level=0.2)
CONDITIONS["S_memory_0.6"] = _cond(overrides={"memory_reliability": 0.6}, factor="memory_reliability", level=0.6)


def run_paths(out, name, seed):
    out = Path(out)
    return out / "runs" / name / ("seed_%s" % seed), out / "summaries" / ("%s__seed%s.json" % (name, seed))


def _settings(name):
    c = CONDITIONS[name]
    return {"constructs": dict(c["constructs"]), "constituent_overrides": dict(c["constituent_overrides"]), "profile_mode": "compact_panel"}


def execute_run(name, seed, duration, run_root):
    from modules.agent import Agent
    from modules.simulation import SimulationState

    rewrites = collections.Counter()
    orig_pivots = Agent._apply_policy_pivots

    def counting_pivots(self, decision, *a, **k):  # pass-through: only counts Pilot->middleware action rewrites
        before = decision.selected_action.value
        result = orig_pivots(self, decision, *a, **k)
        if result.selected_action.value != before:
            rewrites["%s->%s" % (before, result.selected_action.value)] += 1
        return result

    Agent._apply_policy_pivots = counting_pivots
    random.seed(int(seed))
    st = SimulationState(phases=None, project_root=str(run_root), flash_mode=True, experiment_name="char_%s_seed%s" % (name, seed), seed=int(seed),
                         brain_backend="rule_brain", experimental_settings=_settings(name))
    sel, executed, late_exec, mode_sel = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    fail_cat, ev = collections.Counter(), collections.Counter()
    artifacts, val_by_project = set(), collections.Counter()
    last_repair_t = [-1.0]
    mode_ticks = collections.Counter()
    clos = collections.Counter()
    watched = {"team_plan_proposed", "team_plan_agreed", "team_plan_response_emitted_immediately", "team_plan_clarification_requested", "communication_exchange",
               "communication_exchange_succeeded", "artifact_consulted", "artifact_consult_assessed", "artifact_consult_suppressed_redundant",
               "recovery_pivot_applied", "policy_pivot_rerouted_rulebrain", "closure_uncommitted_reservation_cleared", "project_stalled",
               "closure_support_focus_applied", "blocked_project_support_pressure_applied", "whiteboard_escape_hatch_forced", "post_inspect_followup_forced",
               "artifact_externalization_action_executed", "mecha_action_blocked", "construction_validated_incorrect", "construction_validated_correct"}

    def listener(e):
        n, p = e["event_type"], e["payload_data"]
        if n in watched:
            ev[n] += 1
        if n == "pilot_selection_via_adapter":
            sel[p.get("selected_action")] += 1
            agent = next((a for a in st.agents if a.name == p.get("agent")), None)
            mode = (getattr(agent, "control_state", None) or {}).get("mode", "?") if agent else "?"
            mode_sel["%s|%s" % (mode, p.get("selected_action"))] += 1
        elif n == "executable_action_completed":
            executed[p.get("decision_action")] += 1
            if st.time >= 400:
                late_exec[p.get("decision_action")] += 1
        elif n == "action_execution_failed":
            fail_cat["%s:%s" % (p.get("decision_action"), p.get("failure_category"))] += 1
        elif n == "artifact_consulted":
            artifacts.add(p.get("artifact_id"))
        elif n == "construction_validation_attempted":
            val_by_project[p.get("project_id")] += 1
        elif n == "construction_repair_episode":
            last_repair_t[0] = float(st.time)

    st.logger.register_event_listener(listener)
    traj, tw400, error = [], None, None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(int(duration)):
                st.update(1.0)
                for a in st.agents:
                    mode_ticks[(getattr(a, "control_state", None) or {}).get("mode", "?")] += 1
                for pid, p in st.environment.construction.projects.items():
                    if p.get("status") == "ready_for_validation" and p.get("closure_status") in ("assigned", "in_progress") and p.get("closure_owner"):
                        owner = next((a for a in st.agents if a.name == p["closure_owner"]), None)
                        s = owner.project_closure_state if owner else {}
                        committed = bool(s.get("active") and s.get("project_id") == pid)
                        clos["committed_owner_ticks" if committed else "stale_owner_ticks"] += 1
                if i == 399:
                    tw400 = sum(a.experimental_metrics.get("taskwork_actions", 0) for a in st.agents)
                if (i + 1) % 100 == 0:
                    cm = st.environment.construction
                    traj.append({"t": i + 1, "validated": sum(int(bool(p.get("validated_complete"))) for p in cm.projects.values()), "projects": len(cm.projects),
                                 "taskwork": sum(a.experimental_metrics.get("taskwork_actions", 0) for a in st.agents)})
        with contextlib.redirect_stdout(io.StringIO()):
            measures = st._final_measures()
            try:
                measures["structural_support"] = st.metrics._structural_support_report()
            except Exception as exc:
                measures["structural_support"] = {"error": repr(exc)}
        manifest = st._experimental_manifest()
        folder = str(st.logger.output_session.session_folder)
    except Exception:
        error = traceback.format_exc()
        measures, manifest, folder = {}, {}, str(getattr(getattr(st.logger, "output_session", None), "session_folder", ""))
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                st.stop()
            except Exception:
                pass
        Agent._apply_policy_pivots = orig_pivots

    return {"condition": name, "seed": int(seed), "duration_s": duration, "profile_mode": "compact_panel", "session_folder": folder, "error": error,
            "factor": CONDITIONS[name]["factor"], "level": CONDITIONS[name]["level"], "settings": _settings(name), "manifest_condition": manifest,
            "measures": measures, "trajectory": traj, "taskwork_at_400": tw400, "selected_actions": dict(sel), "executed_actions": dict(executed),
            "late_executed_actions": dict(late_exec), "mode_x_selected_action": dict(mode_sel), "mode_ticks": dict(mode_ticks),
            "failure_categories": dict(fail_cat), "events": dict(ev), "distinct_artifacts_consulted": sorted(a for a in artifacts if a),
            "validation_attempts_by_project": dict(val_by_project), "last_repair_episode_t": last_repair_t[0], "closure": dict(clos),
            "middleware_rewrites": dict(rewrites)}


def _job(job):
    name, seed, duration, out, retries, spec = job
    CONDITIONS[name] = spec  # spawned workers re-import this module: carry the design in the job
    run_root, summary_path = run_paths(out, name, seed)
    attempt, rec = 0, None
    t0 = time.time()
    while attempt <= retries:
        attempt += 1
        if run_root.exists():  # retry: only this run's own directory
            import shutil
            shutil.rmtree(run_root, ignore_errors=True)
        run_root.mkdir(parents=True, exist_ok=True)
        rec = execute_run(name, seed, duration, run_root)
        if not rec["error"]:
            break
    rec["attempts"], rec["wall_s"] = attempt, round(time.time() - t0, 1)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rec, default=str), encoding="utf-8")
    return name, seed, rec["error"] is None, attempt, rec["wall_s"], (rec.get("measures") or {}).get("construction", {}).get("structures_validated")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--conditions-file", default=None, help="JSON design replacing the default 19 conditions")
    ap.add_argument("--list-conditions", action="store_true")
    args = ap.parse_args(argv)
    if args.conditions_file:
        design = json.loads(Path(args.conditions_file).read_text(encoding="utf-8"))
        CONDITIONS.clear()
        for cname, spec in design.items():
            CONDITIONS[cname] = _cond(spec.get("constructs"), spec.get("constituent_overrides"), spec.get("factor"), spec.get("level"))
    if args.list_conditions:
        for cname, spec in CONDITIONS.items():
            print(cname, json.dumps({"constructs": spec["constructs"], "constituent_overrides": spec["constituent_overrides"]}))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in CONDITIONS:
        if args.only and name not in args.only:
            continue
        for s in args.seeds:
            if run_paths(out, name, s)[1].exists():
                continue  # already completed (resume)
            jobs.append((name, s, args.duration, str(out), args.retries, CONDITIONS[name]))
    total = len(jobs)
    t0, done, failures = time.time(), 0, 0
    log = open(out / "progress.log", "a", encoding="utf-8")
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for name, seed, ok, attempts, wall, validated in pool.imap_unordered(_job, jobs):
            done += 1
            failures += 0 if ok else 1
            el = time.time() - t0
            eta = el / done * (total - done)
            line = ("%d/%d done | elapsed %5.0fs | workers %d | failures %d | latest %s seed%s (%s, attempts %d, wall %ss, validated %s) | ETA %5.0fs"
                    % (done, total, el, min(args.workers, total - done), failures, name, seed, "ok" if ok else "FAILED", attempts, wall, validated, eta))
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())