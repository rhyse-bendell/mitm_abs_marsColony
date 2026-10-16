"""Focused probe: observed repair obligations, validation/repair service, startup substitutions, remaining Pilot-action rewrites.

python scripts/repair_obligation_probe.py --out probe_dir A_neutral:0 B_teamwork_low:3 ...   (condition names as in scripts/characterization_batch.py)
python scripts/repair_obligation_probe.py --out probe_dir --grid A_neutral B_teamwork_low --seeds 0 1 2 3 4 5 6 7

Measurement only. The probe wraps BrainContextBuilder.build to READ the affordances the Pilot was already shown (no extra context
builds, so source-memory decay and RNG are untouched), listens to events, and writes one JSON per run.

repair-ready obligation = (agent, project) pair for which that agent's REPAIR affordance was repair_ready (its own held DIK can ground
the repair) while the project was observed failed. Serviced = that agent selected REPAIR for that project later.
"""
import argparse
import collections
import contextlib
import io
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REPAIR = "repair_or_correct_construction"
VALIDATE = "validate_construction"


def run_one(job):
    name, seed, duration, out = job
    import characterization_batch as cb
    from modules.brain_context import BrainContextBuilder
    from modules.simulation import SimulationState

    random.seed(int(seed))
    ready_first = {}  # (agent, project) -> first time repair_ready
    serviced = {}  # (agent, project) -> first REPAIR selection time after ready
    orig_build = BrainContextBuilder.build

    def build(self, sim, agent, *a, **k):
        packet = orig_build(self, sim, agent, *a, **k)
        t = float(sim.time)
        for aff in packet.action_affordances:
            if aff.get("action_type") == REPAIR and aff.get("repair_ready") is True and aff.get("target_id"):
                ready_first.setdefault((agent.name, aff["target_id"]), t)
        return packet

    BrainContextBuilder.build = build
    root = Path(out) / "runs" / ("%s__seed%s" % (name, seed))
    root.mkdir(parents=True, exist_ok=True)
    st = SimulationState(phases=None, project_root=str(root), flash_mode=True, experiment_name="probe_%s_seed%s" % (name, seed), seed=int(seed), brain_backend="rule_brain",
                         experimental_settings=cb._settings(name))
    sel, late_exec, ev = collections.Counter(), collections.Counter(), collections.Counter()
    last_sel, first_attempt, rewrites, boot_forced = {}, {}, collections.Counter(), {}
    mode_ticks = collections.Counter()

    def lst(e):
        n, p = e["event_type"], e["payload_data"]
        t = float(e["time"])
        if n == "pilot_selection_via_adapter":
            a = p["agent"]
            sel[p["selected_action"]] += 1
            last_sel[a] = p["selected_action"]
            first_attempt[a] = True
            boot_forced[a] = False
            if p["selected_action"] == REPAIR:
                key = (a, p.get("target_id"))
                if key in ready_first and key not in serviced and t >= ready_first[key]:
                    serviced[key] = t
        elif n == "startup_bootstrap_grounding_forced":
            ev["startup_substitutions"] += 1 if p.get("from_action") != p.get("to_action") else 0
            boot_forced[p.get("agent")] = True
        elif n == "executable_action_attempted":
            a = p.get("agent")
            if first_attempt.get(a) and last_sel.get(a) and p.get("planner_action_type") and p["planner_action_type"] != last_sel[a]:
                rewrites[last_sel[a] + "->" + p["planner_action_type"] + ("|bootstrap" if boot_forced.get(a) else "")] += 1
            first_attempt[a] = False
        elif n == "executable_action_completed" and t >= 400:
            late_exec[p.get("decision_action")] += 1
        elif n in ("mecha_strategic_substitution_rejected", "construction_repair_episode"):
            ev[n] += 1

    st.logger.register_event_listener(lst)
    error = None
    failed_end, team, measures = [], {}, {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(int(duration)):
                st.update(1.0)
                for ag in st.agents:
                    mode_ticks[(ag.control_state or {}).get("mode", "?")] += 1
            measures = st._final_measures()
            cm = st.environment.construction
            failed_end = sorted(pid for pid, p in cm.projects.items() if p.get("status") == "needs_repair")
            team = measures["experimental"]["team"]
    except Exception:  # keep the run's evidence
        import traceback

        error = traceback.format_exc()
    finally:
        BrainContextBuilder.build = orig_build
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                st.stop()
            except Exception:
                pass
    pairs = sorted(ready_first)
    never = ["%s:%s" % (a, p) for (a, p) in pairs if (a, p) not in serviced and p in failed_end]
    ttf = [round(serviced[k] - ready_first[k], 1) for k in serviced]
    late_tot = sum(late_exec.values()) or 1
    rec = {
        "condition": name, "seed": seed, "error": error, "validated": (measures.get("construction") or {}).get("structures_validated"), "failed_at_end": failed_end,
        "repair_ready_obligations": len(pairs), "obligations_serviced": len(serviced), "obligations_never_serviced_and_still_failed": never,
        "time_to_first_repair_selection_s": ttf, "median_ttf_s": (sorted(ttf)[len(ttf) // 2] if ttf else None),
        "validation_selections": sel[VALIDATE], "repair_selections": sel[REPAIR], "repair_successes": team.get("repair_successes"), "revalidations": team.get("revalidations"),
        "validation_attempts": team.get("validation_attempts"), "late_share": {k: round(late_exec[k] / late_tot, 3) for k in ("wait", "inspect_information_source", "consult_team_artifact")},
        "startup_substitutions": ev["startup_substitutions"], "rewrites_total": sum(rewrites.values()), "rewrites": dict(rewrites), "strategic_rejections": ev["mecha_strategic_substitution_rejected"],
        "mode_ticks": dict(mode_ticks), "taskwork": team.get("taskwork_actions"), "session_folder": str(st.logger.output_session.session_folder),
    }
    (Path(out) / "summaries").mkdir(parents=True, exist_ok=True)
    (Path(out) / "summaries" / ("%s__seed%s.json" % (name, seed))).write_text(json.dumps(rec, default=str), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--grid", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("jobs", nargs="*", help="condition:seed pairs")
    args = ap.parse_args()
    jobs = [(c, int(s), args.duration, args.out) for c in (args.grid or []) for s in args.seeds]
    jobs += [(j.split(":")[0], int(j.split(":")[1]), args.duration, args.out) for j in args.jobs]
    jobs = list(dict.fromkeys(jobs))
    t0, done = time.time(), 0
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, jobs):
            done += 1
            print("%d/%d | %5.0fs | %s s%s validated=%s failed_end=%d ready=%d serviced=%d never=%d startup_subst=%s rewrites=%s%s" % (
                done, len(jobs), time.time() - t0, rec["condition"], rec["seed"], rec["validated"], len(rec["failed_at_end"]), rec["repair_ready_obligations"],
                rec["obligations_serviced"], len(rec["obligations_never_serviced_and_still_failed"]), rec["startup_substitutions"], rec["rewrites_total"], " ERROR" if rec["error"] else ""), flush=True)


if __name__ == "__main__":
    main()