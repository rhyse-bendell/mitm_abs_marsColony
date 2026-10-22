"""Characterization analysis: per-run CSV, condition summary CSV, figures, and report tables from a characterization batch.

python scripts/characterization_report.py <batch_out_dir>
Writes <batch_out_dir>/analysis/{characterization_runs.csv, characterization_condition_summary.csv, report_tables.md, fig*.png}.
Descriptive only: N=5 seeds per condition, no inferential claims.
"""
import csv
import json
import statistics as stats
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACTIONS = ["start_construction", "continue_construction", "transport_resources", "repair_or_correct_construction", "validate_construction"]


STALL_WINDOW_S = 150
STALL_STRIDE_S = 10
STALL_FIRST_WINDOW_END_S = 250  # skip the start-up grounding phase: windows must begin at or after t=100
STALL_DOMINANCE = 0.6
WORLD_PROGRESS_EVENTS = {"construction_material_progress_updated", "construction_progress_updated", "construction_resource_delivered", "construction_build_progress_updated",
                         "construction_validation_attempted", "construction_repair_episode", "construction_completed", "project_connection_added", "project_created_by_start",
                         "construction_validated_correct", "construction_validated_incorrect"}


def scan_events(events_csv, duration):
    """Measurement-only process-evidence scan of a raw events.csv (works identically on any batch, old or new).

    Probable stall = a 150 s window (ending at t>=250) with NO world/project-state progress event, NO first-time DIK acquisition,
    NO meaningful communication outcome, AND a repeated pattern (one action holds >= 60% of the Pilot selections in the
    window). The detector reads logs after the fact; it cannot influence behaviour. It is generic: it names no action.
    """
    import collections

    out = {"stall_seconds": 0, "flag_probable_stall": 0, "stall_at_end": 0, "first_stall_t": None, "longest_no_progress_s": 0, "inspect_no_gain_count": 0,
           "inspect_no_gain_max_streak": 0, "inspect_total_completed": 0, "e2e_rewrites_total": 0, "e2e_rewrites_json": "{}", "strategic_substitutions_rejected": 0,
           "stall_dominant_pattern": ""}
    progress_t, dik_t, comm_t = [], [], []
    selections = []  # (t, agent, action, target)
    seen_dik = set()
    last_sel = {}
    first_attempt_pending = {}
    e2e = collections.Counter()
    e2e_bootstrap = collections.Counter()  # rewrites by the documented start-up grounding scaffold (BOOTSTRAP mode), reported separately
    bootstrap_forced = {}
    streak = collections.Counter()
    try:
        f = open(events_csv, "r", encoding="utf-8", newline="")
    except OSError:
        return out
    with f:
        for r in csv.DictReader(f):
            t, n = float(r["time"]), r["event_type"]
            if n in WORLD_PROGRESS_EVENTS:
                progress_t.append(t)
                continue
            if n in ("pilot_selection_via_adapter", "startup_bootstrap_grounding_forced", "executable_action_attempted", "inspect_completed", "inspect_success_no_new_dik", "inspect_success_dik_changed",
                     "communication_exchange", "dik_acquired_from_inspect", "dik_acquired_from_artifact", "shared_source_dik_acquired_agent", "mecha_strategic_substitution_rejected"):
                p = json.loads(r["payload"])
            else:
                continue
            a = p.get("agent")
            if n == "pilot_selection_via_adapter":
                selections.append((t, a, p.get("selected_action"), p.get("target_id")))
                last_sel[a] = p.get("selected_action")
                first_attempt_pending[a] = True
                bootstrap_forced[a] = False
            elif n == "startup_bootstrap_grounding_forced":
                bootstrap_forced[a] = True
            elif n == "executable_action_attempted":
                if first_attempt_pending.get(a) and last_sel.get(a) and p.get("planner_action_type") and p["planner_action_type"] != last_sel[a]:
                    (e2e_bootstrap if bootstrap_forced.get(a) else e2e)[f"{last_sel[a]}->{p['planner_action_type']}"] += 1
                first_attempt_pending[a] = False
            elif n == "communication_exchange":
                if p.get("meaningful"):
                    comm_t.append(t)
            elif n == "mecha_strategic_substitution_rejected":
                out["strategic_substitutions_rejected"] += 1
            elif n == "inspect_completed":
                out["inspect_total_completed"] += 1
            elif n == "inspect_success_no_new_dik":
                out["inspect_no_gain_count"] += 1
                streak[a] += 1
                out["inspect_no_gain_max_streak"] = max(out["inspect_no_gain_max_streak"], streak[a])
            elif n == "inspect_success_dik_changed":
                streak[a] = 0
            if n in ("dik_acquired_from_inspect", "dik_acquired_from_artifact", "shared_source_dik_acquired_agent", "inspect_success_dik_changed"):
                ids = [*p.get("new_information_ids", []), *p.get("new_data_ids", []), *p.get("new_rule_ids", [])]
                if p.get("information_id"):
                    ids.append(p["information_id"])
                ids = [str(i).split(":", 1)[-1] for i in ids if "Inferred rule" not in str(i)]  # composite inferred-rule strings are recombinations, not new DIK
                if any((a, i) not in seen_dik for i in ids):
                    dik_t.append(t)
                seen_dik.update((a, i) for i in ids)
    out["e2e_rewrites_total"] = sum(e2e.values()) + sum(e2e_bootstrap.values())
    out["e2e_rewrites_json"] = json.dumps(dict(e2e + e2e_bootstrap), sort_keys=True)
    out["e2e_rewrites_nonbootstrap_total"] = sum(e2e.values())
    out["e2e_rewrites_nonbootstrap_json"] = json.dumps(dict(e2e), sort_keys=True)
    out["e2e_rewrites_bootstrap_total"] = sum(e2e_bootstrap.values())
    import bisect

    def any_in(ts, lo, hi):
        ts.sort()
        i = bisect.bisect_right(ts, lo)
        return i < len(ts) and ts[i] <= hi

    last_prog = 0.0
    allp = sorted(progress_t + dik_t + comm_t)
    prev = 0.0
    for t in allp + [float(duration)]:
        out["longest_no_progress_s"] = max(out["longest_no_progress_s"], t - prev)
        prev = t
    selections.sort()
    stall_ends = []
    t_end = STALL_FIRST_WINDOW_END_S
    while t_end <= duration:
        lo = t_end - STALL_WINDOW_S
        quiet = not any_in(progress_t, lo, t_end) and not any_in(dik_t, lo, t_end) and not any_in(comm_t, lo, t_end)
        if quiet:
            win = [x for x in selections if lo < x[0] <= t_end]
            if win:
                c = collections.Counter(x[2] for x in win)  # repeated action; targets are reported separately (agents differ)
                (pattern, cnt), = c.most_common(1)
                if cnt / len(win) >= STALL_DOMINANCE:
                    stall_ends.append(t_end)
                    if not out["stall_dominant_pattern"]:
                        tg = collections.Counter(x[3] for x in win if x[2] == pattern).most_common(2)
                        out["stall_dominant_pattern"] = f"{pattern} share={cnt / len(win):.2f} targets={tg}"
        t_end += STALL_STRIDE_S
    if stall_ends:
        out["flag_probable_stall"] = 1
        out["first_stall_t"] = stall_ends[0] - STALL_WINDOW_S
        covered = set()
        for e in stall_ends:
            covered.update(range(int(e - STALL_WINDOW_S), int(e)))
        out["stall_seconds"] = len(covered)
        out["stall_at_end"] = int(stall_ends[-1] >= duration - STALL_STRIDE_S)
    return out


def g(d, *path, default=0):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def extract(r):
    m = r.get("measures") or {}
    team = g(m, "experimental", "team", default={})
    cons = g(m, "construction", default={})
    ss = g(m, "structural_support", default={})
    per_agent = g(m, "experimental", "per_agent", default={})
    first = next(iter(per_agent.values()), {})
    resolved = first.get("resolved_constituents", {})
    mc = r.get("manifest_condition", {})
    pa = mc.get("per_agent", {}) if isinstance(mc, dict) else {}
    pa0 = next(iter(pa.values()), {}) if pa else {}
    ex, late, ev = r.get("executed_actions", {}), r.get("late_executed_actions", {}), r.get("events", {})
    total_ex, late_tot = sum(ex.values()) or 1, sum(late.values()) or 1
    dur_s, dur_n = team.get("duration_seconds_by_action", {}), team.get("duration_count_by_action", {})
    status = cons.get("projects_by_status", {})
    traj = {t["t"]: t for t in r.get("trajectory", [])}
    tw = team.get("taskwork_actions", 0)
    tw400 = r.get("taskwork_at_400")
    mode_ticks = r.get("mode_ticks", {})
    mt = sum(mode_ticks.values()) or 1
    rew = r.get("middleware_rewrites", {})
    houses = g(ss, "housing", default={})
    row = {
        "condition": r["condition"], "letter": r["condition"][0], "seed": r["seed"], "factor": r.get("factor"), "level": r.get("level"),
        "profile_mode": r.get("profile_mode"), "error": (r.get("error") or "")[:80].replace("\n", " "), "attempts": r.get("attempts"), "wall_s": r.get("wall_s"),
        "teamwork_potential": g(r, "settings", "constructs", "teamwork_potential"), "taskwork_potential": g(r, "settings", "constructs", "taskwork_potential"),
    }
    for k in ("dik_push_propensity", "dik_request_propensity", "externalization_propensity", "memory_reliability", "coordination_propensity", "backup_propensity", "task_execution_efficiency"):
        row["c_" + k] = resolved.get(k)
    for k, v in (pa0.get("residual_mechanism_values") or {}).items():
        row["resid_" + k] = v
    row["build_speed"] = (pa0.get("constituent_mechanisms") or {}).get("build_speed")
    # process
    for k in ("taskwork_actions", "taskwork_completed", "taskwork_seconds", "provision_attempts", "provision_successes", "unsolicited_provision_attempts", "request_attempts", "request_successes",
              "externalization_episodes", "team_plan_proposals", "dik_acquired", "dik_forgotten", "dik_reacquired", "dik_retained", "backup_actions", "shared_infrastructure_actions",
              "validation_attempts", "validation_failures", "repair_selections", "repair_applications", "repair_successes", "repair_blocked", "revalidations"):
        row[k] = team.get(k, 0)
    row["solicited_provision_attempts"] = row["provision_attempts"] - row["unsolicited_provision_attempts"]
    row["backup_share"] = round(row["backup_actions"] / row["taskwork_actions"], 4) if row["taskwork_actions"] else 0
    for a in ACTIONS:
        short = a.split("_")[0]
        row["dur_total_" + short] = dur_s.get(a, 0)
        row["dur_n_" + short] = dur_n.get(a, 0)
        row["dur_mean_" + short] = round(dur_s.get(a, 0) / dur_n[a], 4) if dur_n.get(a) else None
    row["mean_seconds_per_taskwork_action"] = round(row["taskwork_seconds"] / row["taskwork_actions"], 4) if row["taskwork_actions"] else None
    tk = team.get("taskwork_by_kind", {})
    row.update({"taskwork_transport": tk.get("transport", 0), "taskwork_build": tk.get("build", 0), "taskwork_validate": tk.get("validate", 0), "taskwork_repair": tk.get("repair", 0)})
    row["consult_selected"] = r.get("selected_actions", {}).get("consult_team_artifact", 0)
    row["consult_executed"] = ex.get("consult_team_artifact", 0)
    row["artifact_consulted_events"] = ev.get("artifact_consulted", 0)
    row["distinct_artifacts_consulted"] = len(r.get("distinct_artifacts_consulted", []))
    row["consult_suppressed_redundant"] = ev.get("artifact_consult_suppressed_redundant", 0)
    row["externalize_executed"] = ev.get("artifact_externalization_action_executed", 0)
    for k in ("team_plan_proposed", "team_plan_agreed", "team_plan_response_emitted_immediately", "team_plan_clarification_requested", "communication_exchange", "communication_exchange_succeeded"):
        row["ev_" + k] = ev.get(k, 0)
    row["coordination_events"] = sum(ev.get(k, 0) for k in ("team_plan_proposed", "team_plan_agreed", "team_plan_clarification_requested", "team_plan_response_emitted_immediately"))
    row["executed_total"] = sum(ex.values())
    row["wait_share_all"] = round(ex.get("wait", 0) / total_ex, 4)
    row["late_wait_share"] = round(late.get("wait", 0) / late_tot, 4)
    row["late_inspect_share"] = round(late.get("inspect_information_source", 0) / late_tot, 4)
    row["late_consult_share"] = round(late.get("consult_team_artifact", 0) / late_tot, 4)
    row["consult_exec_share"] = round(ex.get("consult_team_artifact", 0) / total_ex, 4)
    for md in sorted(mode_ticks):
        row["modeshare_" + md] = round(mode_ticks[md] / mt, 4)
    row["middleware_rewrites_total"] = sum(rew.values())
    row["rewrites_to_wait"] = sum(v for k, v in rew.items() if k.endswith("->wait"))
    row["rewrite_types"] = json.dumps(rew, sort_keys=True)
    row["recovery_pivots"] = ev.get("recovery_pivot_applied", 0)
    row["policy_pivot_rerouted"] = ev.get("policy_pivot_rerouted_rulebrain", 0)
    row["stale_closure_owner_ticks"] = r.get("closure", {}).get("stale_owner_ticks", 0)
    row["committed_closure_owner_ticks"] = r.get("closure", {}).get("committed_owner_ticks", 0)
    row["closure_reservations_cleared"] = ev.get("closure_uncommitted_reservation_cleared", 0)
    row["projects_stalled_events"] = ev.get("project_stalled", 0)
    row["max_validation_attempts_one_project"] = max(r.get("validation_attempts_by_project", {}).values() or [0])
    row["start_construction_failures"] = sum(v for k, v in r.get("failure_categories", {}).items() if k.startswith("start_construction"))
    row["action_failures_total"] = sum(r.get("failure_categories", {}).values())
    row["taskwork_last200"] = (tw - tw400) if tw400 is not None else None
    # world
    row["projects_started"] = len(cons.get("projects", []))
    row["projects_complete"] = status.get("complete", 0)
    row["projects_needs_repair"] = status.get("needs_repair", 0)
    row["projects_in_progress"] = status.get("in_progress", 0)
    row["structures_validated"] = cons.get("structures_validated", 0)  # legacy all-types count; connectors included (not a performance readout)
    ob = cons.get("outcome_breakdown") or {}
    for key in ("validated_core_structures", "validated_houses", "validated_greenhouses", "validated_generators", "validated_connectors", "validated_food_connectors", "validated_water_connectors",
                "connector_projects_started", "connector_projects_with_links", "speculative_connector_projects", "connector_links_total", "validated_connectors_with_links",
                "validated_connectors_without_links", "connector_share_of_projects_started", "site_slots_used", "site_slots_total"):
        row[key] = ob.get(key)
    row["supported_civilian_capacity_min"] = g(ob, "supported_civilian_capacity", "min")
    row["supported_civilian_capacity_max"] = g(ob, "supported_civilian_capacity", "max")
    row["supported_vip_capacity_if_compatible_max"] = g(ob, "supported_vip_capacity_if_vip_compatible", "max")
    row["ss_bottleneck_reasons"] = json.dumps(g(ss, "bottleneck", "all_reasons", default={}), sort_keys=True)
    row["site_slots_exhausted"] = int(bool(ob.get("site_slots_exhausted")))
    row["supported_housing_max"] = g(ob, "structurally_supported_housing", "max")
    row["supported_greenhouses_max"] = g(ob, "structurally_supported_greenhouses", "max")
    row["connections_total"] = g(team, "connections", "total", default=g(m, "experimental", "connections", "total"))
    row["connections_cross_site"] = g(m, "experimental", "connections", "cross_site")
    row["bricks_delivered_to_projects"] = cons.get("bricks_delivered_to_projects", 0)
    piles = cons.get("resource_piles", {})
    row["bricks_remaining"] = sum(p.get("remaining", 0) for p in piles.values())
    row["bricks_initial"] = sum(p.get("initial", 0) for p in piles.values())
    row["ss_housing_validated"] = houses.get("validated_structures", 0)
    row["ss_housing_civilian_capacity"] = houses.get("civilian_capacity", 0)
    row["ss_housing_with_food_link"] = houses.get("with_food_link", 0)
    row["ss_housing_with_water"] = houses.get("with_water_connection", 0)
    row["ss_fully_supported_cap_min"] = g(houses, "fully_supported_civilian_capacity", "min")
    row["ss_fully_supported_cap_max"] = g(houses, "fully_supported_civilian_capacity", "max")
    row["ss_fully_supported_structures_max"] = g(houses, "fully_supported_structures", "max")
    row["ss_food_greenhouses_validated"] = g(ss, "food", "validated_greenhouses")
    row["ss_food_effective_cap_max"] = g(ss, "food", "effective_civilian_equivalent_capacity", "max")
    row["ss_water_generators_validated"] = g(ss, "water", "validated_generators")
    row["ss_water_effective_connections"] = g(ss, "water", "effective_connections")
    row["ss_violations"] = len(ss.get("violations", [])) if isinstance(ss.get("violations"), list) else 0
    row["ss_limiting_factor"] = g(ss, "bottleneck", "limiting_factor", default=None)
    for t in (100, 200, 300, 400, 500, 600):
        row[f"validated_t{t}"] = traj.get(t, {}).get("validated")
        row[f"taskwork_t{t}"] = traj.get(t, {}).get("taskwork")
    # flags
    row["flag_absorbing"] = int(bool(row["error"] == "" and (((tw - tw400) == 0) if tw400 is not None else False) or row["late_wait_share"] > 0.5))
    row["flag_excess_consult"] = int(row["consult_exec_share"] > 0.25)
    row["flag_repeated_validation"] = int(row["max_validation_attempts_one_project"] >= 6)
    row["flag_repeated_start"] = int(row["start_construction_failures"] >= 10)
    lr = r.get("last_repair_episode_t", -1)
    row["last_repair_episode_t"] = lr
    row["flag_repair_starvation"] = int(row["projects_needs_repair"] > 0 and lr < 400)
    row["flag_stale_closure"] = int(row["stale_closure_owner_ticks"] >= 30)
    row["flag_exception"] = int(bool(row["error"]))
    ev_csv = next(iter(sorted(Path(r.get("session_folder", "")).glob("logs/events.csv"))), None) if r.get("session_folder") else None
    scan = scan_events(ev_csv, r.get("duration_s", 600)) if ev_csv else {}
    for k, v in scan.items():
        row[k] = v
    row["any_flag"] = int(any(row[k] for k in row if k.startswith("flag_")))
    return row


FLAGS = ["flag_absorbing", "flag_probable_stall", "flag_excess_consult", "flag_repeated_validation", "flag_repeated_start", "flag_repair_starvation", "flag_stale_closure", "flag_exception"]


def main(out):
    out = Path(out)
    ana = out / "analysis"
    ana.mkdir(exist_ok=True)
    rows = []
    for p in sorted((out / "summaries").glob("*.json")):
        rows.append(extract(json.loads(p.read_text(encoding="utf-8"))))
    rows.sort(key=lambda r: (r["condition"], r["seed"]))
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(ana / "characterization_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    conds = sorted({r["condition"] for r in rows})
    numeric = [c for c in cols if c not in ("condition", "letter", "factor", "profile_mode", "error", "rewrite_types", "ss_limiting_factor") and any(isinstance(r.get(c), (int, float)) for r in rows)]

    def vals(cond, key):
        return [r[key] for r in rows if r["condition"] == cond and isinstance(r.get(key), (int, float))]

    summ = []
    for c in conds:
        s = {"condition": c, "n_runs": len([r for r in rows if r["condition"] == c])}
        for k in numeric:
            v = vals(c, k)
            if v:
                s[k + "_mean"] = round(float(np.mean(v)), 4)
                s[k + "_sd"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0
                s[k + "_min"] = min(v)
                s[k + "_max"] = max(v)
                s[k + "_median"] = round(float(np.median(v)), 4)
        summ.append(s)
    scols = []
    for s in summ:
        for k in s:
            if k not in scols:
                scols.append(k)
    with open(ana / "characterization_condition_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        w.writerows(summ)

    # ---------- figures
    rng = np.random.default_rng(0)
    colors = plt.cm.tab10(np.arange(10))
    seeds = sorted({r["seed"] for r in rows})

    def strip(ax, xs_labels, key, title=None, ylabel=None):
        for i, c in enumerate(xs_labels):
            v = vals(c, key)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=22, alpha=0.8, color=colors[i % 10], edgecolor="k", linewidth=0.3, zorder=3)
            if v:
                ax.hlines(np.mean(v), i - 0.3, i + 0.3, color="k", linewidth=2, zorder=4)
        ax.set_xticks(range(len(xs_labels)))
        ax.set_xticklabels([x.replace("_", "\n", 1) for x in xs_labels], fontsize=6, rotation=90)
        ax.set_title(title or key, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Fig 1 manipulation checks
    checks = [("dik_push_propensity", "push", ["unsolicited_provision_attempts", "provision_attempts"]),
              ("dik_request_propensity", "request", ["request_attempts"]),
              ("externalization_propensity", "extern", ["externalization_episodes"]),
              ("coordination_propensity", "coord", ["team_plan_proposals", "coordination_events"]),
              ("backup_propensity", "backup", ["backup_share"]),
              ("task_execution_efficiency", "taskeff", ["dur_mean_start", "dur_mean_transport"]),
              ("memory_reliability", "memory", ["dik_forgotten"])]
    names = {c.split("_", 1)[1] if c[1] == "_" else c: c for c in conds}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    manip_rows = []
    for ax, (const, short, readouts) in zip(axes.flat, checks):
        cs = {}
        if short == "memory":
            cs = {0.2: "R_memory_0.2", 0.6: "S_memory_0.6", 1.0: "A_neutral"}
        else:
            lo = [c for c in conds if c.split("_")[1] == short and c.endswith("_low")]
            hi = [c for c in conds if c.split("_")[1] == short and c.endswith("_high")]
            cs = {0.1: lo[0], 0.5: "A_neutral", 0.9: hi[0]}
        for j, key in enumerate(readouts):
            xs, ys, means = [], [], []
            for lvl, c in cs.items():
                v = vals(c, key)
                xs += [lvl + (j - (len(readouts) - 1) / 2) * 0.03 + rng.uniform(-0.01, 0.01) for _ in v]
                ys += v
                means.append((lvl, float(np.mean(v)) if v else np.nan))
            ax.scatter(xs, ys, s=20, alpha=0.75, label=key, color=colors[j], edgecolor="k", linewidth=0.3)
            ax.plot([m[0] for m in means], [m[1] for m in means], "-", color=colors[j], linewidth=1.5)
            manip_rows.append((const, key, [(l, round(mv, 3)) for l, mv in means]))
        ax.set_title(const + "  ->  " + ", ".join(readouts), fontsize=8)
        ax.set_xlabel("resolved " + const, fontsize=8)
        ax.set_xticks(sorted(cs))
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.suptitle("Fig 1. Manipulation checks: each constituent vs. its most proximal readout (dots = seeds, lines = condition means; N=5, descriptive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig1_manipulation_checks.png", dpi=130)
    plt.close(fig)

    # Fig 2 outcomes
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    strip(axes[0, 0], conds, "validated_core_structures", "Validated CORE structures (house/greenhouse/generator) at 600 s")
    strip(axes[0, 1], conds, "ss_fully_supported_cap_max", "Fully supported civilian capacity (max bound)")
    strip(axes[1, 0], conds, "ss_housing_validated", "Validated housing structures")
    strip(axes[1, 1], conds, "projects_needs_repair", "Projects still needing repair at 600 s")
    fig.suptitle("Fig 2. Outcomes / structural support by condition (dots = seeds, bar = mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig2_outcomes.png", dpi=130)
    plt.close(fig)

    # Fig 3 process
    keys = [("unsolicited_provision_attempts", "Unsolicited DIK provisions"), ("solicited_provision_attempts", "Solicited provisions"), ("request_attempts", "DIK requests"),
            ("externalization_episodes", "Externalization episodes"), ("coordination_events", "Coordination events (plan proposed/agreed/clarif./response)"),
            ("backup_share", "Backup share of taskwork"), ("consult_executed", "Artifact consults executed"), ("repair_successes", "Repair successes"),
            ("late_inspect_share", "Late (t>=400) share of executed micro-actions that are INSPECT")]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for ax, (k, t) in zip(axes.flat, keys):
        strip(ax, conds, k, t)
    fig.suptitle("Fig 3. Process measures by condition (communication, externalization, coordination, backup, consults, repair)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig3_process.png", dpi=130)
    plt.close(fig)

    # Fig 4 duration
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    te = {0.1: "P_taskeff_low", 0.5: "A_neutral", 0.9: "Q_taskeff_high"}
    for j, a in enumerate(ACTIONS):
        short = a.split("_")[0]
        xs, ys, mm = [], [], []
        for lvl, c in te.items():
            v = vals(c, "dur_mean_" + short)
            xs += [lvl + (j - 2) * 0.02 for _ in v]
            ys += v
            mm.append(float(np.mean(v)) if v else np.nan)
        axes[0].scatter(xs, ys, s=18, color=colors[j], edgecolor="k", linewidth=0.3, label=short, alpha=0.8)
        axes[0].plot(list(te), mm, "-", color=colors[j])
    axes[0].set_xlabel("resolved task_execution_efficiency")
    axes[0].set_ylabel("mean seconds per executed action (per run)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Mean duration by action type", fontsize=9)
    axes[0].grid(alpha=0.3)
    for j, (k, lab) in enumerate((("dur_n_transport", "completed transport actions"), ("taskwork_actions", "taskwork actions (all)"))):
        mm = []
        for lvl, c in te.items():
            v = vals(c, k)
            axes[1].scatter([lvl + (j - 0.5) * 0.03 + rng.uniform(-0.01, 0.01) for _ in v], v, s=22, edgecolor="k", linewidth=0.3, color=colors[j], alpha=0.8, label=lab if lvl == 0.1 else None)
            mm.append(float(np.mean(v)) if v else np.nan)
        axes[1].plot(list(te), mm, "-", color=colors[j])
    axes[1].set_xlabel("resolved task_execution_efficiency")
    axes[1].set_ylabel("count per run")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Emergent action-mix shift (confound for pooled duration readouts)", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Fig 4. task_execution_efficiency vs observed task durations", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig4_duration.png", dpi=130)
    plt.close(fig)

    # Fig 5 memory
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    mem = {0.2: "R_memory_0.2", 0.6: "S_memory_0.6", 1.0: "A_neutral"}
    for ax, (k, t) in zip(axes, [("dik_acquired", "DIK acquired"), ("dik_forgotten", "DIK forgotten"), ("dik_retained", "DIK retained (final)"), ("dik_reacquired", "DIK reacquired")]):
        mm = []
        for lvl, c in mem.items():
            v = vals(c, k)
            ax.scatter([lvl + rng.uniform(-0.02, 0.02) for _ in v], v, s=25, edgecolor="k", linewidth=0.3, color="tab:purple")
            mm.append(float(np.mean(v)) if v else np.nan)
        ax.plot(list(mem), mm, "-k")
        ax.set_xlabel("memory_reliability")
        ax.set_title(t, fontsize=9)
        ax.set_xticks(list(mem))
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 5. memory_reliability vs forgotten / retained / reacquired DIK (team totals; dots = seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig5_memory.png", dpi=130)
    plt.close(fig)

    # Fig 6 seed variability
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    for ax, key in zip(axes, ["structures_validated", "taskwork_actions"]):
        for si, s in enumerate(seeds):
            ys = [next((r[key] for r in rows if r["condition"] == c and r["seed"] == s), np.nan) for c in conds]
            ax.plot(range(len(conds)), ys, "-o", color=colors[si], alpha=0.8, label="seed %d" % s, markersize=4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=7)
        ax.set_title(key + " - each line is one seed across conditions (seed = shared initial conditions)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=5)
    fig.suptitle("Fig 6. Seed variability: individual seed trajectories across conditions", fontsize=10)
    fig.tight_layout()
    fig.savefig(ana / "fig6_seed_variability.png", dpi=130)
    plt.close(fig)

    # Fig 7 pathology heatmap
    mat = np.array([[sum(r[f] for r in rows if r["condition"] == c) for f in FLAGS] for c in conds])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=max(5, mat.max()))
    ax.set_xticks(range(len(FLAGS)))
    ax.set_xticklabels([f.replace("flag_", "") for f in FLAGS], rotation=40, ha="right")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Fig 7. Flagged pathological runs (count of 5 seeds) by condition")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(ana / "fig7_pathology.png", dpi=130)
    plt.close(fig)

    # Fig 8 trajectories (bonus)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, c in enumerate(conds):
        ts = [100, 200, 300, 400, 500, 600]
        ym = [np.mean(vals(c, f"validated_t{t}")) for t in ts]
        ax.plot(ts, ym, "-o", label=c, color=plt.cm.tab20(ci % 20), markersize=3)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("validated structures (mean over seeds)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title("Fig 8. Validated-structure trajectory by condition")
    fig.tight_layout()
    fig.savefig(ana / "fig8_trajectories.png", dpi=130)
    plt.close(fig)

    # tables
    with open(ana / "report_tables.md", "w", encoding="utf-8") as f:
        f.write("## Condition means (sd) over seeds\n\n| condition | validated | needs_repair | taskwork | unsol.prov | requests | extern | coord ev | backup share | consults | repairs ok | late WAIT | forgotten |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for c in conds:
            def ms(k, d=1):
                v = vals(c, k)
                return f"{np.mean(v):.{d}f} ({np.std(v, ddof=1):.{d}f})" if len(v) > 1 else "-"
            f.write(f"| {c} | {ms('structures_validated')} | {ms('projects_needs_repair')} | {ms('taskwork_actions',0)} | {ms('unsolicited_provision_attempts')} | {ms('request_attempts')} | {ms('externalization_episodes')} | {ms('coordination_events')} | {ms('backup_share',2)} | {ms('consult_executed')} | {ms('repair_successes')} | {ms('late_wait_share',2)} | {ms('dik_forgotten')} |\n")
        f.write("\n## Flagged runs (count of seeds)\n\n| condition | " + " | ".join(x.replace("flag_", "") for x in FLAGS) + " | any |\n|---|" + "---|" * (len(FLAGS) + 1) + "\n")
        for c in conds:
            f.write(f"| {c} | " + " | ".join(str(sum(r[x] for r in rows if r['condition'] == c)) for x in FLAGS) + f" | {sum(r['any_flag'] for r in rows if r['condition'] == c)} |\n")
        f.write("\n## Manipulation-check readout means by resolved level\n\n")
        for const, key, ms_ in manip_rows:
            f.write(f"- {const} -> {key}: " + ", ".join(f"{l}: {v}" for l, v in ms_) + "\n")
        f.write("\n## Resolved constituents sanity (per condition, from run records)\n\n")
        for c in conds:
            r0 = next(r for r in rows if r["condition"] == c)
            f.write(f"- {c}: " + ", ".join(f"{k[2:]}={r0[k]}" for k in r0 if k.startswith("c_")) + "\n")
    print("runs:", len(rows), "conditions:", len(conds), "->", ana)


if __name__ == "__main__":
    main(sys.argv[1])