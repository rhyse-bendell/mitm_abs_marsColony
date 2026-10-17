"""Compare two characterization batches (same design) on pathology, substitution and manipulation-signature measures.

python scripts/characterization_compare.py <old_batch_dir> <new_batch_dir> [--out <dir>]
Each batch dir must already have analysis/characterization_runs.csv (scripts/characterization_report.py). Descriptive only; success is
judged on artificial-pathology / substitution counts and preserved manipulation signatures, never on outcome counts.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CHECKS = [  # (constituent, short, readout, level->condition-name resolver)
    ("dik_push_propensity", "push", "unsolicited_provision_attempts"),
    ("dik_request_propensity", "request", "request_attempts"),
    ("externalization_propensity", "extern", "externalization_episodes"),
    ("coordination_propensity", "coord", "ev_team_plan_agreed"),
    ("backup_propensity", "backup", "backup_share"),
    ("task_execution_efficiency", "taskeff", "dur_mean_transport"),
    ("memory_reliability", "memory", "dik_forgotten"),
]


def load(batch):
    rows = list(csv.DictReader(open(Path(batch) / "analysis" / "characterization_runs.csv", encoding="utf-8")))
    return rows


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def mean(rows, cond, k):
    v = [num(r, k) for r in rows if r["condition"] == cond and num(r, k) is not None]
    return float(np.mean(v)) if v else float("nan")


def level_conditions(rows, short):
    conds = sorted({r["condition"] for r in rows})
    if short == "memory":
        return {0.2: "R_memory_0.2", 0.6: "S_memory_0.6", 1.0: "A_neutral"}
    lo = next(c for c in conds if c.split("_")[1] == short and c.endswith("_low"))
    hi = next(c for c in conds if c.split("_")[1] == short and c.endswith("_high"))
    return {0.1: lo, 0.5: "A_neutral", 0.9: hi}


def total(rows, k):
    return sum(num(r, k) or 0 for r in rows)


def rewrite_types(rows, col):
    c = Counter()
    for r in rows:
        try:
            c.update(json.loads(r.get(col) or "{}"))
        except ValueError:
            pass
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)
    out = Path(args.out) if args.out else Path(args.new) / "analysis" / "comparison_vs_old"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Characterization comparison\n\nold: `{args.old}`  ({len(old)} runs)\nnew: `{args.new}`  ({len(new)} runs)\n"]

    def row(label, f):
        return f"| {label} | {f(old)} | {f(new)} |"

    lines += ["## Pathology / substitution counts (95 runs each)\n", "| measure | old | new |", "|---|---|---|"]
    lines += [
        row("middleware rewrites of Pilot actions (end-to-end, per Pilot selection)", lambda R: int(total(R, "e2e_rewrites_total"))),
        row("  of which by the documented start-up grounding scaffold (BOOTSTRAP)", lambda R: int(total(R, "e2e_rewrites_bootstrap_total"))),
        row("  of which outside start-up grounding", lambda R: int(total(R, "e2e_rewrites_nonbootstrap_total"))),
        row("strategic substitutions rejected (reported by the guard)", lambda R: int(total(R, "strategic_substitutions_rejected"))),
        row("pivot-level rewrites seen by the batch wrapper", lambda R: sum(sum(json.loads(r.get("rewrite_types") or "{}").values()) for r in R)),
        row("rewrites to WAIT", lambda R: int(total(R, "rewrites_to_wait"))),
        row("probable stall runs (generic process-evidence flag)", lambda R: int(total(R, "flag_probable_stall"))),
        row("probable stall seconds (summed over runs)", lambda R: int(total(R, "stall_seconds"))),
        row("stalled at end of run", lambda R: int(total(R, "stall_at_end"))),
        row("absorbing runs (no taskwork in last 200 s / late WAIT > 0.5)", lambda R: int(total(R, "flag_absorbing"))),
        row("completed inspections", lambda R: int(total(R, "inspect_total_completed"))),
        row("no-gain (repeated) inspections", lambda R: int(total(R, "inspect_no_gain_count"))),
        row("longest consecutive no-gain inspection streak (one agent)", lambda R: int(max(num(r, "inspect_no_gain_max_streak") or 0 for r in R))),
        row("late (t>=400) INSPECT share of executed micro-actions, mean", lambda R: f"{np.mean([num(r, 'late_inspect_share') or 0 for r in R]):.2f}"),
        row("repair starvation runs", lambda R: int(total(R, "flag_repair_starvation"))),
        row("repeated-validation runs", lambda R: int(total(R, "flag_repeated_validation"))),
        row("excess-consult runs", lambda R: int(total(R, "flag_excess_consult"))),
        row("stale closure-owner runs", lambda R: int(total(R, "flag_stale_closure"))),
        row("exceptions", lambda R: int(total(R, "flag_exception"))),
        row("taskwork actions per run, mean", lambda R: f"{np.mean([num(r, 'taskwork_actions') or 0 for r in R]):.0f}"),
        row("validated CORE structures per run, mean (descriptive only)", lambda R: f"{np.mean([num(r, 'validated_core_structures') or 0 for r in R]):.2f}"),
        row("validated connectors per run, mean (auxiliary; kept separate)", lambda R: f"{np.mean([num(r, 'validated_connectors') or 0 for r in R]):.2f}"),
        row("runs with all site slots exhausted", lambda R: int(total(R, "site_slots_exhausted"))),
        row("validated structures, all types incl. connectors (legacy; NOT a performance readout)", lambda R: f"{np.mean([num(r, 'structures_validated') or 0 for r in R]):.2f}"),
    ]
    lines += ["\n## Rewrites by type OUTSIDE start-up grounding (end-to-end, count of Pilot selections whose first attempted action differed)\n", "| type | old | new |", "|---|---|---|"]
    co, cn = rewrite_types(old, "e2e_rewrites_nonbootstrap_json"), rewrite_types(new, "e2e_rewrites_nonbootstrap_json")
    for k in sorted(set(co) | set(cn), key=lambda k: -(co.get(k, 0) + cn.get(k, 0))):
        lines.append(f"| {k} | {co.get(k, 0)} | {cn.get(k, 0)} |")

    lines += ["\n## Manipulation signatures (condition means of the proximal readout)\n", "| constituent | readout | level | old | new |", "|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, (const, short, readout) in zip(axes.flat, CHECKS):
        lv = level_conditions(old, short)
        xs = list(lv)
        mo = [mean(old, lv[x], readout) for x in xs]
        mn = [mean(new, lv[x], readout) for x in xs]
        for x, a, b in zip(xs, mo, mn):
            lines.append(f"| {const} | {readout} | {x} | {a:.3f} | {b:.3f} |")
        w = 0.35
        idx = np.arange(len(xs))
        ax.bar(idx - w / 2, mo, w, label="old", color="tab:gray")
        ax.bar(idx + w / 2, mn, w, label="new", color="tab:blue")
        ax.set_xticks(idx)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(f"{const} -> {readout}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes.flat[-1].axis("off")
    fig.suptitle("Manipulation signatures: old vs new batch (condition means; N=5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "manipulation_signatures_old_vs_new.png", dpi=130)
    plt.close(fig)

    lines += ["\n## Per-condition (validated mean / probable stalls / no-gain inspections / rewrites)\n", "| condition | validated old | new | stalls old | new | no-gain inspects old | new | rewrites old | new |", "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted({r["condition"] for r in old}):
        def agg(R, k):
            return sum(num(r, k) or 0 for r in R if r["condition"] == c)
        lines.append(f"| {c} | {mean(old, c, 'structures_validated'):.1f} | {mean(new, c, 'structures_validated'):.1f} | {int(agg(old, 'flag_probable_stall'))} | {int(agg(new, 'flag_probable_stall'))} | "
                     f"{int(agg(old, 'inspect_no_gain_count'))} | {int(agg(new, 'inspect_no_gain_count'))} | {int(agg(old, 'e2e_rewrites_total'))} | {int(agg(new, 'e2e_rewrites_total'))} |")
    lines += ["\n## Runs flagged as probable stall\n", "| batch | condition | seed | stall s | first stall t | pattern |", "|---|---|---|---|---|---|"]
    for label, R in (("old", old), ("new", new)):
        for r in R:
            if r.get("flag_probable_stall") == "1":
                lines.append(f"| {label} | {r['condition']} | {r['seed']} | {r['stall_seconds']} | {r['first_stall_t']} | {r['stall_dominant_pattern'][:80]} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "comparison.md")


if __name__ == "__main__":
    main()