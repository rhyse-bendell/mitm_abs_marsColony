# Core simulator final checkpoint (2026-10-25) - READY for the larger characterization batch

Supersedes `2026-10-22_core_simulator_final_diagnostic_CHECKPOINT.md` (which stopped at 11/12 runs, NOT READY pending review).
No simulator code was changed in this block; only the missing run, the extractor (output tooling under the git-ignored Outputs/) and this document.

## Diagnostic
- Output: `Outputs/final_diag_20261022` (git-ignored). P/A/B/C x seeds 0,1,4, 600 sim-s, compact_panel, Teamwork=.5/Taskwork=.5, qwen3.5:9b for LLM roles.
- 12/12 complete, 0 controller errors, 0 malformed/failed decisions. C seed 4 was re-run from scratch (the killed partial is kept at `runs/C/seed_4_partial_killed`).
- C seed 4 resume: the headless guard throttled 8x (`gpu_vram_headroom_lost`; waits <= 1809 s, 14456 s total; host was low on memory), then recovered and the run completed normally (decisions 3 s each afterwards). Wall clock for that run is therefore not comparable (15863 s); inference latency columns exclude waits.
- Extractor fix: repair counters now use `construction_repair_episode` / any `*repair*` selection (previous version reported 0).

## Matrix (core validated = houses/greenhouses/generators; conn = connector projects started (speculative); sup = max supported capacity)
cond seed: decisions(LLM) / core / H,G,Gen / conn(spec) / sup cap / max streak / repair eps / gate-refused validations
- P0: -/6/3,1,2/2(0)/9/9/9/22   - A0: 97/1/0,1,0/5(4)/0/23/0/18   - B0: 147/2/0,0,2/11(10)/0/5/3/0   - C0: 212/0/0,0,0/15(14)/0/7/0/0
- P1: -/5/3,1,1/2(0)/6/11/14/40  - A1: 69/4/0,0,4/2(2)/0/8/4/8     - B1: 163/7/0,7,0/0(0)/0/17/8/0     - C1: 259/7/0,6,1/6(4)/0/15/17/0
- P4: -/4/2,0,2/2(0)/0/6/6/20    - A4: 98/10/1,4,5/3(1)/3/8/24/2   - B4: 184/9/2,3,4/2(0)/6/9/27/0     - C4: 273/2/0,0,2/20(20)/0/10/5/0
First-attempt valid: A 91/97, 61/69, 96/98; B 146/147, 163/163, 179/184; C 210/212, 258/259, 262/273. Retries 0-11 per run. Mean inference 3.1-4.5 s.
Foods/water connectors validated: P 1/1,1/1,0/0; A 0/0,0/0,1/1; B 0/0,0/0,1/1; C 1/0,1/1,0/0.
Longest no-taskwork window: 27-153 s (windows >=120 s: A0, C4 only). DIK requests, plans, externalization: low and similar across conditions (see analysis/runs.json).
Matched-seed deltas (core validated): A-P: -5,-1,+6; B-P: -4,+2,+5; C-P: -6,+2,-2; B-A: +1,+3,-1; C-B: -2,0,-7. No significance testing at N=3; the sign is not consistent across seeds.

## Readiness questions
1. Known-blocked validations selectable? LLM: no (listed under blocked_actions; 0 reached the gate in 12 runs). Procedural Baseline: it still selects them (gate-rejected 20/40/20 per run) and the gate rejects without state change; this is baseline policy inefficiency, not a simulator defect, and it handicaps P.
2. Private-source silently retargeted? No (0 reroutes, 12 runs).
3. Substantive Pilot choices silently substituted? No: 63 `mecha_strategic_substitution_rejected` events (P/A/B), all `applied: false`.
4. Generator -> Greenhouse water support works? Yes (supported greenhouses > 0 in P0, P1, A4, B4, C1).
5. Endpoint-less connectors functional? No: endpoint-less validation is refused (`connector_has_no_valid_endpoint_link`); every validated connector had links; 0 wrong-endpoint attempts.
6. Guard safe? Yes: 8 throttles in C4 recovered and resumed; no throttles in the other 11 runs.
7. Truncation distorting? Tier-1 (actions, blockers, projects) intact in 11/12 runs (one shrink to 6 projects in one C0 decision at t=597 with 24 projects). Tier-2/3 history, shared-update and part of held-DIK sections are heavily omitted in 20-175 decisions of B/C runs. Documented controller-context limitation, applies to all LLM roles equally; not tuned.
8. Connector spam persists? Yes, as legal behaviour (up to 20 speculative connector projects in C4, consuming site slots). Legal START affordance; no source support yet for a prerequisite, so deliberately not changed.
9. Perseveration persists? Yes in some LLM runs (A0 23x same validate of a failed greenhouse; B1 17; C1 15). World state is truthful (prompt shows `needs_repair`/`validation_failed`; repair blocked by the Engineer lacking R_GREENHOUSE_SUPPORT_DEPENDENCY, i.e. distributed expertise). Model behaviour, not a simulator fault. Note: outcome status "succeeded" for a validation that returned "incorrect" is ambiguous feedback; candidate for later clarification, not changed.
10. Traces interpretable? Yes.
11. Remaining contaminating defect? None found.

## Limitations to carry into characterization
Procedural Baseline selects gate-blocked validations; speculative-connector spam; validation re-attempts on unchanged failed projects are legal no-ops; truncated history sections in LLM prompts; host memory pressure can trigger long guard waits (wall time not comparable); N=3.

## Not done (by instruction)
No larger batch, no prompt/coefficient tuning, no human-testbed work. feature/human-participant-testbed untouched (b1d728b); stash@{0} untouched; CLAUDE.md untracked.
