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

---

# Addendum: final controller-contract cleanup (supersedes the Procedural-validation and ambiguous-feedback limitations above)

Code commit: see `git log` ("Controller-contract cleanup ..."); the frozen SHA is the commit that contains this addendum (recorded in the hand-off message and `git log -1 -- <this file>`).

## Changes
1. `RuleBrain._policy_core` and the Procedural fallback (`ProceduralBaselinePilotAdapter.choose_fallback_action`) now take their candidates from `controller_context.partition_affordances` - the same executable/blocked boundary that builds the LLM `executable_actions`. Blocked affordances (not-ready VALIDATE, not-ready REPAIR, inaccessible private INSPECT) stay in the context packet as information and cannot be selected. No VALIDATE special case; no change to preferences among executable actions.
2. Validation outcomes seen by Pilots: `status: completed`, `execution_status: completed`, `validation_result: passed|failed` (was `status: succeeded` for any executed validation). Reasons (missing rules, support detail) stay only in the authoritative event log; `is_valid` / `missing_expected_rules` never reach the prompt (0 occurrences in A0 prompts). Validation rule and outcome unchanged. Other actions' outcome records are unchanged.
3. Tests: `tests/test_controller_contract_cleanup.py` (10 tests; 5 fail on the pre-change code): blocked VALIDATE/REPAIR/INSPECT never selected by RuleBrain over 6 modes; selectable once executable; fallback restricted; LLM partition unchanged; failed/passed validation reporting; non-validation outcomes unchanged; result reaches the prompt.

## Regression
Related focused set (45 files incl. authority, connector, gate, pilot-adapter, rule-brain, validation): 379 passed / 35 failed; the same 35 fail on the unmodified code (pre-existing, plus the known flaky rng test `test_generate_plan_respects_method_state_source_targeting`, which fails intermittently on both trees). No new failure.

## Small diagnostic (`Outputs/contract_diag_20261025`; P seeds 0,1,4 + A seed 0, 600 s, same settings)
Blocked-validation selections reaching the gate / refused validations: P0 22->0, P1 40->0, P4 20->0, A0 18->0 (`mecha_action_blocked` 11/20/10/9 -> 0/0/0/0). 0 controller errors; 0 private-source reroutes; substitution events all `applied: false`; task package untouched.
Procedural taskwork (core validated H/G/Gen; before -> after): P0 6 -> 5; P1 5 -> 5; P4 4 -> 2; supported capacity max P0 9 -> 0, P1 6 -> 0, P4 0 -> 0. Single runs per cell with a changed decision/RNG sequence; direction is mixed and N=1, so not interpreted or tuned. Connect selections 11->4, 9->10, 6->2; blocked-project support pressure still applied (152/315/344 events).
A0: Engineer longest identical streak 23 -> 27 (now on `validate_construction` of a failed water connector, not the greenhouse). Every repeat is reported as `validation_result: failed` in the prompt (403 failed / 3 passed records), so the repetition persists despite unambiguous feedback: model behaviour, not an interface ambiguity. A0 also validated 5 core structures (was 1) with 11 repair episodes (was 0).
Host note: A0 waited 1807 s in guard warm-up (desktop applications held ~2.3 GB of VRAM) and then ran normally; wall time not comparable.

## Status
READY for the larger characterization batch. Remaining documented limitations: LLM perseveration on re-validating failed projects; speculative-connector proliferation; truncated history/communication/held-DIK sections in LLM prompts; Procedural taskwork changed (mixed) by removing wasted blocked selections - characterize, do not tune; N=3 / host memory pressure caveats.
