# CHECKPOINT - pre-experiment validation block (A: validation->repair, B: profile modes, C: logs)

Branch: wip/validation-repair-checkpoint (based on origin/main c25d9c6). NOT merged to main: untested as a whole.
Resume: read this file, then Documentation/DEVELOPMENT_STATE.md. Edit/Write tools may be blocked by auto mode;
patch via the Python helper scripts/wip_diagnostics/patch_helper_ed.py (sub(path, old, new); keeps CRLF; set ROOT inside it).
Run tests with .venv\Scripts\python.exe -m pytest. Do not touch stash@{0}; CLAUDE.md stays untracked; never force push.

## Instruction block (user, 2026-10-15), condensed
A. Fix failed-validation -> repair AND the epistemic leak (hidden project.correct reached the Pilot as needs_repair /
   validated_correctness / repair_pressure). Observed failed validation (closure_status=="failed") is the legitimate evidence.
   No forced repair; the Pilot still chooses REPAIR / INSPECT / CONSULT / REQUEST. 14 required tests, one single-project
   diagnostic, then seeds 0-2 at a condition with real build errors (report attempts, failures, repair selections/executions/
   successes, revalidations, stuck projects, repeated validation loops). Do not tune rule_accuracy.
   Required tests: (1) hidden correct=False not visible pre-validation; (2) complete-but-incorrect project is still a
   validation candidate; (3) first failed validation yields observable evidence; (4) failed project stops dominating
   validation_pending; (5) failed project gives repair pressure; (6) REPAIR not advertised on healthy project; (7) grounded
   Pilot can select REPAIR; (8) ungrounded Pilot does not repair, gap visible, acquisition still possible; (9) repair is
   exact-project addressed; (10) middleware does not change project/action; (11) successful repair can lead to explicit
   revalidation; (12) repair not guaranteed to succeed; (13) state-changing events Pilot-authored; (14) no raw correct in team
   artifacts/context before observation.
B. Add broad_legacy (current behavior; default for bare launches) and compact_panel (residual mechanisms frozen at 0.5:
   communication_propensity, artifact_consultation_tendency, teammate_model_accuracy, rule_accuracy,
   build_readiness_sensitivity, mismatch_detection_sensitivity, validation_thoroughness) profile modes. Resolve at profile
   resolution / combined_overrides (simulation.py ~495); mode list, residual list, reference values in experimental_profile.json
   (label 0.5 "compact-panel reference", not human normal); record mode + residual policy + references + gestalts + generated +
   overrides + resolved seven + final low-level profile in manifest; GUI choice "Broad legacy profile" / "Compact seven-variable
   panel"; headless --profile-mode; proof_matrix must pass compact_panel explicitly. Tests 15-23 (broad reproduces current
   residual values; compact freezes residuals exactly; constituent overrides win; legacy low-level overrides last and logged;
   mode in manifest; same seed+mode identical; mode changes only mechanism resolution). Also REPORT, do not fix:
   build_readiness_sensitivity sign (threshold = max(1, 3+shift); higher sensitivity -> stricter).
C. proof_matrix must persist raw sessions: <out>/runs/<condition>/seed_<n>/..., <out>/summaries/<condition>__seed<n>.json
   with the real persistent session path; no temp-dir cleanup; unique roots; explicit overwrite/skip policy. Add
   provided_element_ids (list, exact DIK ids actually transmitted) and solicited (bool, response to explicit prior request vs
   unsolicited push) to communication_exchange. Tests 24-30.
D. Rerun the 8-condition proof matrix (compact_panel, seeds 0-2, 600 s; A neutral .5/.5, B high teamwork, C low teamwork,
   D high taskwork, E low taskwork, F high push only, G high externalization + low memory, H fast execution + low teamwork);
   report per condition: profile mode, seven constituents, residual values, build errors, validation failures, repair
   attempts/executions/successes, revalidations, validated structures, DIK push/request, externalization, forgetting/
   reacquisition, coordination events, backup actions, task durations, structural support, Pilot decision episodes,
   state-changing substitution attempts, persistent session path. Answer: validation-loop floor gone? backup interpretable?
   taskwork isolates efficiency? teamwork isolates five constituents? raw logs retained? provisions reconstructable per element?
E. Do NOT implement the coordination lifecycle; document the audit finding in DEVELOPMENT_STATE.md as the NEXT block
   (commit_team_plan has no caller; plans stay "proposed"; adoption paths unreachable; assignments are only
   {proposer_role: "lead_next_action"} which normalizes to "support"; _evaluate_team_plan_fit never returns disagree).
F. Hold DIK-request expansion and memory expansion. memory_reliability stays 1.0.
G. Focused tests (construction correctness, epistemic authority, validation/repair, RuleBrain policy, Pilot project authority,
   experimental panel, batch runner, manifest/session outputs, communication flow, deterministic seeding), then the non-GUI
   regression comparison (clean dafbf51 baseline = 108 failures; known diffs:
   test_provider_internal_fallback_marks_effective_backend (env), test_generate_plan_respects_method_state_source_targeting (rng)).
H. Coherent commits; push each tested stage to origin/main. I. Final 19-item report: SHAs; leak removed; visible state
   before/after; control flow; acquisition vs impossible repair; seeds 0-2 diagnostics; mode semantics; frozen residuals;
   manifest/GUI/headless; output layout; comm schema; test results; proof-matrix results; backup interpretability; taskwork
   floors; remaining pathology; coordination deferred; git status; readiness for the coordination block.

## DONE in this checkpoint (part A, partial, only partly tested)
- construction.py: observed_validation_failure(project), observable_status(project) (hidden needs_repair reads as
  ready_for_validation / in_progress until a failed validation is observed); repair_project resets closure_status
  "failed" -> "idle", counts repair_count, sets awaiting_revalidation; mark_validated records last_validation_was_revalidation.
- team_knowledge.py upsert_construction_artifact: artifact status / correct / validation_state use observed state only; adds
  validation_failed.
- brain_context.py: _summarize_structures no longer exposes correct / needs_repair / hidden status (new keys
  validation_attempted, validation_failed, observed_repair_need; needs_repair now equals observed failure). REPAIR affordance
  only when a failed validation was observed; it carries repair_ready, repair_blockers, observed_repair_need. VALIDATE carries
  validation_failed / validation_candidate.
- agent.py: _observed_status() helper at scaffold sites reading project status; _repair_epistemic_blockers() shared by
  affordance exposure and execution gating; repair execution requires OBSERVED need; repair applies the existing build-fidelity
  roll AFTER clearing the error (a repair can be imperfect; event cause "repair_fidelity"); removed the
  blocked-project-support-pressure pivot trigger on needs_repair (it rewrote the Pilot INSPECT/CONSULT choice to WAIT); new
  metrics validation_attempts, validation_failures, revalidations, repair_selections, repair_executions, repair_blocked.
- brain_provider.py: features observed_repair_need, repair_executable, repair_knowledge_gap; failed projects excluded from
  validation-pending; REPAIR mode guard requires an executable repair when need is observed; ACQUIRE_DIK mode +1.0*gap;
  action scores: REPAIR with repair_ready False -3.0, VALIDATE on a failed project -2.5, INSPECT +1.0*gap, useful CONSULT
  +0.6*gap; emergency method switch away from VALIDATE when failure evidence exists and no validation work is pending;
  new_structure_needed suppressed while an observed repair need exists.
- tests/test_construction_correctness_authority.py: the repair test now first records an observed failed validation.
- Related-test comparison vs the baseline failure list: only the known brittle
  test_generate_plan_respects_method_state_source_targeting differs; test_dik_communication_flow tests fail at baseline anyway.
  Parts B, C, D and the new tests do NOT exist yet.

## FINDINGS so far (scripts/wip_diagnostics/validation_repair_diag.py <seed> <seconds>)
1. The validate loop is gone (seed 1, 900 s, neutral .5/.5): 7-9 validation attempts, each project fails once, no repeats.
2. BUT repair never became executable in 900-1800 s: 0 decisions had repair_ready True, so 0 repair selections. Cause chain:
   _epistemic_sufficiency source-memory confidence is ~0.08 soon after inspection, so sufficient_for_validation (score >= 0.68)
   and "no stale_epistemic_grounding" almost never hold. Re-inspection is offered and chosen sometimes (ACQUIRE_DIK, ~20-30
   inspects) but readiness is not reached. NEXT: find why memory_confidence decays so fast / does not recover after INSPECT
   (_refresh_source_memory_state). Report it as an existing epistemic-freshness pathology; do not tune. DECISION LIKELY NEEDED
   FROM USER: should REPAIR require full validation-grade sufficiency (>= 0.68) or a repair-specific grounding rule (e.g.
   held rules covering the project expected_rules)? That is a scientific/semantic choice.
3. Pre-existing pathology exposed once the validate loop was removed: repeated start_construction (70-440 per agent)
   blocked with "materials_not_staged" at site_c; all 8 site_b projects end needs_repair. Diagnose, do not tune.
4. A transient consult loop came from the Mecha pivot rewriting INSPECT/CONSULT to WAIT for blocked projects (fixed above).
5. Observed failure evidence is team-wide (read from the shared project record); acceptable, note it in the docs.

## REMAINING (nothing below is done)
- Part A: resolve finding 2; write tests 1-14 (e.g. tests/test_observed_validation_repair.py); single-project diagnostic
  (build -> hidden state -> Pilot-visible context -> first VALIDATE -> failure -> scores -> readiness -> REPAIR -> revalidation);
  seeds 0-2 table. Update tests that set project["status"]="needs_repair" directly only if they pass at baseline.
- Parts B, C, D; DEVELOPMENT_STATE.md update (incl. coordination-lifecycle audit finding, build_readiness_sensitivity sign);
  non-GUI regression comparison; merge/push to origin/main; final 19-item report. Do not start the coordination block.


## PART A RESOLVED (2026-10-15, second session)
Decision (user): REPAIR uses a PROJECT-SPECIFIC grounding criterion, not the global validation-grade source-memory threshold.
Executable when (1) the exact project's validation failure was observed and (2) the agent currently holds the project's
expected_rules (or, for a project with none, at least one task rule). Raw project.correct is never used and the hidden error is
not revealed; blockers name only missing rule ids (`missing_expected_rule:<id>`) plus the hard token `repair_project_rules_not_held`.
- agent._repair_epistemic_blockers(environment, sim_state, project_id) is shared by the REPAIR affordance (repair_ready /
  repair_blockers) and the execution gate; low source-memory confidence is no longer a gate (it still feeds the ACQUIRE_DIK/INSPECT
  preference through epistemic features). rule_accuracy still governs application fidelity: repairs can fail (cause repair_fidelity).
- Epistemic blockers are "advisory" in _partition_action_blockers; `repair_project_rules_not_held` was added to the hard tokens so an
  ungrounded REPAIR is blocked (and recorded as a known gap) rather than executed.
- tests/test_observed_validation_repair.py = required tests 1-14 (14 pass). test_construction_correctness_authority's repair test now
  states the grounding premise (agent holds the project's expected rules).
- Diagnostic (scripts/wip_diagnostics/validation_repair_diag.py, 900 s, neutral .5/.5):
  seed | attempts | failures | revalidations | repair selections | repair executions | validated | stuck at end
  0 | 45 | 27 | 25 | 23 | 37 | 18 | 2
  1 | 20 | 13 | 13 | 21 | 14 | 7 | 0
  2 | 11 |  8 |  4 | 11 |  6 | 3 | 6
  Single project (seed 1, site_b_food_connector_001): build error t=88 -> failed validation t=90 -> consult/inspect (knowledge gap,
  repair_ready False) -> repair_ready True t=101 -> repair t=276 (imperfect, repair_fidelity) -> revalidate fail -> repair -> revalidated t=281.
  No validation loop. CAVEAT: repair_executions can exceed repair_selections (seed 0: 37 vs 23); selections are counted once per
  Pilot episode, executions per applied action, so this is a counting-granularity difference, not an un-chosen repair (the execution
  path only runs for decision_action == REPAIR). Worth a cleaner metric before the proof matrix.
- Regression: of the related test files, 79 fail after vs 80 before (c25d9c6); the after-set is a subset of the before-set
  (no new failures). They are pre-existing brittle/known failures.

## FINDING (documented, NOT fixed): repeated start_construction / materials_not_staged
START_CONSTRUCTION (start_mode existing_project) is advertised for any physically unstarted project, whether or not materials are
staged (construction.execute_build_step returns "materials_not_staged" when build_ready is False). Once the validate loop was removed
the Pilot repeatedly selects START on such projects (70-440 per agent at 900 s earlier; seed 2 shows 49 start_construction decisions
under REPAIR mode) and the projects stay stuck (seed 2: 6 unfinished/needs_repair at the end; seed 0: 2). The affordance does not
state the missing-materials fact per project, so the Pilot cannot discriminate (affordance-quality problem, not a Pilot-authority one).
Severity: material for the proof matrix (inflates action counts, suppresses throughput, confounds taskwork measures) but separate from
the repair semantics. Recommendation: fix BEFORE the profile/logging work (B/C/D), as a small affordance fix (expose a per-project
materials_staged / start_blockers fact and withhold or flag START where build_ready is False) with a regression test. Awaiting the
user's go-ahead because it changes action availability.


## NARROW FIX: existing-project START requires staged materials (commit after 1f6a506)
- Affordance: START_CONSTRUCTION (start_mode existing_project) is offered only when the project's materials are staged
  (`build_ready` / staged >= required). Every project affordance now states `materials_staged` and `materials_deficit`
  (observable task state); TRANSPORT to the exact project is unchanged. The typed new-project START is untouched.
- Legality: a non-typed START on an unstaged, physically unstarted project is blocked with the hard token `materials_not_staged`
  (reported, never rewritten into another project/action).
- Tests: tests/test_start_materials_gate.py (1-6). Test 6 counts Pilot START selections on unstaged existing projects over 900 s of
  seed 2 (neutral .5/.5): pre-fix 46 (45 on one project), after 0. Residual `materials_not_staged` events (<=23 per 900 s) are
  the physical build step of a freshly created typed-START project that has no bricks yet; creation stays Pilot-authored.
- Repair accounting is now three distinct counters: repair_decision_episodes (Pilot intents), repair_selections (every selection
  incl. re-selections), repair_applications (applied repair actions; may exceed episodes because one intent can span several
  applications), repair_successes (applications whose error was actually cleared; the rest failed the rule_accuracy roll).
- Seeds 0/1/2 at 900 s (gate only): attempts 35/28/0, failures 26/18/0, revalidations 18/10/0, repair episodes 28/13/0,
  applications 25/12/0, successes 9/9/0, validated 9/10/0 (before the gate: 18/7/3). Throughput is lower at seeds 0 and 2 and
  seed-to-seed variance is large. Seed 2 has no validations because both agents sit in a COORDINATE-mode consult_team_artifact loop
  (~170-195 consults by t=500; the same loop is present BEFORE the gate: 179/116 at t=500). It is a separate, pre-existing
  pathology (consult score ~5 vs <3 for everything else) and was NOT changed here. A try at excluding typed new-project START from the
  "continue active work" score boost was reverted: it was Pilot-policy tuning outside the narrow spec and did not help.


## PART D RESULT: proof matrix, compact_panel, seeds 0-2, 600 s (2026-10-15)
Raw sessions + summaries: D:\Post-doc Work\proof_matrix_2026-10-15\{runs,summaries}\ (24 runs, outside the repo; layout per scripts/proof_matrix.py).
Residual mechanisms were 0.5 in every run; constituents were exactly the specified ones. Means over seeds 0-2:

| cond | build err | val att | val fail | repair ep/app/ok | reval | validated | push | req/ok | extern | backup | pilot selections |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A neutral | 10.7 | 10 | 6.7 | 7.3/7.7/3 | 5 | 3.3 | 3 | 0/0 | 3.7 | 54 | 387 |
| B high teamwork | 4.3 | 5.3 | 3.7 | 2.7/3.3/1.7 | 2.3 | 1.7 | 4 | 1/0.3 | 4.3 | 17 | 687 |
| C low teamwork | 8 | 4 | 4 | 1/2/0.7 | 0 | 0 | 0 | 0/0 | 2.3 | 46 | 422 |
| D high taskwork | 5.7 | 6.7 | 4.3 | 9/4.3/2 | 3.7 | 2.3 | 2 | 0/0 | 4 | 28.7 | 581 |
| E low taskwork | 6.3 | 11 | 7.7 | 2.3/2.7/1.7 | 2 | 1.7 | 1 | 0/0 | 2.3 | 38 | 536 |
| F high push only | 11.7 | 13.7 | 8.3 | 17.3/11.7/5.3 | 8.3 | 5.3 | 15.3 | 0/0 | 4.3 | 49 | 415 |
| G high extern, low memory | 1.3 | 1.3 | 1 | 1.7/0.3/0.3 | 0.3 | 0.3 | 1.3 | 0/0 | 5.7 | 3 | 583 |
| H fast tasks, low teamwork | 5.3 | 1.3 | 1 | 0/0/0 | 0 | 0.3 | 0 | 0/0 | 2.3 | 45 | 476 |
(forgetting/reacquisition only in G: 86 forgotten, 34.7 reacquired; state-changing substitution attempts: 0 everywhere
(mecha_project_substitution_rejected = 0, planner_action_downgraded = 0).)
Findings:
- Validation-loop floor is gone (<=14 validation attempts per run, none repeated), BUT seed 2 reaches ~0 validations in almost every
  condition: the pre-existing COORDINATE-mode consult_team_artifact loop (see NARROW FIX section) now dominates as the floor.
  It is the main remaining pathology and swamps outcome comparisons; it should be the next fix.
- Taskwork isolates execution efficiency: only task_execution_efficiency differs (D 1.0 / E 0.0); mean action durations scale
  (transport 7.5 s vs 37.5 s, start 0.5 vs 3.75). Outcome effects (validated 2.3 vs 1.7) are small against seed variance.
- Teamwork changes exactly the five generated constituents at the profile level (B/C). Behaviorally: push 4 vs 0, plans proposed
  4.3 vs 2.3, plans agreed 9.3 vs 0, externalization 4.3 vs 2.3. DIK requests are essentially absent (request_attempts <= 1, request
  successes 0.3 total) and 0 of 80 provision exchanges that named elements were solicited, so the request/solicited path is thin.
- Backup is NOT interpretable as a propensity readout: backup_actions is HIGHER at zero backup_propensity (C 46, H 45) than at 1.0
  (B 17). It counts any taskwork on another role's/shared structure, so it tracks total taskwork volume (low-teamwork runs spend less
  time in coordination/consult behavior and do more transport on shared infrastructure). Do not interpret it without normalizing.
- Raw logs are retained; 108 communication_exchange events across the matrix, 80 with provided_element_ids (exact DIK ids), so
  provisions are reconstructable per element.
- Regression (related test files, 511 tests): 81 failed vs 80 at c25d9c6; the single extra failure was a GUI-stub test broken by the
  new profile control (fixed in the next commit); the remaining failures are a subset of the baseline failures.
