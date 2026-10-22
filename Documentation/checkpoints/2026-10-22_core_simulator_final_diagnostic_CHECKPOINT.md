# Core simulator final diagnostic - CHECKPOINT (2026-10-22)

STATUS: CORE SIMULATOR WRAP-UP PAUSED AFTER FRESH DIAGNOSTIC (11 of 12 runs; matrix INCOMPLETE)

## Git
- local main = fa15e55 (merge commit of the simulator checkpoint branch), origin/main = deb41d7, 8 local commits ahead: LOCAL MAIN NOT YET PUSHED.
- Why: `git push` (checkpoint branch and main) was denied by the Claude Code permission classifier ([Modify Shared Resources]); not retried. The checkpoint branch's 5+1 commits are also unpushed.
- feature/human-participant-testbed untouched (b1d728b). stash@{0} untouched. CLAUDE.md untracked, not committed.

## Regression (head a738e60 vs main deb41d7, non-GUI suite, same ignore list)
main 138 failed/995 passed; checkpoint 136 failed/1045 passed. Failure sets differ only by 2 load-sensitive tests + 1 environmental test (fixed on checkpoint side) and the known brittle rng test `test_generate_plan_respects_method_state_source_targeting` (flaky on BOTH trees alone). No new regression. Focused authority/connector/guard/contract files: 43 passed.

## Work verified in this resumed block (done in commits d0529d3..a738e60, see DEVELOPMENT_STATE.md)
Exhausted/cooldown rerank moved before selection (no post-selection substitution); connector semantics (functional only with valid endpoint link; Generator->Greenhouse water supported; duplicates once); headless resource guard end to end (scripted throttle/recover/resume, cancel, manifest); prompt audit (budget unchanged); outcome_breakdown leads with core structures, connectors separate.

## Diagnostic
- Output: Outputs/final_diag_20261022 (runs/, summaries/, progress.log, analysis/extract.py, analysis/runs.json). Driver: driver.py in that folder.
- 11 of 12 completed, 0 controller errors, 0 timeouts. C seed 4 was killed mid-run at the user's request (partial logs preserved under runs/C/seed_4). Not rerun.
- Guard active in every LLM run (synchronous_headless), 0 throttle events, 0 resource wait. Wall: P 53-118 s, A 365-454 s, B 630-771 s, C ~917 s per run.
- condition seed: validated_core (H/G/Gen) | connector projects (speculative) | supported H/G (max) | max streak | refused validations (events)
  P0: 6 (3/1/2) | 2 (0) | 3/1 | 9 | 22
  A0: 1 (0/1/0) | 5 (4) | 0/0 | 23 | 18
  B0: 2 (0/0/2) | 11 (10) | 0/0 | 5 | 0
  C0: 0 | 15 (14) | 0/0 | 7 | 0
  P1: 5 (3/1/1) | 2 (0) | 2/1 | 11 | 40
  A1: 4 (0/0/4) | 2 (2) | 0/0 | 8 | 8
  B1: 7 (0/7/0) | 0 | 0/0 | 17 | 0
  C1: 7 (0/6/1) | 6 (4) | 0/2 | 15 | 0
  P4: 4 (2/0/2) | 2 (0) | 0/0 | 6 | 20
  A4: 10 (1/4/5) | 3 (1) | 1/4 | 8 | 2
  B4: 9 (2/3/4) | 2 (0) | 2/3 | 9 | 0
  C4: not completed
- Authority: private-source reroutes 0 in all runs; LLM blocked-validate selections reaching the gate 0; no wrong-endpoint attempts. Prompt: est/actual 1.11-1.16, max prompt 6.2k tokens, but truncation (omitted items) occurred in B/C decisions (34-118 per run) - not yet analysed (what was dropped).
- Caveats NOT yet analysed: (1) Procedural runs show 20-40 refused-validation events that mention missing support/endpoint-less connectors (the Procedural Baseline appears to still select validation of endpoint-less connectors; the extractor counts events, not selections - verify); (2) A seed 0 Engineer chose VALIDATE on the same incorrect greenhouse 23 times in a row (model perseveration; no repair); (3) speculative connector proliferation persists in LLM B/C (10-14 speculative connector projects), consuming site slots; (4) LLM conditions mostly validate fewer core structures than P at seeds 0 and 1.

## Preliminary readiness: NOT READY (pending review)
Reason: matrix incomplete; unexamined truncation in B/C; possible Procedural endpoint-less-connector validation selections; connector spam persists in LLM runs (legal behaviour, but needs a decision whether START connector affordance needs a prerequisite). No evidence of authority/guard/connector-semantics defects.

## WHEN WE RETURN
1. Inspect analysis/runs.json and the 11 runs (truncation content, Procedural refused validations, connector spam).
2. Fix only genuine simulator defects revealed.
3. Push local main when permission allows (user can run: `! git push origin checkpoint/controller-neutral-resource-guard-20261017 main`).
4. If clean, freeze/tag the simulator.
5. Run the larger characterization batch.
6. Analyse it.
7. Only after freeze, return to feature/human-participant-testbed.
