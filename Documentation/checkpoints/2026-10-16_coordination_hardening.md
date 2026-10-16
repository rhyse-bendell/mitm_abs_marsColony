# CHECKPOINT - coordination hardening block (2026-10-16)

Base: f2ba5ae (minimal team-plan lifecycle). Scope fixed by the researcher: controller-neutral plan context, assignment follow-through
semantics, one active plan, epistemic boundary, missing-knowledge exposure, characterization against f2ba5ae. NOT in scope: backup
operationalization, request/push ecology, any coefficient tuning, any LLM/Ollama implementation. Success is not mission performance.

## What changed
1. Controller-neutral plan context (`team_plan_context/1`, `Agent.team_plan_context`, exposed as `team_state["team_plan_context"]` and
   `task_context["team_plan_context"]`; see Documentation/generic_brain_contract.md). Built only from the agent's acquired copy of a plan
   (`team_plan_state["known_plans"]`: consulted / received by message / authored). The live store is no longer exposed to the Pilot:
   unread plans are an existence flag only; `team_shared_knowledge`, `active_team_plan`, `team_plan_summaries`, `my_role_assignment`,
   `plan_readiness`, artifact summaries and `recent_shared_updates` are built from the agent's own copy or have team_plan_* lifecycle chatter removed.
   Communicated plans (TPP) are the sender's own copy, not the store's. The Procedural Baseline's `team_plan_*` features use the same context.
2. Assignment lifecycle by task-level success criterion (`Agent._assignment_criterion_for`): transport = material condition met
   (`resource_complete` / delivered >= required); repair = the OBSERVED failure no longer stands; validate = project validated, or a validation
   attempted after the assignment was taken up; build/continue = physically complete; site-targeted/other = one aligned action (compat).
   A matching completed action (movement sub-actions excluded) is PARTIAL PROGRESS (`team_plan_aligned_action_completed`, `aligned_actions`);
   the assignment retires only when its criterion is met (also by someone else's work), it is blocked (2 decisions without the legal action),
   invalidated (criterion already satisfied at adoption), superseded or expired. New distinct events: `_superseded`, `_expired`, `_invalidated`
   (previously all were `team_plan_assignment_blocked`). Pilot deviation is unchanged: a preference, logged, never forced.
3. One active plan. `TeamKnowledgeManager.propose_team_plan` refuses a NEW plan while a proposed/committed plan is live unless the proposal names
   it in `supersedes_plan_id` (explicit supersession, immediate, logged `explicit_supersession_by_proposal`); refusals are logged
   (`team_plan_proposal_refused`). Task policy `team_plan_policy.single_active_plan` (engine default true). A plan leaves the live states as
   `completed`/`blocked` when every concretely assigned role has reported an outcome (`record_assignment_outcome`), or `expired`. A
   review never replaces a plan by itself: an agent whose part is done (or who had none) lets the plan stand; only a blocked assignment or an
   expired plan makes `revision_due`, and revision may supersede only the plan the agent adopted. Because externalising also carries the author's DIK, a
   refused competing plan still attaches the author's plan-relevant DIK to the EXISTING plan artifact (`_contribute_to_live_plan_artifact`; plan content
   untouched). Without this, refusals silently removed artifact-mediated knowledge sharing (first comparison run: validated structures fell in every condition).
4. Missing knowledge. An assignment the agent cannot ground stays open as `ungrounded` (not retired), visible in the context with `grounding_reason`
   and `recognized_gap_tokens` (the missing_expected_rule ids already stated on the repair affordance, recorded via `_record_known_gaps_from_blockers`);
   no knowledge is granted; it grounds later if the agent's own acquisitions make the action legal (`team_plan_assignment_grounded_later`).
5. Measures: `final_measures.team_plan_process` gained plan exits, committed-plan seconds, agent-seconds aligned, assignments credited with own aligned
   work, refusals, superseded/expired/invalidated/grounded-later counts; `taskwork_completed` now counts completed taskwork decision_actions (move_to excluded).

## Characterization (coordination 0.1 / 0.5 / 0.9, seeds 0-7, 600 s, compact_panel, teamwork = taskwork = 0.5; means over 8 seeds)
Old = f2ba5ae, new = this block. Descriptive only. The old `assignments_completed`/`taskwork_aligned*` were inflated by movement completions and one-action
retirement and are not like-for-like; blocked(legacy-equivalent) = blocked + superseded + expired + invalidated in the new tree.
| measure | 0.1 old/new | 0.5 old/new | 0.9 old/new |
|---|---|---|---|
| proposals | 3.4 / 1.3 | 4.8 / 2.3 | 8.1 / 3.5 |
| proposals refused (single-plan rule) | - / 2.3 | - / 2.8 | - / 6.5 |
| committed plans | 0 / 0 | 4.3 / 1.9 | 6.5 / 3.0 |
| plans superseded (supersession rate) | 0 / 0 | 2.5 (0.51) / 0.1 (0.04) | 3.8 (0.44) / 0.4 (0.11) |
| plans expired | 3.3 / 1.3 | 2.1 / 2.0 | 3.3 / 2.4 |
| adoptions (distinct plan-agent) | 0 / 0 | 10.5 / 5.5 | 17.3 / 8.4 |
| grounded assignments (distinct) | 0 / 0 | 6.6 / 3.6 | 10.6 / 6.0 |
| assignment start rate | - / - | 0.43 / 0.57 | 0.41 / 0.59 |
| assignment completion rate | - / - | 0.17 / 0.38 | 0.19 / 0.38 |
| completed with own aligned work | - / 0 | - / 0.5 | - / 0.9 |
| deviations | 0 / 0 | 20.6 / 13.1 | 29.6 / 24.0 |
| blocked (legacy-equivalent) | 0 / 0 | 6.3 / 2.9 | 10.3 / 4.6 |
| mean agents aligned while a plan is active | - | 1.44 / 1.42 | 1.51 / 1.67 |
| agent-seconds aligned / committed-plan seconds (new only) | - | 215 / 167 | 326 / 228 |
| validated structures (not a success criterion) | 6.0 / 7.1 | 6.5 / 7.8 | 6.4 / 5.9 |
| probable stalls (runs of 8) | 0 / 0 | 0 / 0 | 0 / 0 |
| middleware rewrites (total, 8 runs) | 1 / 0 | 1 / 0 | 0 / 0 |
Interpretation: low/neutral/high coordination stay ordered (no commitment at 0.1; proposals, commitments, adoptions and grounded assignments rise with
coordination); supersession churn fell about an order of magnitude; follow-through per grounded assignment rose; strictly plan-aligned taskwork is small
(about 0.7 aligned completions per run) because the Pilot's plan_assignment term is only a preference (not tuned). About half of the assignments of a
committed plan still end by plan expiry (committed_plan_ttl_s 90 s).

## Limitations / left out
- A plan learned only by message is known to the Pilot but is adopted (assignment taken up) only after the artifact is consulted (existing rule, unchanged).
- One agent's blocked assignment can still make a revision proposal that supersedes the plan for the others (allowed by the specification; explicit and logged).
- A declined/ungrounded acknowledgement still makes receivers rewrite that role's store assignment to `support_consultation` (existing TPA behaviour); the agent
  keeps its own copy and can still ground it later.
- Connect-structure assignments are never proposed (connect affordances carry no target_id); epistemic_refresh assignments have no task-level criterion.
- START assignments on a site (not an existing project) use one aligned action as criterion; the structure type is not part of an assignment.
