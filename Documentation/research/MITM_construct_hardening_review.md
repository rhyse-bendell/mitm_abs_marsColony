# MITM construct hardening review

Branch `research/mitm-construct-hardening`, from frozen `main = aba5f04` (main is unchanged and is not merged here).
Scope: correct the genuine defects found by the 45-run construct-validation screening, resolve the operationalization gaps, and
**not** to tune the simulator for task performance. No model coefficient, task semantics, Pilot strategy, or artifact-correctness
meaning was changed. Parts 4 and 5 are design proposals awaiting scientific review; nothing in them is implemented.

The 45-run screening (`Outputs/mitm_construct_validation_20261025_1155/`, git-ignored) is preserved unchanged as the
**pre-hardening scientific baseline**. All "pre" numbers below come from it.

## 1. Summary

| Part | Finding | Disposition | Commit |
|---|---|---|---|
| 1 | Received TKRQ wrote the sender's need into the recipient's `known_gaps`; gaps were not cleared on acquisition | **Fixed** | `c7a9e21` |
| 2 | Efficiency applied twice (`_scaled_duration` and the `duration_scale` hook) | **Fixed** | `dc2f5ee` |
| 3 | Teammate model was a copy of the teammate's hidden rule set; help signals read teammates' hidden `known_gaps` | **Fixed** (minimal evidence-grounded model) | `64253b3` |
| 4 | `externalization_episodes == team_plan_proposals` (45/45) | **Diagnosed; design proposed** | none |
| 5 | Construction correctness independent of held knowledge | **Diagnosed; design proposed; stopped for review** | none |

Commits are on the research branch only; the report commit is listed in section 12. A further finding from the probe, the request-reply channel (H1, section 5), is documented and not patched.

## 2. Part 1: recognized knowledge-gap state

### Diagnosis
* `Agent.receive_message`, `TKRQ` branch, ran `self.known_gaps.update(content)`. A request is the **sender's** information need; it
  became a gap in the **recipient's** own state. The recipient's next outgoing request (`generate_message` sends every
  `known_gaps` entry) then asked for items the sender itself had asked about, and the recipient could "recognize" a gap for an item
  it already held (screening finding F2).
* `known_gaps` tokens (`data:`, `information:`, `rule:`, `missing_expected_rule:`) were cleared only at a few inline acquisition sites
  (data/info/rule receipt, inspect). Acquisition by other routes (derivation, artifact consult, reacquisition after forgetting) left the
  token behind, so agents re-requested what they held (F1: 64 of 121 high-request requests came from agents already holding the item).

### Updated semantics
* `known_gaps` holds **only the agent's own recognized gaps**: derived from its own task-method requirements (`_update_recognized_gaps`)
  and standing construction/repair blockers (`_record_known_gaps_from_blockers`).
* A received request is stored separately in `Agent.received_dik_requests[sender] = {"ids", "time"}` and, as before, as the transient
  `theory_of_mind[sender]["requested_dik_ids"]` that drives the reply. The reply logic and the queued provision are unchanged.
* `Agent._reconcile_known_gaps` runs every tick (after `_update_recognized_gaps`) and again immediately before an outgoing request
  is built. It removes a `data:`/`information:` token when that element is in the agent's mental model, and a `rule:` or
  `missing_expected_rule:` token when the normalized rule is held. Elements that are not held stay open, a forgotten element
  becomes a gap again through the existing recognition step, and standing blockers such as `missing_validation_rule_knowledge`
  are never reconciled here.
* Provenance is untouched (`acquired_by`, `dik_ledger` mode/source/time). Each closure emits `dik_gap_resolved`
  with the ledger acquisition mode. Nothing is granted, and `dik_derivations.csv` is untouched.
* `received_engineering_request` (a communication-context flag) and the help signal in `brain_context` now read received requests,
  not `known_gaps`.

### Tests (`tests/test_construct_hardening.py`)
Received request creates no recipient gap and still queues the correct provision; resolved gaps are removed and unresolved/standing
ones retained; rule and `missing_expected_rule` gaps resolve by any route; reconciliation grants nothing and keeps provenance;
a forgotten element is re-recognized; an agent with a stale gap token does not request what it holds; the request chain no longer
infects the answerer. Mutation check: restoring the old `known_gaps.update` line makes the first and last of these fail.

## 3. Part 2: efficiency → duration, applied once

### Pathway audit
`task_execution_efficiency` (constituent) → mechanism `build_speed` (identity) → hook `action_duration/<action>/duration_scale`
(`config/experimental/mechanism_to_hook.csv`, transform `inverse_duration_scale`, range [0.5, 1.5]) → `Agent._duration_scale`.
`build_speed` was **also** read directly by `Agent._scaled_duration`.

| | formula | neutral (build_speed 0.5) | build_speed 0.1 / 0.9 | range over [0, 1] |
|---|---|---|---|---|
| **old** | `base × (1.25 − 0.75·bs) × (1.5 − bs)` | ×0.875 | ×1.645 / ×0.345 | ×1.875 … ×0.25 |
| **new** | `base × (1.5 − bs)` | **×1.000** | ×1.40 / ×0.60 | ×1.5 … ×0.5 |

(`_scaled_duration` still clamps at 0.2 s and rounds to 3 decimals.) The new factor is the `inverse_duration_scale` hook alone.
`build_speed = 0.5` is the documented neutral/reference value, so authored base durations are exact at the neutral setting.
(An earlier summary quoted "×0.39 to ×1.88"; the exact figures over the tested range are in the table above.)

Applies identically to transport, start_construction, continue_construction, validate_construction and repair_or_correct_construction
(all five go through `_scaled_duration(base, action_name)`); other durations (movement, inspect, communicate, idle) are untouched.

### Intended consequence
The neutral setting is now ×1.0 rather than ×0.875, so **every** baseline duration of these five actions is 14% longer than
in the screening. This is a correction of the reference point, not a tuning, and it is why all absolute outcome levels must be
re-estimated (section 10).

### Tests
Duration equals base × one hook factor for each action at build_speed 0, 0.1, 0.5, 0.9, 1; neutral leaves the authored duration unchanged;
monotone and bounded (×1.5 … ×0.5); build_speed alone (without its hook) cannot change a duration; end-to-end from the resolved
profile at 0.1/0.5/0.9 gives the hook's own range. Mutation check: reinstating the second factor fails all of them.

## 4. Part 3: teammate-model evidence

### Audit of update pathways
| Pathway | Before | After |
|---|---|---|
| `perceive_environment` (≤ 2.0 distance) | copied `agent.mental_model["knowledge"].rules` | records only publicly visible behaviour (moving/idle, `last_seen`) |
| `communicate_with` (after each exchange) | `knowledge_ids = set(self.mental_model[...].rules)` | records the DIK tokens the sender actually **put in its messages** as `believed_ids` (source `communicated_provision`); a stated request is recorded as `believed_missing_ids` (source `stated_request`); the sender records what it handed over (`provided_by_me`) |
| `receive_message` (TKP provenance) | read `knowledge_ids` | unchanged (now reads belief) |
| `brain_context.teammate_help_signals` | read each teammate's own `known_gaps` | received requests / stated lack, plus the teammate's visible status text |
| routing (`_route_request_receiver`) | matched gap ids against copied rules | `_request_route_score`: belief + public role responsibility + past provision |

Evidence classes: *public role responsibility* (task-package `role_scope` of the element) is a separate additive routing term and is
not stored as a belief about holdings; *observed behaviour* gives goals and `last_seen` only; *communicated* and *provided* content give
`believed_ids`/`believed_missing_ids` with source and time in `evidence`; *actual teammate knowledge* is read only by
`modules/teammate_model_metrics.py`, an analysis-only module that no agent code imports.

The existing `_should_apply_tom_update` degradation gate is kept.

### Properties and limits (not tuned)
* The model is **evidence-limited**: it learns only what a teammate demonstrated by sending it, so recall is bounded by what was shown;
  items a teammate later forgets stay believed (stale), so precision is not forced to 1.
* All communicated content is currently truthful (an agent can only send what it holds). **Mistaken or false claims are not modelled**;
  the evidence record has a `source` and `holds` field so a claim-type source can be added later without a schema change.
* No uncertainty weights are stored beyond source/time. A graded-confidence belief model, and a role-based prior with explicit
  uncertainty (the smallest genuine transactive-memory model), are **not implemented**; see section 11.
* `tom_summary`, which is passed into the Pilot context, now carries these beliefs instead of copied rule sets.

### Measurement (analysis only)
`team_model_accuracy` gives belief precision/recall against each target's actual holdings; `routing_quality` scores the observer's own
routing rule against ground truth for every element it lacks that a teammate holds, with the random-routing reference alongside.

### Tests
Perception does not copy hidden rules or information; provision is evidence only for what was sent (held-but-unshown stays unknown), for
both parties; a stated request is evidence of lack, not holding; the model is imperfect (recall < 1) and goes stale after forgetting;
help signals ignore a teammate's hidden gaps until they are communicated; the accuracy and routing measures read ground truth.

## 5. Short mechanistic probe (Part 6)

**Design.** Procedural Baseline, `compact_panel`, Teamwork = Taskwork = 0.5, 600 s, seeds 0/1/4, no Ollama. Five of the screening's cells
(`N_baseline`, `request_high` 0.9, `push_high` 0.9, `taskeff_low` 0.1, `taskeff_high` 0.9) re-run on the hardened branch
(15 runs, all completed, 0 controller errors) and compared with the same cells in the preserved 45-run baseline. N = 3 per cell:
descriptive only, no significance tests. Files: `Outputs/mitm_hardening_probe_20261025/` (git-ignored; `analysis/pre_post_comparison.csv`,
`teammate_model_accuracy.csv`, `safety.json`, scripts, raw runs). A first launch wrote to a wrong output directory because the runner changes
directory; it was stopped after one partial run, the stray files were moved (not deleted) to `_misdirected_interrupted_run/`, and the
probe was relaunched with an absolute path.

**1. Resolved gaps no longer trigger repeated self-requests (verification 1, 2).**
Requests issued by an agent that already held the item: `request_high` **21.3 → 0.0** per run (the other cells were already 0).
All requests in `request_high` now go to the role that owns the requested element (100%; previously 27/41, 11/14 and 43/66 by seed).
The Engineer, which holds `I_ENG_CONNECTOR_RULES`, made 23 and 3 requests for it in seeds 4 and 1 before the fix and 0 after. Mean open
recognized gaps fell in every cell (e.g. baseline 0.37 → 0.18).

**2. Efficiency produces one documented duration effect (verification 3).**
Mean executed duration, three seeds, before → after:

| action | neutral (bs 0.5) | low (bs 0.1) | high (bs 0.9) | low/high ratio |
|---|---|---|---|---|
| transport | 26.25 → **30.0** | 49.35 → **42.0** | 10.35 → **18.0** | 4.8 → **2.33** |
| continue_construction | 1.75 → **2.0** | 3.29 → **2.8** | 0.69 → **1.2** | 4.8 → **2.33** |
| validate_construction | 0.875 → **1.0** | 1.645 → **1.4** | 0.345 → **0.6** | 4.8 → **2.33** |

After the fix the means equal base × (1.5 − bs) exactly (30 × 1.4 = 42, 30 × 0.6 = 18); before, they equalled base × (1.25 − 0.75·bs)(1.5 − bs).
Consequences for throughput are large and expected: completed transports at high efficiency 58.3 → 29.3, `taskwork_completed` 147.3 → 104.3
(baseline 95.7 → 84.0, low 56.3 → 71.7). The efficiency effect on completed work keeps its direction (low 71.7 < baseline 84.0 < high 104.3) but
is about half as steep.

**3. Teammate models do not copy hidden knowledge (verification 4).**
Against ground truth (all data, information and rule tokens the target actually holds): before, precision 1.00 and all-token recall 0.47
(rule-only recall 0.987; the model was a copy of the rule set). After (15 runs, 10 s snapshots): precision **1.00** wherever any belief existed (11 of 15 runs;
the other 4 formed no belief because no provision occurred), mean all-token recall **0.02** (final 0.04, max 0.10). The model
is accurate but nearly empty: it knows only what teammates demonstrated. Routing quality (probability that the top-scoring teammate for an
element the observer lacks actually holds it, ties split) is **0.96** (range 0.92–0.99) against a random-routing reference of 0.60 (0.53–0.80).
This comes mainly from the **public role-responsibility** term in the routing score, which correctly points at the owning role; the learned belief
contributes little at this recall. It is not evidence of a rich transactive model.

**4. Existing DIK acquisition and sharing pathways (verification 5).**
Push pathway intact: `push_high` provision attempts 15.3 → 16.7, DIK acquired by sharing 35.0 → 36.7, role-private diffusion (time-averaged)
0.610 → 0.611; baseline diffusion 0.179 → 0.177. **Request-driven sharing is weaker and shows a latent defect (H1 below).**

**5. Authority and substitutions (verification 6, 7).** No controller errors in 15 runs. The existing-pivot rewrites counted by the harness
were 2 in 15 runs (`inspect_information_source → start_construction` ×1, `→ repair_or_correct_construction` ×1) versus 4 in the same 15 cells
and 9 in all 45 runs before. One type (`inspect → repair_or_correct_construction`) did not occur in the screening; it belongs to the existing
inspect-pivot family and the rate did not rise. The ActionGate and Environment test files pass unchanged (section 8).

### H1: removing gap contagion exposed a missing request-reply channel (documented, **not** patched)
A request is answered by queueing the reply on the answerer (`pending_outbound_messages`), which is delivered only when the answerer next
communicates **with that requester**. Before the repair, the answerer received the requester's gap by contagion, recognized it as its own, and so
initiated request-mode exchanges that carried the queued reply as a side effect. The old design therefore delivered replies through the
defect. Now the answerer has no reason to contact the requester: in `request_high` the requester asked 40/141/78 times (86 on average
versus 40 before), with 113 of the 141 requests in seed 1 respondable, yet provision exchanges following a request pair fell from 6/0/3 to 1/0/1,
and `request_successes` from 26/3/42 to 0/2/1. (Over half of the old "successes" were the Engineer re-acquiring an item it held:
23 of 42 in seed 4.) A received request does not become a solicited-provision opportunity for the answerer because
`theory_of_mind[...]["requested_dik_ids"]` is cleared at the next perception tick (within 2.0 of the requester); this was already so before and
I preserved that lifetime.
A minimal fix would persist a received request until it is answered, expose it as a solicited-provision opportunity for the answerer's Pilot, and
clear it on fulfilment. It changes what an answerer is prompted to do and should be reviewed first, since it could be read as a new response mechanism.
Until decided, **do not use request-propensity results from this branch or from the screening as evidence about request behaviour.**


## 6. Part 4: externalization versus plan proposal

### Diagnosis
All three of the proposed explanations hold in part:

1. **Counted twice.** `externalization_episodes += 1` (`agent.py`, end of the `EXTERNALIZE_PLAN` execution branch) runs on the same
   code path, immediately after `team_plan_proposals += 1`; every early exit in that branch (duplicate plan, refused by an active plan)
   skips both. The two counters therefore cannot differ.
2. **A single policy route.** The Pilot has one externalization identity (`externalize_plan`); its only counted effect is to author a
   `team_plan` artifact. A Pilot has no way to externalize knowledge without proposing a plan.
3. **Knowledge externalization already exists but is not counted separately.** The plan artifact carries `carried_information` (the
   task-declared plan-relevant information the author holds; `_attach_carried_information`), consulted later and turned into
   `dik_acquired_from_artifact`. When the single-active-plan rule refuses a competing plan, `_contribute_to_live_plan_artifact`
   still externalizes the author's held information onto the live plan **without a proposal**. The construction subsystem also has
   typed workspace entries (`claim`, `evidence`, `uncertainty`, `design_note`; `record_project_epistemic_externalization`).

Evidence from the 45 screening runs (event logs): 111 `team_plan_proposed`; 299 `externalization_plan_refused_active_plan`, of
which **70 contributed held information to the live plan**; 111 typed workspace entries, all the default `design_note` (one per proposal; the note text is
generic and its references are simply the author's last two held rules, not a chosen claim); 0 agent-authored `whiteboard_plan` artifacts.
(The event `artifact_externalization_action_executed` also fires for consults: 2,855 = 111 externalizations + 2,744 consults, so it is
not an externalization count.) So at least 70 externalization acts exist that are
not plan proposals and are not counted as episodes.

### Smallest controller-neutral mechanism (proposed, not implemented)
* **M1, measurement only (no behaviour change).** Emit one `knowledge_externalized` event whenever an agent puts held DIK onto the workspace
  (new plan artifact with carried information, or contribution to a live plan), listing the element ids. Define
  `externalization_episodes` as those events and keep `team_plan_proposals` as plan artifacts. They then diverge only when the events do:
  a proposal with no carried information is a plan but not a knowledge externalization; a contribution is a knowledge externalization
  but not a plan. This does not add events; it names two that already occur. (Screening data can be re-derived with M1 now, from the
  existing `externalization_plan_refused_active_plan` payload.)
* **M2, functional separation (needs review).** Reuse the existing `epistemic_action` parameter of `externalize_plan`
  (`externalize_claim | evidence | uncertainty | design_note`). When it is set and the author supplies references to DIK it holds,
  the workspace entry is written (and its references become consultable, as `carried_information` is today) **without** creating a
  team plan. Plan proposal stays the effect of the plan intent. No new action identity, no new communication architecture; Pilots
  that do not supply the parameter behave as now. The Procedural policy would need a documented rule for when to externalize
  without a plan (the `externalize_gap` opportunity flag already exists and is already weighted by the constituent).

Theory: MITM separates producing an external representation of knowledge from proposing or committing to a team plan.
M1 is required for any externalization-versus-planning contrast; M2 is required only if externalization is to occur without planning.
Decision requested: M1 only, or M1 + M2.

## 7. Part 5: how knowledge should affect construction correctness

### Audit: held knowledge → artifact → validation
1. **Held knowledge → Pilot choice.** Held rules/information determine readiness blockers (`_build_readiness_blockers`; the
   `start_construction` readiness threshold is 3.0) and validation readiness. They shape which actions are legal and attractive. They do not shape what is built.
2. **Choice → artifact.** A construction action carries a project id and nothing else. Structure types (`house`, `greenhouse`,
   `water_generator`, connectors) have fixed `required_resources` and a fixed list of `build_steps`; `execute_build_step` marks the next
   step complete. There is no design decision, parameter or feature in the action, so an informed and an uninformed agent
   produce identical artifacts.
3. **Artifact correctness.** `project["correct"]` (hidden) is set False by `record_build_quality` when `rng > fidelity` on a build step
   (and re-rolled after each repair), with `fidelity = (construction_fidelity hook + rule_accuracy)/2`, 0.5 at the defaults.
   `rule_accuracy` is a trait, not held task knowledge.
4. **Validation.** `is_valid = correct ∧ physical_ready ∧ support_ready ∧ rule_ready(validator's held rules) ∧ epistemic_ready`.
   Held rules enter only as the validator's eligibility to validate (`rule_ready`).

**Answer:** knowledge affects eligibility to attempt, validate and repair, never the structure produced. The screening's empty
missing-rules lists on all 447 failed validations follow directly: failures came from the stochastic roll (and, see 5, from readiness
conditions), not from missing knowledge.

5. **Related defect, documented, not patched.** `mark_validated(False)` writes `project["correct"] = False`. A *failed validation*
   for a non-artifact reason (support not ready, validator lacks the rule) therefore marks a physically correct structure incorrect and
   forces a repair. Validation then creates incorrectness rather than observing it. In the screening, 415 of 447 failed validations
   were preceded by a logged build error with no intervening repair; the remaining 32 are consistent with repeated failures or with
   non-artifact causes (the event stream cannot separate them). Fixing this changes what correctness means, so it is part of the
   decision below.

### Source requirements (Experimenter Protocol, Task Design, role briefs, `config/tasks/mars_colony`)
Humans can always physically attempt a structure; correctness depends on the realized features: House (Architect brief): enclosed except
one airlock 1 brick wide and 1 brick high; headroom; single solid floor colour, walls/ceiling a different colour; floor space; spacing of
at least 6 studs; connected to food and water. Greenhouse (Botanist): soil (brown 2x2 + green plate), clear cover, two opposite walls
(breezeway), air gap, watered. Water generator (Engineer): grey foundation, translucent-blue 2×2 stack two bricks high, cap layer, at most
two connected structures, direct connections only. Source discrepancy to adjudicate: the Task Design document says housing headroom is at
least **two** bricks; the Architect brief and the task package (`D_ARCH_HEADROOM_1_BRICK`) say **one**.

### Proposed implementation (not implemented; stop for scientific design review)
Intent: knowledge changes what an agent specifies and chooses; execution skill separately introduces slips; the artifact state
decides correctness; validation only observes.

1. **Specification features, task-grounded.** The task package gains `structure_specs.csv`: per structure type, a small set of *slots*
   (for example `house.airlock_width`, `house.headroom`, `house.floor_colour_uniform`, `house.wall_colour_differs`, `greenhouse.cover`,
   `watergen.cap`), each with a closed option set that contains the compliant value(s) and the plausible non-compliant ones, and the
   DIK element that states the requirement (e.g. `D_ARCH_AIRLOCK_SIZE`). Slots attach to existing `build_steps` (`place_airlock`
   → `airlock_width`).
2. **Choice at the action.** `start_construction`/`continue_construction` accept an optional `design_choices` parameter drawn from the
   slot's option menu. The menu is identical for every agent and does not mark any option correct. The **Pilot** chooses:
   an LLM Pilot reads its held DIK (already in its context); the Procedural Baseline uses a documented rule, "intend the compliant
   option when the agent holds the DIK element that states the requirement, otherwise draw uniformly from the options". The Mecha neither
   chooses nor repairs a choice. An omitted choice is recorded as `unspecified` and resolved by the task package's declared default
   (a decision for review, because the default may be non-compliant).
3. **Execution error stays separate.** The existing fidelity roll becomes a *slip* on the realized slot (`intended → realized`), at the
   current rate, so that skill and knowledge are separable: a rule-holder can slip, a non-holder can be right by chance. No change to
   `fidelity` itself.
4. **Authoritative correctness from artifact state.** The Construction subsystem stores the realized slot values on the project and derives
   `correct` as a pure function of them against the spec. `record_build_quality` becomes "record realized feature". Repair re-opens a
   slot and is subject to the slip again. Validation reads the derived value and no longer writes it (this also removes the
   `mark_validated(False)` defect above). No controller receives `intended`, `realized`, or `correct`.
5. **Measurable knowledge-to-action links.** Events `design_choice_made {slot, options, chosen, held_requirement_dik, basis}` and
   `execution_slip {slot, intended, realized}`; per-slot P(compliant intent | requirement held) versus P(compliant intent | not held), P(slip),
   and a decomposition of artifact correctness into compliant-intent × no-slip.
6. **Tests.** Informed agents intend the compliant option (deterministically for the Procedural rule), uninformed agents draw uniformly;
   fidelity 1.0 gives realized = intended; an informed agent still produces an incorrect structure under a forced slip; an uninformed agent can
   produce a correct one by chance; `correct` is never computed from knowledge possession; validation does not modify `correct`; no
   controller context contains the hidden fields; ActionGate legality is unchanged.

### Alternative considered and rejected
Lowering the slip probability when the requirement is held. It makes knowledge a probability modifier of an unchanged action, so it is
`correct = has_required_knowledge` in disguise and gives no action-to-artifact pathway.

### Decisions needed before implementing
(a) Approve changing the meaning of `project["correct"]` to a derived function of realized features. (b) The slot list and option sets per
structure, and which source resolves the headroom discrepancy. (c) The default for an `unspecified` choice. (d) Whether agents may
observe realized features of an artifact (the human analogue is seeing it), which is a separate and larger change. (e) How the slip rate
maps to the existing 0.5 default. Without (e) the structure error rate would change level, which must be calibrated, not tuned.

## 8. Tests

New: `tests/test_construct_hardening.py` (20 tests: gaps 8, duration 5, teammate model 7). Commands run:

* `python -m pytest -q tests/test_construct_hardening.py`: **20 passed**. Mutation checks: restoring the contagion line, the second
  duration factor, or the hidden-rule copy makes 8 of them fail.
* Two existing tests were updated for intended changes: `test_build_speed_scales_macro_action_durations` (sets the hook, since `build_speed`
  now enters durations only through it) and `test_substantive_pilot_actions_are_not_rewritten_end_to_end` (the sufficiency floor
  `selected > 100` became `> 80`: neutral durations are 14% longer, so fewer decisions occur in 240 s; the no-rewrite assertion is unchanged).
* The 31 existing test files that mention gaps, theory of mind, durations, communication or teammates (509 tests) were run on the frozen
  baseline (a worktree at `aba5f04`) and on the branch before the threshold edit: **baseline 75 failed / 434 passed; branch 77 failed / 432 passed**.
  The two additional branch failures were the threshold test above (edited afterwards; re-run together with the new tests and
  `test_trait_externalization_effects.py`: the threshold test passes and the only failures are 2 that also fail on the baseline) and
  `test_generate_plan_respects_method_state_source_targeting`, which is flaky on the baseline too (it fails or passes in isolation on both trees).
  The 75 baseline failures are pre-existing; none was introduced or fixed here.
* A full-suite comparison was started and stopped because it exceeded practical time; **the full suite was not run on either tree**, and the
  related-file comparison was not repeated in full after the threshold edit.

## 9. Which previously observed effects remain trustworthy

Unaffected by the repaired pathways (they depend on quantities none of the three fixes touches):
* Memory-reliability dose-response on forgetting, reacquisition and retained DIK (retention code unchanged).
* Push proximal effects: selection rate, provision attempts, DIK acquired by sharing, role-private diffusion (confirmed in the probe).
* Team-plan proposal and commitment counts, and the externalization-equals-proposal diagnosis (F4).
* Construction findings F5 (random per-step correctness) and F6 (the legacy overlap index is invalid).
* The finding that role-private knowledge spreads only through interaction channels.

## 10. Which screening results require replication

* **Everything involving task_execution_efficiency.** Magnitudes were inflated (low/high 4.8× versus 2.33×) and the neutral point moved by 14%, so
  all absolute outcome levels (core structures, capacity, throughput, transport counts) and the efficiency dose-response sizes must be re-estimated.
  The direction of the efficiency effect on completed work is preserved.
* **Everything involving request.** Request-high counts, the floor at low, B-tier successes, "gaps closed", the downstream effects of the 64/121
  stale requests, and routing. Request-reply behaviour is additionally unresolved (H1).
* **Teammate-model results** (F7: precision 1.00 and recall 0.998) are superseded; any claim using the teammate model or routing is void.
* **Coordination cost at 600 s** (core structures fell in 3 of 3 seeds): the cost depends on action durations, so it needs replication.
* **Cross-effects** (unsolicited provision displaced by request, coordination, backup or efficiency): the request and efficiency parts changed,
  and the N = 3 scan could not identify cross-effects anyway.
* **Backup**: the help-signal input to `productive_coordination` changed.
* The conclusion that supported capacity is floor-dominated at 600 s is unlikely to change, but must be re-established with the new durations.

## 11. Outstanding scientific decisions

1. **H1, request-reply channel.** Persist received requests until answered and expose them as a solicited-provision opportunity? (A new or
   corrected response mechanism; deferred.)
2. **Part 4.** M1 only (count what already happens as two distinct events), or M1 + M2 (externalize typed knowledge without a plan)?
3. **Part 5.** Approve a derived-from-artifact-features meaning of correctness; the slot list and option sets; the headroom source (1 vs 2 bricks);
   the default for unspecified choices; observation of realized features; calibration of the slip rate.
4. **Teammate model.** Add a role-based prior with explicit uncertainty and/or graded confidence? Model mistaken or false claims?
5. **Neutral duration reference.** Confirm that build_speed 0.5 should mean the authored duration (×1.0) rather than the old ×0.875.
6. **Failed validation writes `correct = False`** (section 7, item 5): include it in the Part 5 decision.

## 12. Readiness and commits

**Not ready for a second construct-validation screen of the whole panel.** The three repairs behave as intended, but (i) the request pathway needs
decision H1, and (ii) without Part 5 a screen would repeat the finding that knowledge cannot affect the structure produced. A narrower screen of
push, memory, coordination and efficiency (which do not depend on Parts 4, 5 or H1) could run on the repaired code if wanted, with the caveat that
team-cognition-to-outcome claims remain limited by Part 5.

Commits on `research/mitm-construct-hardening` (not merged; `main` remains `aba5f04`):
* `dc2f5ee` efficiency applied once;
* `c7a9e21` received requests separated from self-recognized gaps, gaps reconciled against holdings;
* `64253b3` evidence-grounded teammate model, help-signal leak removed, analysis-only accuracy metrics, new tests;
* the final commit adds this report.

Not touched: `main`, `feature/human-participant-testbed`, `stash@{0}`, `CLAUDE.md`, the 45-run baseline dataset.



---

# Part B. Second hardening block: approved decisions implemented

Sections 6 and 7 above were proposals; they are now implemented as described in `MITM_contracts.md` (the contract) and below. H1 (request-reply) is
resolved. Commits on `research/mitm-construct-hardening`: `03f5a7c` request-response ledger; `fbe3f51` externalize_knowledge; `735be6d` artifact-derived
correctness and specification separation; `6137509` repair menu for unlocalized failures; `cab04bc` generative-Pilot tests; `4ebefaa` test alignment; `a2a7583`
Pilot-authority guard fix; and the commit adding this part.

## B1. What changed
* **Request-response (decision 1).** Persistent `request_ledger`; the recipient's Pilot sees a per-requester COMMUNICATE affordance and chooses; the Mecha no
  longer queues DIK payloads. Selection, delivery attempt, provision, uptake and gap resolution are separate events; partial, unanswered, expired, duplicate and
  out-of-range cases are tested (`tests/test_request_response_lifecycle.py`, 13 tests).
* **Externalization (decision 2, M1 and M2).** `externalize_knowledge {kind, element_id[, project_id]}`; distinct events; `externalization_episodes` counts DIK put
  on the workspace; same affordances and gate for Procedural and Ollama (`tests/test_knowledge_externalization.py`, `tests/test_generative_pilot_shared_semantics.py`).
* **Construction (decisions 3-5).** `construction_specs.csv` (14 slots, final packets, headroom minimum one brick); explicit `design_slot`/`design_option`;
  unresolved choices block the step; one fidelity draw per build action realizes a slip; `correct` is derived from realized features; validation, support and
  repair no longer write it; validation feedback is limited to what the validator can assess (`tests/test_construction_specification.py`, 18 tests).
* **Efficiency (decision 6).** Mapping unchanged, neutral 1.0; a translation-level test shows transport, start, continue, repair and validate each get exactly
  one hook multiplier. The neutral change is a deliberate model-version change (see `MITM_contracts.md`); no speed was altered to restore scores.

## B2. Regression against frozen main, by failure identity
Same 104 non-GUI test files, run on a worktree at `aba5f04` and on the branch: **baseline 103 failed / 1048 passed; branch 102 failed / 1049 passed. New failures: 0.**
One baseline failure passes on the branch (`test_provider_internal_fallback_marks_effective_backend`, flaky). The new test files (69 tests) all pass. The 103 baseline
failures are pre-existing and untouched.
Existing tests were edited only where the contract intentionally changed, never loosened: scripted Pilots now supply the specification choice; tests that set a hidden
flag now inject a realized defect; the proof-matrix test uses the request ledger; the action-argument spec set includes the new actions; two timing-dependent checks
became behavioural invariants (the Pilot-authority end-to-end test requires the no-rewrite invariant to be exercised on at least three substantive action families instead of
a decision-count floor; the blocked-selection test requires that control returns to the Pilot, not that it does so within 12 s).

## B3. Pivot audit (Pilot authority)
Instrumented 12 runs (baseline, request-high, extern-high, push-high; seeds 0/1/4; 600 s) logging every pivot that changed an action. First pass: 4 rewrites, **2 genuine
violations**: a forced recovery pivot replaced a Pilot's `externalize_knowledge` with an inspect, because the new action was missing from the substantive and state-mutating
guard sets (my omission). Fixed and tested; no strategy was changed to hide it. After the fix: 2 rewrites in 12 runs, both with an input that was not the Pilot's anchored
decision and an output equal to the Pilot's own choice in that tick, so no substantive Pilot choice was replaced. Rewrites in the 45-run screening and the first probe were not
logged at this granularity and cannot be re-attributed.

## B4. Causal probes (Procedural, compact panel, 600 s, seeds 0/1/4, 15 runs; N=3, descriptive)
| Probe | Result |
|---|---|
| A legitimate request leads to new knowledge in the requester | `request_high`: of 48 requests per run, 1.0 element provided, 0.67 uptakes with acquisition, 1.0 requester gaps resolved through a request; the full chain occurred in 2 of 3 seeds |
| More request propensity gives more meaningful requests, no false gaps | requests 0 / 4.3 / 48 (low / baseline / high; screening 0.3 / 2.0 / 40.3); requests from agents already holding the item: 0 in every cell (screening high: 21.3) |
| The Procedural response is rare | a response was offered in 66 of 927 decisions and selected 3 times (communicate was chosen 10 times); the +1.0 solicited-provision term is a pre-existing hypothesized default and was not tuned |
| Shared artifact gives uptake independent of plans | `extern_high`: 14.3 knowledge externalizations per run (7.0 claims, 1.3 evidence, 2.0 design notes, 10.7 contributions to the shared artifact) against 4.3 at low and 5.7 at baseline, with plan proposals unchanged (2.7 / 2.3 / 2.7); 3.7 uptakes per run from the knowledge workspace (0 at low and baseline). In the screening the two counts were identical in 45 of 45 runs |
| Holding a construction rule changes selected specifications | informed choices (n=140): intended compliant 100%; uninformed (n=205): 38% |
| An informed Pilot can still slip | informed slip rate 39%; informed realized compliant 66% (48 of 140 became defects) |
| An uninformed Pilot can produce a correct design | uninformed realized compliant 38%; 5 fully specified valid structures were built wholly by uninformed actors (4 wholly by informed ones) |
| Failed validation creates no defect | no run produced a failure with a structurally valid artifact, and no defect appeared without a realization event; the Environment-level invariant rests on unit tests and the scripted probe (`scripted_probes.json`: a failed validation left `correct` and the violation list unchanged). Validation only executes when ready, so this case does not arise naturally |
| A valid artifact can lack functional support | 7 structurally valid structures without functional support in the probe; scripted probe: a valid house with no connectors stays valid and validation is refused for missing support rather than failing the artifact |

Probe files: `Outputs/mitm_hardening_probe4_20261025/` (final code); earlier attempts `probe2`, `probe3` and the pre-fix audit are kept; `mitm_pivot_audit_20261025`.

## B5. Compared with the 45-run screening (same cells)
Requests from agents that already held the item fell from 21.3 to 0 (high). Externalization episodes now differ from plan proposals. **Validated core structures fell from
3.7-4.7 to 1.0-2.3 per run.** This follows from the specification: an uninformed builder reaches a valid structure with probability 0.01-0.17, an informed one 0.06-0.25
(against f^n before), and repair now needs an observed or localized defect and a new choice that can slip again. It is an intended consequence of making knowledge matter,
not a tuned setting, and its level is not calibrated. Validation attempts rose (16.7 to 23.0 at baseline) because failed structures are revalidated.

## B6. Remaining limitations and unresolved assumptions
See `MITM_contracts.md` section 6. Chief ones: the slot design and the 0.5 slip rate are uncalibrated abstractions; the informed Procedural rule is deterministic; defects
surface only through a validator with the right DIK; Procedural response and externalization rates are low under hypothesized weights; the closure-dialogue status (TPS) is
still auto-queued; at 600 s supported capacity remains floor-dominated.

## B7. Readiness for the second construct-validation screen
**Ready for a second screen of mechanisms and proximal/intermediate pathways**, with conditions: runs of at least 3000 s and at least 5 seeds (at 600 s validated structures
are 1-2); treat absolute outcome levels as a new model version, not comparable with `aba5f04`; do not read Procedural request or externalization uptake rates as properties of
MITM (they come from hypothesized weights); keep outcome-level hypothesis claims out of that screen. The larger experiment was not launched.
