# MITM simulator contracts (hardening branch)

Concise statement of the contracts the hardened Procedural and generative Pilots share. Companion to
`MITM_construct_hardening_review.md` (evidence, tests, probes). Branch `research/mitm-construct-hardening`, frozen comparison
baseline `main = aba5f04`.

Model-version note: this branch changes the model. In particular `task_execution_efficiency = 0.5` now means the authored duration
(multiplier 1.0, previously 0.875 with a second compounding factor), a deliberate model-version change, not a tuned setting. Results from
`aba5f04` and from this branch are not directly comparable in absolute timing or throughput.

## 1. Request / reply lifecycle

A received request is the **requester's** information need. It is never a gap in the recipient's own knowledge (`known_gaps` holds only
self-recognized gaps).

```
requester: recognized gap --request_assistance--> TKRQ(requested element ids)
recipient: received            record kept in Agent.request_ledger (requester, asked ids, time, provided ids, closing reason)
           actionable          the asked ids the recipient HOLDS and has not provided; shown to its Pilot as a COMMUNICATE affordance
                               addressed to the requester (response_to_request, can_provide, cannot_provide, request_age_s, reachable)
           response selected   the Pilot chooses that affordance (event dik_response_selected)      -- the Mecha never chooses it
           delivery attempted  a communicate action addressed to the requester runs; range is checked (dik_response_delivery_attempted,
                               dik_response_delivery_failed:out_of_range)
           provided            elements actually transmitted; only elements the recipient possesses (dik_response_provided)
           uptake              the requester's mental model changed (dik_request_uptake, measured; the recipient does not know it)
requester: gap resolved        its own ledger sync / reconciliation (dik_request_gap_resolved, dik_gap_resolved)
closing    provided_all | expired (REQUEST_TTL_S = 120 s, hypothesized) | withdrawn (a newer request no longer asks for it)
```

Rules: a partial answer provides only held elements and keeps the rest open (they become actionable if later acquired); selection is not
provision and provision is not uptake; a duplicate request that does not postdate a delivery owes no second reply; a reply never causes a
request, so no loop; the legacy validation-closure dialogue (requests that carry a project id) may still queue a status (TPS) message but
never DIK content.

## 2. Externalized knowledge versus team planning

Two canonical actions, same legal semantics for every Pilot:

* `externalize_plan`: authors a team plan (goals, targets, assignments). It may also carry the plan-relevant information the author holds.
* `externalize_knowledge` with `action_args {kind, element_id[, project_id]}`:
  `claim` (a held canonical rule or information element), `evidence` (a held information element), `uncertainty` (a recognized gap the agent
  lacks), `design_note` (a held element tied to an existing project). It writes to the shared `knowledge_workspace` artifact (information and rules
  can be taken up by consultation); it never creates or revises a plan. The ActionGate refuses a reference the agent does not hold or recognize.

Measurement: `knowledge_externalized` (kind, element ids), `shared_artifact_contribution` (a second author adding to an existing shared artifact),
`team_plan_proposed`, `artifact_consulted`, `dik_acquired_from_artifact` (uptake). Consultation without new content is not uptake.
`externalization_episodes` counts acts that put DIK content on the workspace (explicit entries, information carried by a plan, contributions),
not plan proposals; `uncertainties_externalized` is counted separately.

The externalization constituent scores `externalize_knowledge` when a grounded option exists; plan proposal is scored by coordination.
The utility of the new action (0.35) and its mode preference are hypothesized defaults, not calibrated.

## 3. Intended versus realized construction features

Source: `config/tasks/mars_colony/construction_specs.csv` (14 slots over five structure types; final Architect, Botanist and Engineer packets;
the headroom minimum is **one** brick; the earlier Task Design document says two, a source-version difference). One slot per build step.

* **Options.** Every agent receives the same menu for the next step's slot (`design_slot`, `design_option` as `action_args` of
  start/continue construction; of repair for a reworked slot). Nothing in the menu marks an option correct.
* **Intended.** The Pilot's explicit choice. A step with an unresolved required choice is not built, and nothing fills it in or improves it.
* **Realized.** Equals the intended option unless an execution slip occurred. The slip draw is the earlier single fidelity draw per build
  action (`fidelity = (construction_fidelity hook + rule_accuracy)/2`, 0.5 under the compact panel), not one draw per slot; a slip realizes a
  uniformly drawn other option. Recorded per slot: intended, realized, slipped, fidelity, actor, time, episode (`actor@time:project`).
* **Validity.** `project['correct']` is a derived mirror of "no realized slot is non-compliant" (Construction, `derive_structural_validity`);
  nothing else writes it.

## 4. Structural validity, functional support, validation

| Layer | Owner | Reads | Never changed by |
|---|---|---|---|
| structural validity | Construction, from realized options | realized features | validation, support, validator knowledge |
| functional support | Construction, from attached connectors | support counts | validation, structural validity, validator knowledge |
| validator's assessment | the validating Pilot's own DIK | requirement elements it holds | world truth |
| observed validation outcome | recorded on the project (team-visible) | passed/failed, observed violations, cannot-assess, missing support | hidden defects |

A validation passes only if the structure is valid, support is ready, the validator holds the expected rules, and the epistemic workspace is
ready. A failed validation sets a closure state only; it does not touch the artifact. Repair reworks a slot (new intended and realized option)
and may itself slip. Controller-visible context carries observations only (observed validation failure, observed violations, option menus), never
`correct`, intended/realized options or the specification log.

## 5. Knowledge-to-action causal pathways

1. Held DIK → Pilot decision: requirement-bearing elements inform which option is intended (Procedural: documented rule, see below; generative
   Pilots: they read the held content).
2. Intended option → execution slip → realized option → derived structural validity.
3. Held rules → validator's eligibility to validate (`rule_ready`) and ability to observe violations → targeted repair menu.
4. Held information/rules → externalization options → shared workspace → consultation → uptake by teammates.
5. Requests → response opportunities → provision → uptake → gap resolution.

**Procedural Baseline decision rule for a slot** (`resolve_design_variant`): if the agent holds any element that states the slot's requirement it
intends a compliant option (uniform among them); otherwise it draws uniformly among the offered options, stably per agent/project/slot. Non-chosen
options are scored down so each slot counts as one variant in the action-family pick; the best option keeps the unmodified score, so knowledge
changes which option is attempted, not how attractive building is. The fallback policy applies the same rule.

## 6. Limitations and unresolved scientific assumptions

* **Repair menu.** After a failed validation the repair menu lists the slots with an OBSERVED violation; if the validator could not localize one, it lists the
  slots nobody could assess (where the defect may hide). Physical permission to rework any built slot is separate from epistemic readiness (the existing
  expected-rule gate). A blind rework by an agent that cannot assess a slot can damage a compliant slot (it may slip or choose by chance).
* **Slip alternatives considered.** (a) One independent roll per slot: rejected, it multiplies the 0.5 exposure. (b) A slip always yields a defect: rejected, it would
  equate slip with defect. (c) Lower slip probability when the requirement is held: rejected, it is correct = has-knowledge in disguise. Chosen: one roll per build action
  (as before); a slip realizes a uniformly drawn other option. Uncalibrated consequences: informed P(valid structure) stays near the earlier f^n (house 0.19, greenhouse
  0.13, water generator 0.06, connectors 0.25) while uninformed is far lower (0.07, 0.04, 0.01, 0.17). Blocked attempts no longer inject errors. A completed step with no
  recorded realization (possible only in hand-assembled test states) is reported as unspecified, not defective.
* **Remaining Mecha scaffolds.** The closure-dialogue status message (TPS) is still queued automatically on a received request (no DIK content). A forced recovery pivot can
  still change non-substantive Pilot choices. `externalize_knowledge` was added to the Pilot-authority guard after the pivot audit found it missing.

* The slot list, the option sets and the one-slot-per-step mapping are a minimal source-grounded abstraction, not a complete model of the
  physical task; geometry, spacing and capacity remain outside it.
* The slip rate (0.5 per build action) is a hypothesized default, not a human error rate. A slip now produces a defect with probability below one
  (some alternatives are also compliant, e.g. headroom 1 and 2), and blocked attempts no longer inject errors, so the structure error rate differs
  from `aba5f04` for reasons unrelated to knowledge. This was not recalibrated.
* A slip draws uniformly among the other options; the structure of slip errors (which wrong option) is unmodelled.
* An informed Procedural agent always intends a compliant option; informed-but-wrong choices therefore arise only for generative Pilots or
  scripted decisions. Whether this should be probabilistic is an open modelling choice.
* "Holds a requirement element" is binary and any-of; partial or mistaken understanding of a requirement is not modelled.
* Validation passes without the validator having assessed every slot (the existing rule-readiness gate decides eligibility); stricter
  slot-level certification is possible but would raise the knowledge demand.
* Agents cannot observe realized features directly; defects surface only through validation by a validator who holds the relevant requirement.
* Teammate-belief model remains evidence-limited and has no confidence, role prior or false-claim mechanism.
* Legacy side effects not changed: plan proposals still write one generic project note; the closure-dialogue protocol and some recovery pivots
  remain Mecha-side scaffolds (audited in the review).
* Task packages without `construction_specs.csv` keep the earlier single-flag correctness semantics.
