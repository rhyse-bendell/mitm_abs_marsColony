# Human Participant Testbed

Branch: `feature/human-participant-testbed` (from `main` @ `deb41d7`). The checkpoint branch
`checkpoint/controller-neutral-resource-guard-20261017` and `stash@{0}` are untouched.

Status: first playable vertical slice (Architect = Human Participant; Botanist/Engineer = Procedural Baseline).
Sections 1-3 are the H1 audit (findings confirmed against the implementation, not against older docs).
Sections 4+ describe what was built and the contracts it keeps.

---------------------------------------------------------------------------------------------------

## 1. H1 audit: what the simulator already is

### 1.1 Spatial representation
* **Continuous 2-D world, metres-like units, 10.0 x 7.4** (`modules/environment.py` `RAW_OBJECTS`, or the task package
  `config/tasks/mars_colony/*.csv` when present). Agent position is a float `(x, y)`. There is no tile occupancy of agents;
  agent-agent collision is off by default (`navigation.ignore_agent_collision = True`).
* **Navigation** is owned by the Environment: `GridManager`/`Environment.plan_path` (A* on a 0.35 grid), `is_point_navigable`,
  `path_reachable`, blocked zones (walls and the A-C blocked region). `Agent.move_toward(target, dt, environment)` walks the agent
  toward `agent.target` each tick; speed is an agent property. A move therefore needs only a destination point.
* **Named places**: `Environment.zones` (rectangles), `interaction_targets` (`Team_Info`, `Architect_Info`, `Botanist_Info`,
  `Engineer_Info`, `whiteboard`, `Build_Site_A/B/C`) each bound to a zone and an object. Project ids (`site_b_house_001`) resolve to a
  site interaction point via `get_interaction_target_position`.
* **Proximity/legality** is already spatial: `get_interaction_access(position, target, role)` (zone / radius / bridge lock),
  `evaluate_information_access` (per-object `access_radius`, default 0.4; also applies the ROLE check), `can_interact_with_table`
  (0.35 radius or in-zone), source access slots (`select_source_access_point`).
* **Resource piles**: `ConstructionManager.resource_nodes` (`ResourcePile`: id, position, quantity). Pile A (site A side) and a second
  pile that is only usable once the B-C bridge is complete.
* **Bridge/gating**: `bridge_bc` (`BridgeState`); Site C is not buildable until it is complete (`is_interaction_target_unlocked`).
* **Carry / "cart"**: there is no cart object. The carried load is `agent.transport_state["carrying"] = {"quantity", ...}`. A TRANSPORT
  is a *composite* action (walk to pile -> pick up -> walk to the project's site -> deliver) with a per-agent duration.

Conclusion: the backend IS spatial enough for point-and-click over the existing continuous space. **No new physics model is
introduced.** Point-and-click destinations are ordinary navigation targets; the Environment still decides reachability.

### 1.2 Existing typed actions (`ExecutableActionType`) and how a participant interaction maps onto them
| Participant interaction | Canonical action | Notes |
|---|---|---|
| click map / object -> walk there | `move_to_target` | In the enum and translation mapping since the beginning but **never offered** and had no gate rule. Gains an optional `x`/`y` argument (general; usable by any controller) and a read-only gate check. |
| open shared brief / own brief / other role's brief | `inspect_information_source` (`target_id` = packet) | Existing gate refuses another role's private packet for non-baseline controllers with `private_source_inaccessible`. |
| read whiteboard / team artifacts | `consult_team_artifact` (`whiteboard`) | Existing. |
| write to whiteboard / externalize | `externalize_plan` | Existing; its content is Mecha-generated from the actor's own plan/DIK, not free text. |
| typed team chat | `communicate` | Existing action identity. Existing execution is Mecha-generated DIK exchange (`generate_message`); a typed message needs a text argument (see 1.3). |
| load + carry + unload resources | `transport_resources` (`target_id` = exact project id or `bridge_bc`) | Composite; offered per project that needs bricks. There is no separate "load"/"unload" verb in the engine; they are stages of one committed transport (`transport_state.stage`). |
| start a project | `start_construction` + `action_args.structure_type` | Typed START, any role, any type. |
| contribute / continue | `continue_construction` / existing-project `start_construction` | Project-addressed. |
| repair | `repair_or_correct_construction` | Offered only after an OBSERVED failed validation; gated on the actor's own DIK. |
| validate | `validate_construction` | Offered with `validation_ready` + blockers read from the same readiness check the gate uses. |
| connect structures | `connect_structures` + `action_args` | Existing, any role. |
| team plan | `externalize_plan`, `consult_team_artifact`, plus plan assignments carried in messages | Existing lifecycle. |
| observe / reassess / wait | `observe_environment`, `reassess_plan`, `wait` | Not offered to the human as game verbs. |

### 1.3 What needs a new GENERAL abstraction
Nothing new at the action-identity level. Two *argument* extensions to existing actions, both general (not human-only):
1. `move_to_target.action_args = {x, y}` -- numeric destination (the schema validator only knew string args).
2. `communicate.action_args = {text, scope}` -- a typed message body. Today a communicate action's content is generated by the Mecha from the
   agent's own DIK; free text is a new *kind of content*, not a new action. It is delivered as a message and logged; it is **not parsed**
   for DIK (the simulator cannot know what a sentence means), so a typed message never silently transfers any DIK to an artificial
   teammate and never reveals another agent's private DIK to the sender.

### 1.4 Things the audit found that matter for humans (design consequences)
* The tick loop is a single-threaded `SimulationState.update(base_dt)`; GUI and headless runner both drive it. An agent only decides when it
  has no active action; Ollama decisions are async jobs and the *world* is frozen by an "inference barrier" while one is pending.
  -> realtime mode must not apply that barrier.
* Several per-agent routines simulate *cognition on behalf of the agent* and are inappropriate for a human: probabilistic uptake of a packet
  (`_attempt_epistemic_transition`, accuracy 0.95), periodic forgetting (`_update_memory`), team-plan uptake/response tick
  (`_team_plan_lifecycle_tick`, plan auto-adoption), and the ambient mismatch detector (`compare_and_repair_construction`). The human
  controller turns these off for the human agent only; they run unchanged for artificial agents.
* Validation/repair/build-correctness in the engine are gated on the actor's *simulated* DIK (held rules at build time, expected rules).
  For a human the simulator can only record **exposure** (what they opened/received). It cannot know what they understood. First slice:
  exposure is the proxy that legality/correctness already reads. **This is an open scientific decision** (see section 9).
* `Agent.is_generative_pilot()` = "controller_type != procedural_baseline"; the gate uses it to refuse another role's private packet.
  A human controller is therefore refused exactly like Ollama, and nothing is inspected in its place.

---------------------------------------------------------------------------------------------------

## 4. Architecture implemented

```
Participant browser (modules/participant_client/*)           <- no game state of its own; draws the last `state`, sends move / menu-item id / chat
        |  WebSocket (loopback, token, origin check, one client per role)
ParticipantServer (modules/participant_server.py)            <- stdlib only; touches nothing but the gateway
        |
ParticipantGateway (modules/participant_gateway.py)          <- validates client messages -> HumanIntent / UI events; serves the last published view
        |  thread-safe queues (HumanParticipantController)
-------- simulation thread (Tk observer GUI loop or scripts/run_human_testbed.py) ------------------------------------------------
Agent._run_human_pilot_controller  (drained every tick)
        |  HumanIntent -> canonical BrainDecision (modules/human_participant.intent_to_decision)
AgentActionGate.evaluate  (UNCHANGED authority; new read-only rules for move_to_target and typed communicate)
        |  legal -> Agent._translate_brain_decision_to_legacy_action (same translation every Pilot uses)
Environment / ConstructionManager                            <- owns legality and every state change
```
Observer side: the existing logger/event stream, Agent States, Interactions, DIK, construction and Data tabs are untouched; they receive the human's
events (`human_*`, `chat_message_sent`, the normal `action_*`/`inspect_*`/`project_*` events) exactly as for any agent, and each Agent States panel is
titled with the controller (`[Human Participant]`, `[Procedural Baseline]`, `[Local Ollama]`).

Files: `modules/human_participant.py` (controller, intents, telemetry, participant-facing blocker catalogue), `participant_view.py` (view model,
reach, menus), `participant_gateway.py`, `participant_server.py`, `participant_client/`, `realtime_testbed.py` (scheduler + `HumanTestbedSession`),
`controller_spec.py` (controller type, execution-mode resolution), `action_gate.py` / `action_schema.py` (move + typed chat), small guarded branches in
`agent.py` / `simulation.py` (see section 6), `interface.py` (host + labels), `headless_runner.py` (batch refuses humans), `scripts/run_human_testbed.py`.

## 5. Spatial model and point-and-click

There is no WASD. The scientific construct is where the participant allocates themselves and what they choose to do there, so the avatar is a
role-coloured dot moved by clicks.
* Click empty map -> `move_to_target {x, y}` (continuous world point). The gate rejects non-navigable (`destination_not_navigable`) or unreachable
  (`destination_unreachable`) points; the Environment's A* then walks the agent at its normal speed. Movement is logged as a state transition
  (`human_action_intent` with the destination, `human_movement_arrived` on arrival, `human_movement_redirected` when changed mid-walk).
* Click an object far away -> walk into range. Named destinations (`Architect_Info`, `Team_Info`, `Build_Site_B`, a project id) use the Environment's own
  interaction points; piles and the B-C bridge have no interaction target, so the destination is a navigable point next to them.
* Click an object when near -> its menu. "Near" is `Environment.interaction_proximity` (the zone/radius already used for access) for packets, whiteboard and
  sites, 0.6 for piles and 0.9 for the bridge (`participant_view.object_reach`). It is purely spatial; role and legality stay with the gate. The check is
  enforced twice, greyed-out in the client AND on the simulation thread (`not_in_interaction_range`), so a client that ignores the UI is still refused.
* A redirect is allowed whenever the agent is only walking. While committed to a substantive action (reading, carrying, building, validating...) any other
  action is refused (`busy_committed`, shown as "You are busy with another action.") and nothing is queued. Typed chat is the exception: it never occupies the avatar.

## 6. HumanParticipantController contract

* Inbound: a bounded, thread-safe intent queue (`submit_intent`). An intent is already a canonical typed action (`ExecutableActionType` value, `target_id`,
  structured `action_args`) plus the UI object it came from. Only actions in `HUMAN_SUBMITTABLE_ACTIONS` are accepted (move, inspect, consult, communicate,
  transport, start/continue/repair/validate, connect). Reassess/observe/wait/meeting/request_assistance and plan externalization are not human verbs.
* The controller never invents, repeats or completes an action. An empty queue means nothing happens (it is not a WAIT; no `pilot_selection_*` event exists
  for a human). It holds no reference to the Environment or the simulation (tested statically).
* Every intent goes through `AgentActionGate`. Illegal -> `human_action_blocked` (research row with the raw blockers; the participant sees the class-A sentence
  or the generic one) and the agent does nothing. A legal intent whose translation still comes back as a blocked placeholder idle is also reported and not executed.
* The adapter returns no fallback and no reroute (`choose_fallback_action` / `handle_blocked_action` return None).
* Switched off for a human only (artificial agents are unchanged): simulated forgetting (`_update_memory`), probabilistic packet uptake (exposure is deterministic
  and consumes no RNG), team-plan response/adoption ticks, ambient mismatch detection (`compare_and_repair_construction`), inspect re-ranking/retargeting (the named
  source is used exactly), stale-target recovery, `receive_message` auto-replies (a message to a human is shown as a line; nothing is answered or planned for them).
* Human vs simulated memory: the DIK ledger records exposure (packets opened, messages shown). The participant view shows only exposed data/information text,
  never the simulator's derived rules and never a model of what the person believes.
* Disconnect: closing the tab or 20 s of silence marks the controller disconnected; new intents are refused (`participant_not_connected`), the avatar idles, the
  world continues, a reconnect resumes. Stop flushes open source-dwell records and closes the controller.

## 7. realtime_testbed scheduling

`execution_mode = simulation` (default; every existing run) is the previous stepping, unchanged. `realtime_testbed` is selected automatically when a
`human_participant` is present (explicitly requesting `simulation` with a human is a configuration error) and is recorded in the manifest.
`RealtimeScheduler` runs the same `SimulationState.update` in 0.1 s ticks, as many as the wall clock has earned (1x by default; at most 1 s of wall time is
replayed per call, the rest is counted as `dropped_wall_s`). The mission clock runs while a person thinks; an idle human never freezes anyone; the global
Ollama inference barrier is bypassed in this mode (an Ollama call runs on a worker and only its own agent waits); actor-specific durations and busy states are
unchanged; Pause pauses the scheduler (and resets its anchor so nothing is replayed); Stop stops once. Batch runs refuse human controllers.

## 8. Participant-visible state (privacy boundary)

`participant_view/1` (`participant_view.build_participant_view`) is a whitelist projection:
* the physical layout everyone sees (walls, packets/boards, tables, sites, bridge, piles, whiteboard) and teammates' position plus a coarse observable activity;
* the participant's own position, busy state, carried bricks, and the mission clock/phase;
* construction as the agent observes it (`built_state`: type, observed status, materials, observed validation failure), never the hidden `correct` flag;
* pile stock only when near; the Information panel holds only exposed items of opened sources; chat only messages to or from this participant;
* menus derived from the Middleware's affordances (below).
Never present: other roles' packets or held DIK, hidden correctness, utilities/rankings, experimenter measures, other agents' controller identity, prompts,
TeamKnowledgeManager data, derived rules. `tests/test_human_participant_privacy.py` scans the serialized view for forbidden keys, other roles' packet text,
teammates' private DIK, simulator-derived rules and controller names, and shows the view is identical whether a project is hidden-correct or hidden-incorrect.
Blockers: class A (participant-visible: private packet, out of range, busy, no bricks, unreachable...) get specific sentences; everything else (readiness,
knowledge, correctness-adjacent) collapses to "That isn't possible right now." while the research row keeps the raw blockers.

Menus: for every selectable object the gateway lists the Middleware affordances for that object (typed START per structure type at a site; carry/continue/start/
validate/repair/connect per project; carry/build at the bridge; carry from a pile; read for packets and the whiteboard). Each option has an opaque id; the
intent behind it never leaves the server, and a forged id is refused (`stale_option`). Another role's packet still offers "Read briefing" (the physical
metaphor allows the attempt); the gate refuses it with `private_source_inaccessible` and nothing is opened in its place.

## 9. Participant client layout (screenshot: `human_participant_testbed_client.png`)

Header chips: role (colour dot), phase, mission time / total, phase remaining, run status (Running / PAUSED / ENDED), carried bricks, busy state, destination.
Left: the top-down canvas (walls, packets, whiteboard, tables A/B/C, bridge, piles, project markers with materials and status, teammates as role-coloured dots
with a coarse activity label, the participant as a ringed dot, a destination marker). Hovering an object shows "Title / Click to interact" or "Title / Move closer
to interact". Right column: the contextual menu (when an object is selected and near; disabled entries explain themselves), the Information panel (opened
sources, collapsible; collapsing one reports dwell) and Team chat (history + input). Errors appear as a toast. Esc closes the menu, Enter focuses chat. The client
holds no game state, renders server text as text nodes only, and shows no debug data.

## 10. Telemetry

Scientific trace `logs/human_participant_events.jsonl` (also mirrored to the observer event stream as `human_*`). Every row has `participant_id`, `agent_id`,
`role`, `controller_type=human_participant`, `sim_time`, `wall_ts`/`wall_iso`, `intended_action`, `grounded_action`, `outcome`, `blockers`, `public_message`.
Events: `action_intent` (accepted: movement, information source, artifact, chat, transport/pickup, project start, contribution, repair, validation, connect),
`action_blocked`, `action_completed`, `movement_arrived` / `movement_redirected`, `interaction_target_selected` (menu opened), `source_opened` / `source_closed`
(simulated and wall dwell) / `source_reopened`, `chat_sent` (plus a `chat_message_sent` event and an Interactions edge per recipient), `message_received`.
Pointer clicks go to a separate low-level `logs/human_input_trace.jsonl`. Never recorded: keystrokes, unsent or empty chat drafts, hover.

## 11. Compatibility evidence (H12)

1. All-Procedural headless run unchanged: a seeded run (seed 3, 300 sim-s) produces a byte-identical fingerprint on `main` (deb41d7) and on this branch
   (14,780 events, identical per-type counts, positions, projects, piles and Pilot-selection hash); `main` is itself deterministic across two runs.
2. Ollama configuration resolves unchanged (`test_existing_ollama_configuration_still_resolves_normally`, `test_ollama_controller.py`).
3-4. Human + Procedural coexist and serialize in the manifest (`test_manifest_records_human_controller_and_execution_mode`).
5. Another role's packet text never reaches the view (`test_human_participant_privacy.py`).
6-8. Gate spy, static import boundary, forged ids, blocked-not-substituted (`test_human_participant.py`).
9. Disconnect/close fails safe; reconnect works; Stop closes the controller.
10. Simulation mode timing: the barrier bypass and scheduler apply only when `execution_mode == realtime_testbed`; the all-Procedural fingerprint is identical.
11. Observer stream: human events, normal agent events and controller labels reach the logger and the Tk panels (`test_observer_event_stream_*`, `test_human_participant_gui.py`).
Whole non-GUI suite, `main` vs this branch (same ignore list, run concurrently): 107 failed / 1005 passed / 2 skipped / 1 xfailed on both, with the identical set
of 107 pre-existing failures (the human tests were excluded from the branch run and pass separately: 32 tests).

## 12. Known limitations and open scientific decisions

* Build correctness, validation readiness and repair grounding in the engine read the actor's simulated DIK. For a human the simulator records exposure only, so
  exposure is the proxy those rules use. Whether humans should be gated this way (or by the task's own rules independent of exposure) is for the researcher.
* Typed chat is delivered and logged verbatim; artificial teammates do not interpret free text, so it transfers no DIK to them. Structured DIK-grounded sharing
  (TIP/TDP) by a human is not offered yet. Plan externalization and whiteboard authoring are not human verbs yet; whiteboard reading is offered but its
  adoption path still carries the artificial-agent probability draw.
* Procedural agents may contribute to projects a human started (shared world); labels do not say who did what.
* Pile and bridge reach radii (0.6 / 0.9) are interface constants, not task parameters.

## 13. Remaining work

* Full single-human gameplay: whiteboard authoring/externalization, DIK-grounded sharing choices, explicit load/unload stages (the engine has one composite carry), a
  richer map (structure footprints), end-of-mission summary for the participant, participant-side pause display polish, per-role onboarding text.
* Multiple humans: gateways/servers already key by role; needs per-human clients on one session, chat addressing, simultaneous-input fairness tests, observer
  view of N participants.
* Human + Ollama mixed teams: scheduling already bypasses the inference barrier in realtime mode; needs validation under real inference load and Ollama-side
  awareness of human chat.
* Formal human-subject use: consent/instruction flow, participant IDs and session control, LAN binding with authentication, clock-sync/latency telemetry, data
  retention and anonymisation, a decision on the exposure-vs-understanding issue above, and formal piloting.

---------------------------------------------------------------------------------------------------

# Milestone 2: human cognition/action semantics

## 14. H13 audit: how the engine decides build truth (before this milestone)

Task truth used: the final Experimenter Script and the Team / Architect / Botanist / Engineer briefs (`Documentation/2..4c *.docx`), as already encoded in
`config/tasks/mars_colony/dik_elements.csv`. Note for context: in the physical experiment there is **no real-time validation**. Participants build; the
colony is photographed at phase end and scored afterwards ("whatever number of colonists your colony can currently support is final"). The simulator's
VALIDATE action is therefore an abstraction of "check what you built"; the only feedback it can legitimately return to a person is pass/fail.
Also in the task truth but not represented by the engine: resource carts must be built and used to move bricks, and structures must be >= 6 studs apart.

### 14.1 Where truth lives
* One hidden boolean per project: `project["correct"]` (`ConstructionManager.create_project` -> True). Nothing in the engine represents *what* was built
  (colours, openings, stacks); a project is a brick count + `build_steps` + the flag. `expected_rules` (one `R_*` id per archetype) is a label, not a validator.
* Only two things ever write `correct`:
  1. `record_build_quality(correct=False)`: called from `Agent._apply_externalization_and_construction_effects` on each build action when
     `self._rng.random() > fidelity`, `fidelity = (construction_fidelity hook + rule_accuracy mechanism) / 2` (an **agent mechanism draw**, not DIK) and again after a repair
     (`repair_fidelity`). 
  2. `repair_project` (sets True blindly: "clears a recorded build error") and `mark_validated` (sets False on a failed validation, True on a pass).
* `observed_validation_failure` / `observable_status` hide `correct` from agents until a validation actually fails; `update()` derives the truth status `needs_repair`.

### 14.2 Where actor DIK/rules enter (A-F of the brief)
| | Procedural / Ollama | Human (milestone 1) |
|---|---|---|
| A Pilot decision generation | held DIK feeds the baseline policy features/terms and the Ollama canonical context | none (a person decides); exposure only logged |
| B physical action legality | `_build_readiness_blockers` (>=2 info items, >=1 rule, role grounding, inspected source, prerequisite rules...) gates START/CONTINUE at translation and in `_commit_typed_start`; REPAIR gated on `_repair_epistemic_blockers` (project's `expected_rules` must be in held knowledge) plus an observed failure; VALIDATE gated on `evaluate_project_validation_readiness(agent_supported_rules=held rules)` at the gate, at translation and at execution; affordance flags `validation_ready`/`repair_ready` mirror them | **same epistemic gates applied to the person** via the exposure-derived mental model: a legal human action could be refused for "not knowing" a rule |
| C artifact construction state | `execute_build_step` flips a step; provenance records `held_*_at_build`; a `design_note` workspace entry is auto-written per step | same |
| D artifact correctness | hidden `correct`, introduced by the fidelity draw (agent mechanism) | the same draw (`rule_accuracy` of the human's agent object) - an invented probability for a person |
| E validation | `is_valid = correct and physical_ready and support_ready and has_required_rules and epistemic_ready`, where `has_required_rules` = the actor's held knowledge (+ team/project evidence) covers `expected_rules`, and `epistemic_ready` = workspace has claim/evidence/design_note; `mark_validated` re-checks readiness before completing | same: a person's validation outcome depended on simulated rules/workspace entries |
| F repair | needs observed failure + held expected rules; success = clear `correct`, then a second fidelity draw | same, silently choosing the "correct repair" |

So milestone 1 coupled a human's legality, correctness, validation and repair to *exposure-derived simulated DIK* and to an *agent fidelity probability*.

### 14.3 Other findings (not changed in this milestone)
* `team_state.construction_artifacts[*].correct` (from `TeamKnowledgeManager.upsert_construction_artifact`) carries the hidden flag into the agent brain context
  (Procedural/Ollama). It is not in the participant view, but it is a hidden-truth channel to artificial Pilots worth a separate decision.
* Structural support accounting (`support_accounting`) assumes one floor unit per house and one soil unit per greenhouse (documented simulator abstractions);
  per-structure footprint/capacity choices therefore do not change supported-colonist numbers yet.
* Placement spacing (6 studs) and resource carts are not represented by the engine.

## 15. The boundary this milestone establishes

```
AI Pilot DIK      -> may influence what the AI chooses (unchanged)
Human exposure    -> logged only (sources opened, dwell, messages shown, information shared); never stands in for cognition
ActionGate        -> physical / task legality only (materials, site, bridge, built state, observed failure for repair)
Artifact state    -> authoritative correctness: a human-authored structure carries its design, and the task rules evaluate THAT design
```
Artificial-controller behaviour is unchanged (the all-Procedural fingerprint is the guard).

## 16. New human construction / correctness semantics

* **Legality is physical/task only for a human.** For a human actor the gate and translation no longer use exposure-derived simulated DIK: `_build_readiness_blockers`
  keeps only "is there a navigable build target", `_repair_epistemic_blockers` is empty, validation readiness is `physical_ready and support_ready` (functional support = the
  food/water connections the task requires of a house; it is world state). Still required: materials/site/bridge, a physically complete structure to validate, and an
  *observed* failed validation before a repair (a person cannot repair what nothing has indicated is wrong). Artificial controllers keep every epistemic gate.
* **Correctness is the task rules evaluated on the participant's design.** A participant START carries `action_args.design` (`{property: option}`) and must choose every
  property (no defaults are ever filled in). The ConstructionManager records `project["design"]` (+ `design_history`) and sets the hidden `project["correct"]` from
  `DesignSchema.evaluate`. No fidelity draw, no `rule_accuracy`, no DIK enters. A contributor's `record_build_quality` is ignored for a designed structure and
  `repair_project` (the old blind "clear the error") does nothing to it. New general artifact-state fields: `authoring`, `design`, `design_history`, `design_evaluation` (hidden).
* **Task data, not code.** `config/tasks/mars_colony/construction_design.json` lists, per structure type, the participant-observable properties, their options, which
  values satisfy the task rules (`valid`, hidden) and the role-brief facts that justify them (`sources`, hidden). `modules/construction_design.py` is the generic loader/evaluator.
  Option order is deliberately not "valid first".

## 17. Participant design choices now available

* House: floor area (free), floor colour, walls/ceiling colour vs floor, headroom, opening, enclosure, spacing to other structures.
* Greenhouse: soil, soil units (free), ceiling, walls, air gap.
* Water generator: foundation, water bricks (colour/size/stack), top.
* Food / water connector: colour.
Judged properties are exactly the ones the Architect/Botanist/Engineer briefs name (cited in the config by DIK element id). Free properties (floor area, soil units) are recorded
but not judged; they do NOT yet change supported-colonist accounting (`support_accounting` still assumes one floor unit per house / one soil unit per greenhouse, a documented
abstraction). Not represented: exact placement/spacing geometry (spacing is a single choice), resource carts, structure footprint on the map.

## 18. Validation and repair semantics (human)

* Validate = the normal validation action; the outcome is the artifact truth (`correct` + physical + functional support). The participant is told only "Validation passed." or
  "Validation did not pass. This structure needs repair." (the physical task has no richer real-time feedback). The authoritative hidden outcome (true correctness, the
  design properties that violated which task rule, with DIK source ids) goes to `logs/human_hidden_outcomes.jsonl` and the experimenter event stream (`human_authoritative_*`),
  never to the participant.
* Validation of a structure another agent built also uses artifact truth, independent of the human's exposure.
* Repair = an explicit design change: the repair form shows the participant's own current choices and the same option lists; they choose what to change; only those
  properties change and the artifact is re-evaluated. An unchanged or irrelevant repair leaves the structure incorrect. A repair with no design is refused (`design_required`).
  Gap: structures that were not designed by a participant (built by an artificial agent) offer no repair surface to a human (`design_not_applicable`).

## 19. Structured information sharing

Free-text chat is unchanged and transfers nothing. An optional `attachments` list (DIK element ids ticked in the "Share information" box) rides on the same `communicate`
action. The gate refuses any id the sender has not **encountered** (`Agent.encountered_dik()`: opened sources, messages received, whiteboard items read; simulator-derived items
excluded) with `attachment_not_encountered`. Delivery calls the existing `receive_message` with canonical TIP (information) / TDP (data) messages carrying the same grounded
objects, so Human -> Procedural (and Ollama, same code path) and Human -> Human receive identical items; for artificial receivers the usual epistemic update pipeline runs.
The visible message stays verbatim. Logged: text, attached ids, sender, recipients, per-recipient delivery and acquisition outcome (`acquired` / `already_held`), a
`communication_exchange` event (`human_authored: true`) for the DIK/interaction traces. Nothing is parsed from free text; no LLM is involved; refused messages are not logged as sent.

## 20. Information panel semantics

The panel is titled **Reviewed information** with the note "A record of what has been shown or sent to you. It is not a record of what you know or remember." It lists
exposure only: items of sources the person opened, items sent by teammates, items read on the whiteboard (grouped by where they were encountered). No simulator confidence,
memory, belief or derived-rule variable exists in the view (tested). Developer rule: *exposure is a research trace, never a proxy for comprehension, memory, belief or correct application.*

## 21. Whiteboard

"Read the whiteboard" shows the current team notes and team plans as written (author, time, text, attached items); attached items count as exposure, there is no adoption
draw. "Add a note..." writes a short text note (<= 280 chars) with optional encountered attachments as a `team_note` team artifact (existing artifact path: author, timestamp,
`carried_information`). The Mecha authors nothing for a person (a human externalize without text is refused). Artificial agents can consult a note through the unchanged
CONSULT path (`team_note` is accepted next to `team_plan` even with no active plan); attached Data are adopted as Data. Telemetry: `whiteboard_note_added`, `whiteboard_read`.

## 22. Realtime cadence finding (pre-existing behaviour, handled only in realtime mode)

Construct and validate effects (and whiteboard effects) run once per tick for the whole length of an action, i.e. one build step per tick. At the calibrated 1.0 simulated-second
tick that is correct; at the 0.1 s realtime tick one action would build/validate ten times as much. In `realtime_testbed` these effects are held to the calibrated cadence
(construct/validate every 1.0 simulated s, artifact effects once per action; `SimulationState.REALTIME_EFFECT_PERIOD_S`). Simulation mode is untouched (fingerprint-identical).
Open question for the researcher: the engine's throughput is defined per tick, not per second; a future clean-up should make effects time-based for every mode.

## 23. Browser playtest (milestone 2) - what it was and what it found

**What it was, honestly.** No hands-on human played. Real Chrome (headless) was driven through the DevTools protocol with real mouse/keyboard events on the real participant page
against a live `realtime_testbed` session (Architect = Human, Botanist/Engineer = Procedural, seed 7), about 3 wall-clock minutes (mission clock ~2:45). Dropdown values were set
by script (a native select popup cannot be driven that way), and the avatar was steered by clicking canvas coordinates. It is a real-UI check, not a substitute for a person.
Three runs were needed; runs 1-2 failed for driver reasons and exposed real UI defects (below). Screenshots: `human_participant_testbed_m2_repair_form.png`, `..._m2_chat_share.png`.

Run 3 completed every step through the UI: walked to and read own packet and team brief; attempted the Botanist packet and got "You are not permitted to view another role's private
briefing."; opened Site B, chose a deliberately flawed greenhouse design (brown dirt only, solid ceiling) and submitted; walked to the pile and carried bricks; sent free chat and a
chat with a ticked attachment (delivered, recipients `acquired`); read the whiteboard and added a note with an attachment (note visible, author/time, the Botanist's committed plan also
shown); validated the greenhouse ("Validation did not pass. This structure needs repair." - hidden log: `soil`, `ceiling` violated); opened the repair form pre-filled with the person's own
choices, changed ceiling and soil, submitted; revalidated ("Validation passed."). All of it is in `human_participant_events.jsonl` / `human_hidden_outcomes.jsonl`.

### Usability problems (kept separate from scientific/architecture issues)
Fixed during the playtest:
* Opening "Share information" (30+ checkboxes) scrolled the whole page and moved the map off-screen; the page now keeps the map fixed and the side panels scroll internally with bounded lists.
* Repair form was titled with the structure name; now "Repair / change the design".
Open:
* Project markers and their labels stack into an unreadable cluster at a busy site (artificial teammates start many structures within seconds); needs fan-out/zoom or a list.
* Feedback lag of roughly 0.4-0.5 s after a click (snapshot cadence 0.25 s + push 0.1 s); the destination marker should appear immediately on click.
* While committed (reading ~6-10 s, carrying) map clicks are refused and the toast is brief; the HUD busy chip should say what is running and how long is left in a more prominent place.
* "Carrying 1" persists after a carry action ends (the engine's carry is two-step: pick up, then bring to a project); nothing tells the person they still hold a brick.
* Dropdown-based design forms are long; no summary of the current design is shown on the map marker (it is in the project menu data only).
* The whole Information list shows every packet item; no search/grouping beyond source.

### Scientific / architecture problems found (not UI)
* Per-tick effect repetition (section 22) - handled for realtime mode only.
* `team_state.construction_artifacts[*].correct` leaks hidden correctness to artificial Pilots (section 14.3) - unchanged, needs a decision.
* Free design properties (floor area, soil units) do not yet change supported-colonist accounting.
* A human cannot repair structures built by artificial agents (no design surface); artificial agents cannot repair participant-designed structures.
* Procedural teammates immediately start many structures (five greenhouses in the first minute), which shapes the human's task; this is Procedural policy, not changed.
* Houses cannot be validated until food and water connectors exist (world truth). A participant sees "This structure still needs its food and water connections."

## 24. Remaining gaps

* Full single-human task fidelity: resource carts, placement/spacing geometry and structure footprints, capacity-bearing design properties wired into support accounting, validation feedback
  as it exists in the physical protocol (none in real time), end-of-mission summary, repair of structures made by others, marker fan-out, onboarding, drawing on the whiteboard.
* Human + Ollama: untested end to end (code paths are shared: `receive_message`, `team_note` consult); needs a model-in-the-loop session and inference-latency telemetry in realtime.
* Multiple humans: gateway/server are per role but unexercised with two live clients; needs chat addressing, simultaneous-input fairness and per-human telemetry joins.
* Formal participant study: consent/instructions, participant IDs, LAN binding with authentication, clock sync and latency logging, retention/anonymisation, pilot with real people, a decision on
  the `correct` leak and on tick-vs-time effect semantics, and an actual human playtest.
