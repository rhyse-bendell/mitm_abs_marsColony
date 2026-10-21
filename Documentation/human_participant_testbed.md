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
