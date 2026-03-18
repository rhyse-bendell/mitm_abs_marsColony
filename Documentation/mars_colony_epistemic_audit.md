# Mars Colony Epistemic Pipeline Audit (Runtime + Task Package)

## 1) Executive summary

The repository already has a substantial **task-package architecture** and **runtime hooks** for DIK processing, communication, externalization, and construction. Canonical task truth is centralized in `config/tasks/mars_colony/*` and loaded into typed structures by `TaskModelLoader`, with strong static validation and constructive-witness analysis in `TaskValidator`. 

However, runtime semantics are currently mixed:

- **Implemented strongly**: source-gated inspection, derivation triggering from held/team DIK, action legality filtering, planner response schema validation, plan-method grounding checks, artifact externalization/adoption, and witness-path auditing.
- **Partially implemented / compressed**: interpretation quality, integration semantics, uptake fidelity, and rule-conditioned execution are often represented as counters/traits/events rather than hard runtime preconditions.
- **Main gap**: constructive witness paths and many task-declared requirements are mainly **audited/reportable**, not always **hard constraints** on execution.

Bottom line: this is not greenfield; most architecture exists. The key delta to vision is tightening runtime enforcement so knowledge-building stages are mandatory where intended, not just observable.

## 2) Repo map of the epistemic pipeline

### Canonical layer (declarative)
- Task package schema and typed loader:
  - `modules/task_model.py`
- Mars task package canonical content:
  - `config/tasks/mars_colony/task_manifest.json`
  - `task_sources.csv`, `source_contents.csv`
  - `dik_elements.csv`, `dik_derivations.csv`
  - `rule_definitions.csv`, `goal_definitions.csv`, `plan_methods.csv`
  - `artifact_definitions.csv`
  - `action_availability.csv`, `action_parameters.csv`, `communication_catalog.csv`
  - `construction_templates.csv`, `environment_objects.csv`, `interaction_targets.csv`, `resource_nodes.csv`
  - `phase_definitions.csv`, `role_definitions.csv`, `agent_defaults.csv`

### Validation + audit layer
- Static package validation + constructive witness generation:
  - `modules/task_validation.py`
- Runtime witness coverage/failure audit:
  - `modules/runtime_witness_audit.py`

### Runtime cognition/execution layer
- Agent epistemic + planner + action loop:
  - `modules/agent.py`
- Simulator authority + world updates + agent stepping:
  - `modules/simulation.py`
- Environment legality/access/pathing:
  - `modules/environment.py`
- Construction world state:
  - `modules/construction.py`
- Team memory/artifacts:
  - `modules/team_knowledge.py`
- Brain contract/context/provider boundary:
  - `modules/brain_contract.py`, `modules/brain_context.py`, `modules/brain_provider.py`
- Action schema:
  - `modules/action_schema.py`
- Legacy DIK object model + fallback packets:
  - `modules/knowledge.py`

## 3) What is already implemented

### A. Canonical task/package layer

1. **Representations present now**
- Data/Information/Knowledge elements: `DIKElement` + CSV-driven rows (`dik_elements.csv`).
- Derivations: `DIKDerivation` + runtime application (`_apply_task_derivations`).
- Rules/goals/plan methods: `RuleDefinition`, `GoalDefinition`, `PlanMethod` + plan-grounding checks.
- Artifacts/externalizations: `ArtifactDefinition` + `TeamArtifact` and construction/whiteboard artifacts.
- Construction requirements: `ConstructionTemplate` includes required resources + expected rules.

2. **Canonical truth model files**
- The canonical model is task package + loader in:
  - `modules/task_model.py`
  - `config/tasks/mars_colony/*` listed above.
- Runtime loads this model at `SimulationState.__init__` and seeds environment/agents from it.

3. **Legacy/parallel systems**
- Legacy fallbacks remain in runtime:
  - hardcoded `RAW_OBJECTS/ZONES/INTERACTION_TARGETS` in environment,
  - fallback DIK packets in `knowledge.init_dik_packets` when no `task_model`,
  - fallback construction projects in `ConstructionManager._build_projects`,
  - legacy goal/action pipeline wrappers in `Agent` (`decide_next_action`, `_run_goal_management_pipeline`).
- These do not fully conflict, but they are parallel pathways and can bypass strict task-package semantics in non-task-driven contexts.

### B. Runtime epistemic process

4. **Actual runtime path** (what currently happens)
1. Source target is selected and slot legality checked (`_resolve_inspect_target`, `_inspect_source`, environment source-slot APIs).
2. Packet DIK is absorbed (`absorb_packet`) under probabilistic uptake.
3. Shared-source additions can be written to team validated knowledge (`_write_shared_source_to_team_knowledge`).
4. Derivations are applied from local + team validated DIK (`_apply_task_derivations`).
5. Planner context includes DIK summaries/artifacts/affordances (`BrainContextBuilder`).
6. Planner returns decision (RuleBrain or optional LLM backend), validated + grounded to legal actions + plan methods.
7. Decision translated to executable legacy action(s), enacted in simulator loop.
8. Construction/project artifacts are externalized and updated; mismatch/repair may occur.

5. **Explicitly modeled vs collapsed**
- Explicit: source access, acquisition, derivation, communication acts, artifact consult/externalize, plan grounding, action legality.
- Collapsed/assumed: semantic interpretation quality and integration depth (mostly inferred via counts/events/traits), and explicit uptake repair loops beyond heuristic/ad-hoc behavior.

6. **Where derivations are available vs enforced**
- Available declaratively: `dik_derivations.csv` + `TaskValidator` reachability closure.
- Runtime triggered: `Agent._apply_task_derivations` in `update_knowledge` and `update`, using held + team validated DIK.
- Not globally enforced: no global requirement that particular derivations must be completed before all downstream actions; readiness/gating is partial.

7. **Communication/integration required vs shortcut**
- Team-only knowledge/rules are identified statically (`team_only_rules`) and can be represented in runtime readiness checks.
- But runtime can still progress through fallback actions and trait-driven behavior; communication/integration is encouraged and logged, not universally mandatory before execution.

8. **Constructive witness paths: constrain or audit?**
- RuntimeWitnessAudit consumes events and classifies step completion/failures.
- This provides strong **observability** and post-hoc accountability.
- It does **not** directly block or enforce simulator transitions by itself.

### C. Agent cognition / brain integration

9. **RuleBrain role now**
- First-class deterministic baseline backend.
- Selects legal next actions from context affordances and phase/readiness cues.
- Also used as fallback for local LLM provider failures.

10. **LLM backend role now**
- Optional (`OllamaLocalBrainProvider` and stubs).
- Produces `AgentBrainResponse` JSON; output is parsed, validated, and can fall back to RuleBrain if invalid/timeout/error.

11. **Authority boundary alignment**
- Mostly aligned with “brain as advisor/planner, simulator as authority.”
- Brains cannot directly mutate world state; decisions are translated and executed by simulator-side action logic.
- Boundary blur is minor and mostly due to legacy wrappers/fallback paths, not direct world mutation by LLM.

### D. Externalization and team cognition

12. **Representation**
- Team-validated DIK map (`validated_knowledge`) and artifact store (`TeamArtifact`) with validation state, uptake count, consulted_by, knowledge summary.
- Construction state continuously externalized as artifacts.

13. **Externalize/adopt/repair support**
- Yes: externalization (`externalize_plan` action path), adoption (`adopt_artifact`), mismatch detection and repair logic (`compare_and_repair_construction`).
- Repair is probabilistic/trait-mediated rather than deterministic protocolized uptake-repair.

14. **Artifacts as inputs vs side effects**
- Artifacts are not just logs: planner context includes artifact summaries; consult action reads artifacts and can influence behavior.
- But some task-declared artifact semantics remain weakly coupled to hard planning prerequisites.

### E. Construction and action grounding

15. **Derived rules to actions connection**
- Via readiness and plan grounding checks: method requirements compare required rules/knowledge/info/data to held sets.
- Construction projects carry `expected_rules`; mismatch/repair compares agent rule set with project expected rules.

16. **Construction correctness dependency strength**
- Mixed:
  - correctness has rule-related semantics (`expected_rules`, mismatch detection),
  - but completion status is still primarily resource-delivery threshold in `ConstructionManager.update`.
- So substantive rule coupling exists but is not yet the sole authority over completion legality.

17. **Gaps preventing reliable “knowledge -> correct execution”**
- Completion can occur without strict rule-proof satisfaction (resource threshold dominance).
- Witness pipeline not used as hard precondition gate.
- Integration/uptake quality factors are mostly observational/probabilistic, not hard semantic checks.
- Legacy fallback pathways can dilute strict task-package gating.

## 4) What is only declarative or audited

- Declarative only (not fully enforced end-to-end):
  - many goal `success_conditions` strings,
  - full richness of `plan_methods.candidate_steps/completion_conditions`,
  - artifact definition semantics (`represents`, validation behavior) beyond partial usage,
  - communication catalog semantics (codes exist, limited hard protocol enforcement).

- Audit/report mostly:
  - constructive witness path coverage and failure categories,
  - many epistemic stage events and telemetry.

- Static validation (pre-runtime strong, runtime weakly coupled):
  - `TaskValidator` proves reachability and witness existence, but runtime does not force adherence to those exact witness paths.

## 5) What is missing at runtime

1. Hard gating of mission-critical execution on validated epistemic milestones (by phase/goal/method).
2. Deterministic enforcement that construction validation depends on satisfied rule predicates, not only delivered resources.
3. Stronger operationalization of interpretation/integration/uptake quality (currently mostly traits + probabilistic hooks + logs).
4. Closer runtime coupling between `constructive_witnesses` and action legality/goal advancement.
5. Unified deprecation/guardrails for legacy fallbacks when running task-package mode.

## 6) Critical blockers

### Critical blockers
- Construction completion legality still resource-dominant.
- Witness-path semantics are not authoritative runtime constraints.
- Team-integration steps for team-only rules are not guaranteed prerequisites in all execution branches.

### Important but non-blocking
- Artifact schema semantics underused for strict planner input constraints.
- Communication codebook not fully mapped to enforced epistemic transitions.
- Some plan-method completion semantics are textual/declarative only.

### Nice-to-have refinements
- Finer-grained uptake/repair state machine.
- Better explicit mapping from each DIK derivation to affected affordance utilities.
- Full retirement of legacy non-task-package fallback paths in production mode.

## 7) Smallest viable next implementation slice

A minimal high-impact slice (without rewrite):

1. Add a **runtime epistemic gate** in `Agent._translate_brain_decision_to_legacy_action` or just before action commitment in `Agent.update`:
   - For mission/phase-critical construction/validation actions, require specific rule IDs from task model (or mapped from selected plan method).
   - If missing, convert action to `inspect_information_source`, `consult_team_artifact`, or `request_assistance`.

2. Add **construction completion validator hook** in `ConstructionManager.update` or agent construction handler:
   - completion requires both resource threshold and rule-validation flag set from agent/team epistemic state.

3. Reuse existing `RuntimeWitnessAudit` step taxonomy as gate conditions for only a small set of critical targets (phase goals + required build methods).

This preserves current architecture and uses existing task model, planner contract, and audit machinery.

## 8) Priority file list for ChatGPT review

- `modules/agent.py`
  - priority: **high**
  - why it matters: central runtime epistemic-to-action pipeline; contains source inspect, derivation execution, planner integration, plan grounding, artifact interaction, construction updates, and repair behavior.
  - exact symbols/sections to inspect:
    - `_inspect_source`, `_write_shared_source_to_team_knowledge`, `_apply_task_derivations`
    - `_build_brain_request`, `_execute_planner_request_sync`, `_validate_plan_method_grounding`, `_translate_brain_decision_to_legacy_action`
    - `update`, `_apply_externalization_and_construction_effects`, `communicate_with`, `compare_and_repair_construction`

- `modules/task_validation.py`
  - priority: **high**
  - why it matters: defines solvability, reachability, team-only rules, and constructive witness paths; key to distinguishing what is statically proven vs runtime-enforced.
  - exact symbols/sections:
    - `TaskValidator.validate`
    - `_compute_reachability`, `_compute_team_reachability`, `_apply_derivation_closure`
    - `_build_constructive_witnesses`, `_element_witness`

- `modules/runtime_witness_audit.py`
  - priority: **high**
  - why it matters: runtime observability for epistemic witness coverage and failures; crucial for deciding what is monitored vs enforced.
  - exact symbols/sections:
    - `_build_critical_targets`, `on_event`, `_block_target`, `finalize`

- `modules/brain_context.py`
  - priority: **high**
  - why it matters: what epistemic/team/artifact state the planner actually sees; governs advisor quality and possible action set.
  - exact symbols/sections:
    - `_build_readiness`, `_affordances`, `build`

- `modules/brain_provider.py`
  - priority: **high**
  - why it matters: RuleBrain baseline logic, LLM fallback behavior, and advisor-only boundary.
  - exact symbols/sections:
    - `RuleBrain._decision_logic`, `RuleBrain.generate_plan`
    - `OllamaLocalBrainProvider.generate_plan`
    - `create_brain_provider`, `select_productive_fallback_action`

- `modules/environment.py`
  - priority: **medium-high**
  - why it matters: authority over source legality, interaction reachability, and action affordance feasibility.
  - exact symbols/sections:
    - `classify_source_access`, `can_agent_use_source_slot`, `can_access_info`, `get_interaction_target_position`

- `modules/construction.py`
  - priority: **medium-high**
  - why it matters: determines whether construction correctness is epistemically constrained or resource-threshold driven.
  - exact symbols/sections:
    - `_build_projects`, `update`, `deliver_resource`

- `config/tasks/mars_colony/dik_derivations.csv`
  - priority: **medium-high**
  - why it matters: explicit DIK transformation graph; compare to runtime trigger points.
  - exact sections: all derivation rows, especially cross-role knowledge derivations (lines 19–32).

- `config/tasks/mars_colony/rule_definitions.csv`, `goal_definitions.csv`, `plan_methods.csv`
  - priority: **medium-high**
  - why it matters: defines target epistemic end-state and intended action sequences.
  - exact sections: all rows; especially rules/goals tied to phase targets and method prerequisites.

- `modules/task_model.py`
  - priority: **medium**
  - why it matters: canonical schema loader and package contract.
  - exact symbols/sections:
    - dataclasses for DIK/rules/goals/methods/artifacts/templates
    - `TaskModelLoader.load` and per-file `_load_*` methods

- `modules/team_knowledge.py`
  - priority: **medium**
  - why it matters: team integration substrate for validated knowledge and artifact uptake.
  - exact symbols/sections:
    - `TeamArtifact`, `TeamKnowledgeManager.externalize_artifact`, `adopt_artifact`, `upsert_construction_artifact`

- `modules/knowledge.py`
  - priority: **medium**
  - why it matters: legacy DIK objects and fallback packet system; identify parallel-path risk.
  - exact symbols/sections:
    - `Data`, `Information`, `Knowledge`, `init_dik_packets`, `init_dik_packets_from_task_model`

- `modules/simulation.py`
  - priority: **medium**
  - why it matters: simulation authority wiring; where team knowledge manager, witness audit, and agent loop are orchestrated.
  - exact symbols/sections:
    - `SimulationState.__init__`, `update`, `stop`

## 9) Recommended next prompt to Codex

Use this exact prompt:

"Implement a **minimal runtime epistemic gate** for Mars Colony without rewriting architecture.

Scope:
1) In `modules/agent.py`, before committing translated executable actions in the live `update(...)` path, enforce:
   - If selected action is one of `start_construction`, `continue_construction`, `validate_construction`, require the plan method (if present) and required rules from `task_model.plan_methods` to be satisfied by agent + team validated knowledge.
   - If not satisfied, automatically downgrade action to one of: `inspect_information_source`, `consult_team_artifact`, or `request_assistance` (choose legal/reachable best option).
   - Emit explicit events with missing prerequisites.

2) In `modules/construction.py` (or centralized construction action handling in `Agent`), add a completion gate:
   - project can become `complete` only when resources are sufficient **and** project has been marked epistemically validated (`validation_passed=True` or equivalent), where validation depends on expected rule satisfaction.

3) Keep RuleBrain first-class and do not let brain backends mutate world state directly.

4) Add/adjust tests in:
   - `tests/test_goal_stack_and_plan_grounding.py`
   - `tests/test_task_validation.py`
   - add a focused new test that proves resource-only completion no longer completes when rule validation is missing.

Deliverables:
- code changes + tests
- concise explanation of runtime behavior before/after
- no greenfield redesign."
