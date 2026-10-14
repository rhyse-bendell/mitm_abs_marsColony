# Action System (Canonical Catalog + Pipeline)

This document is the developer-facing action-system map for the current simulator.

## Source of truth

`ExecutableActionType` in `modules/action_schema.py` is the canonical action identity source.

`modules/action_catalog.py` is the enriched metadata layer for those identities. It defines every executable action with:
- planner visibility
- runtime executability
- expected `allowed_actions` surfacing
- aliases accepted for normalization
- mode/step affinities
- translation and execution ownership
- implementation status (`implemented`, `partial`, `experimental`, `deprecated`)

Catalog identity fields are derived from enum members (`entry.action.value` / `entry.action.name`) rather than manually duplicated.

## Pipeline map: “what actions exist, and where do they go?”

1. **Canonical vocabulary**
   - `ExecutableActionType` in `modules/action_schema.py`
   - enriched metadata in `modules/action_catalog.py` keyed by enum-derived IDs

2. **Affordance generation / planner visibility**
   - `BrainContextBuilder._affordances(...)` in `modules/brain_context.py`
   - emits `context.action_affordances` used as `allowed_actions` in planner requests
   - task package can filter by role/action availability

3. **RuleBrain filtering (mode + method + step)**
   - `RuleBrain.MODE_ACTION_PREFERENCES`
   - `RuleBrain.MODE_METHOD_PREFERENCES`
   - `RuleBrain.STEP_ACTION_MAP`
   - `RuleBrain.FALLBACK_ACTION_ORDER`
   - consistency checks against catalog are enforced via `validate_rulebrain_action_references(...)`

4. **Planner payload normalization / alias handling**
   - `OllamaLocalBrainProvider._normalize_action_type(...)`
   - uses catalog aliases via `normalize_action_alias(...)`
   - supports explicit aliases and rejects ambiguous aliases like `build`

5. **Translation to runtime action dicts**
   - `Agent._translate_brain_decision_to_legacy_action(...)`
   - converts planner action to executor-friendly dict (`type`, `duration`, `target`, `project_id`, etc.)
   - can downgrade or reject on missing requirements

6. **Legality + execution**
   - legality/readiness checks in translation and construction gating helpers
   - runtime execution in `Agent._apply_externalization_and_construction_effects(...)` and related action advancement

## New diagnostics (action loss visibility)

The action pipeline now emits explicit events for diagnosability:
- `action_affordances_generated`
- `action_affordance_absent_reason`
- `action_family_filtered_by_mode` (RuleBrain note channel)
- `action_family_filtered_by_method_step` (RuleBrain note channel)
- `planner_action_translated`
- `planner_action_downgraded`
- `planner_action_rejected_missing_requirements`
- `action_legality_failed`
- `action_execution_failed`
- `action_execution_succeeded`

These events are intentionally narrow: they identify where an action disappeared without rewriting planner architecture.

## Why actions can still disappear

Even with a canonical catalog, an action may still disappear because:
- it was not afforded by context/task-role filters
- mode/method/step weighting selected another legal action
- translation downgraded/rejected due to missing target/project binding
- legality/readiness checks blocked mutation execution
- execution started but failed due to runtime state constraints

This is expected in current architecture; the catalog + instrumentation make it diagnosable.

## Structured action arguments (`action_args`)

`BrainDecision` and `PlannedActionStep` carry an optional `action_args: Dict[str, Any]` (default `{}`) for action-specific arguments. It is validated per action by `modules/action_schema.py::validate_action_args` (allowed keys, required keys, value types, id shape) and enforced by `AgentActionGate`, which rejects with `malformed_action_args` plus a specific token (`missing_action_arg:<k>`, `unsupported_action_arg:<k>`, `unknown_structure_type:<t>`, ...). Actions not listed in `ACTION_ARG_SPECS` accept no arguments. The ConstructionManager remains authoritative for legality.

Only two actions use it so far (other legacy argument hacks, e.g. transport durations in `assumptions`, are deliberately not migrated):

- `start_construction`: `{"structure_type": <task-package structure type>}`. A START with `structure_type` is the only runtime project-creation path (any task structure type): affordances, probes and target lookups are read-only, and the committed START (`Agent._commit_typed_start` -> `ConstructionManager.start_new_project`) creates exactly the Pilot's type (never defaulting to `house`), after readiness is checked. A START without it only continues an existing unfinished project at the target (`start_mode: existing_project`); on a site with none it is rejected (`structure_type_required_for_new_project`). Startable types come from the task package's enabled construction templates (`ConstructionManager.startable_structure_types`); bridges are a separate capacity-unlock path. Each affordance is `start_mode: new_project` with its own `action_args`, one per currently legal (build target, type); the list states legality only.
- `connect_structures`: `{"connector_project_id", "endpoint_a_id", "endpoint_b_id"}`. A connector project must first be started and built; connecting never creates one.

Compatibility: the retired packed `target_id = "connector|a|b"` is still read, but only at `BrainDecision` construction (`__post_init__`), where it is converted into `action_args` and cleared, so everything downstream sees a single representation.
