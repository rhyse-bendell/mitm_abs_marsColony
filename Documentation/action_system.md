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
