# Task Package Architecture

## Engine vs task package boundary

The **engine** (modules in `modules/`) remains the generalized MITM simulator and keeps:
- simulation loop, logging, metrics, and visualization plumbing
- core action vocabulary (`ExecutableActionType`) in code
- generalized DIK, communication, and construction execution mechanics

The **task package** (e.g. `config/tasks/mars_colony/`) now defines experiment/task context:
- DIK sources/content/elements/derivations/rules/goals/plan methods/artifacts
- environment objects, zones, interaction targets, spawn points, resource nodes
- phase definitions
- role definitions and baseline agent defaults
- action availability and action timing/parameters
- communication catalog
- construction templates and expected-rule linkage

## How DIK/rules/goals/plans fit broader task context

`modules/task_model.py` loads a typed `TaskModel` that includes:
- DIK and planning assets for cognition/planner-facing context
- spatial/task context used by runtime environment initialization
- role/action/construction defaults used by simulation startup and execution gating

This keeps researcher-editable CSV/JSON task packages authoritative for experiment context.

## Action model rule

Core action enums remain code-defined (`modules/action_schema.py`).
Task packages do **not** generate new action enums. Instead task config controls:
- whether an action is enabled in this task (`action_availability.csv`)
- role scopes for enabled actions
- default durations and metadata (`action_parameters.csv`)

Runtime integration:
- `BrainContextBuilder` filters legal affordances by task action availability.
- `Agent` applies task action durations/parameters and falls back to `wait` if an action is disabled for role.

## Adding a new non-Mars task package

1. Create `config/tasks/<task_id>/`.
2. Provide all required files listed in `REQUIRED_TASK_FILES` in `modules/task_model.py`.
3. Start from Mars CSV/JSON schema and edit values for your experiment:
   - spatial/task context files (`environment_objects.csv`, `zones.csv`, `interaction_targets.csv`, etc.)
   - role/action files (`role_definitions.csv`, `agent_defaults.csv`, `action_availability.csv`, `action_parameters.csv`)
   - DIK/planning/artifact files
4. Run tests to validate package loading and compatibility:
   - `pytest tests/test_task_model_integration.py tests/test_smoke_simulation.py`
5. Launch simulation with `task_id` pointing at your new package.

## Backward compatibility notes

- Mars remains the default task id.
- Engine modules still include conservative fallbacks if task package content is absent.
- Existing GUI and headless startup flows continue to run against default Mars package.
