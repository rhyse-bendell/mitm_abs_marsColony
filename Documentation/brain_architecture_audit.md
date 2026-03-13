# Brain-Backed Decision Routing Audit

## Live call path (current implementation)
1. `SimulationState.update(...)` updates environment state and proximity communication, then calls `agent.update(dt, environment, sim_state=self)` for each agent.
2. `Agent.update(...)` always updates physiology and DIK acquisition first (`update_physiology`, `update_knowledge`).
3. In full simulation mode (`sim_state is not None`), the **authoritative route** is the brain-backed path:
   - `_plan_trigger_reason(...)` determines whether replanning is required.
   - If no trigger and a plan is still valid, `_continue_cached_plan(...)` reuses cached decision intent.
   - Otherwise `_build_rule_based_brain_decision(...)` builds `BrainContextPacket`, queries configured `BrainProvider`, validates/repairs/rejects decision, and logs outcome.
   - `_adopt_new_plan(...)` persists a reviewed plan record and action intent.
   - `_translate_brain_decision_to_legacy_action(...)` converts typed decision to simulator-safe executable action primitives.
   - `_advance_active_actions(...)` executes through existing simulator mechanics.
4. Simulator truth logic remains outside brain output:
   - movement/path legality,
   - world state updates,
   - structure comparison/repair,
   - communication mechanics,
   - resource and phase constraints.

## Legacy compatibility path
- `sim_state is None` keeps `_run_goal_management_pipeline(...)` for unit-level or compatibility callers.
- Legacy wrappers (`decide`, `decide_next_action`, `select_action`, `update_active_actions`) remain explicitly compatibility-only and are not the authoritative simulation route.

## Decision traceability events
The events stream now records:
- `brain_decision_query` with trigger reason and prior plan id,
- `brain_plan_continued` when a cached plan is reused,
- `brain_decision_outcome` with acceptance state (`accepted`, `repaired`, `rejected`) and validation detail.

## Minimal insertion strategy retained
- No simulator truth rewrite.
- No external API/network dependency added.
- Default backend remains deterministic `RuleBrain`; backend selection is now configurable via `brain_backend`.
