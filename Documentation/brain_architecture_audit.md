# Brain-Backed Decision Support Audit

## Existing live decision path (before this scaffold)
1. `SimulationState.update(...)` updates the environment, allows proximity communication, and then calls `agent.update(...)` for each agent.
2. `Agent.update(...)` updates physiology and knowledge, runs `_run_goal_management_pipeline(...)`, moves toward target, and may communicate if nearby.
3. `_run_goal_management_pipeline(...)` calls:
   - `update_internal_state(...)`
   - `_evaluate_goal_state(...)`
   - `_plan_actions_for_current_goal(...)`
   - `_advance_active_actions(...)`
4. `perform_action(...)` applies action effects (`move_to`, `communicate`, `construct`, `idle`) and simulator/world checks remain outside LLM logic.

## Where code already aligned
- Individual DIK stores and transformations exist in `Agent.mental_model` plus `update_knowledge(...)` and rule inference.
- Team communication scaffolding exists (`generate_message`, `communicate_with`, `receive_message`).
- Theory-of-mind scaffolding exists (`theory_of_mind`, updates in perceive/communicate).
- Headless simulation already works and has smoke tests.

## Gaps identified
- No structured action taxonomy for executable vs internal cognitive events.
- No explicit brain context packet and structured decision schema.
- No team memory manager for externalized artifacts with provenance/uptake.
- No swappable brain backend abstraction.
- No simulator-safe decision validation layer for candidate brain outputs.
- Logging had no compact structured events for context build/decision validation.

## Minimal insertion points used
- Keep simulator authoritative and unchanged for world truth: integrate decision support inside `Agent.update(...)` only when `sim_state` is provided.
- Add `BrainContextBuilder` + `BrainProvider` in `SimulationState` and keep default provider as deterministic `RuleBrain`.
- Translate structured `BrainDecision` into existing legacy execution primitives (`move_to`, `communicate`, `construct`, `idle`, plus macro `transport_resources`) so no giant rewrite is required.
- Add `TeamKnowledgeManager` to simulation state for shared artifacts/validated team memory.
