# Runtime Plumbing Audit (Normalization + Closure Repair Handoff)

This document records the current *existing* runtime plumbing paths after cleanup, without introducing any new behavioral framework.

## A) Config precedence path (legacy `traits` alias → canonical `mechanism_overrides`)

### Entry points
- Agent config payloads are consumed in `SimulationState.__init__` (`modules/simulation.py`).
- Canonical override normalization is done by `SimulationState._normalize_mechanism_overrides`, which delegates to `normalize_mechanism_override_inputs` (`modules/experimental_config.py`).

### Canonical normalization behavior
`normalize_mechanism_override_inputs(config, mechanism_defaults=...)` enforces one internal path:
1. read `mechanism_overrides` as explicit override candidates,
2. read legacy `traits` as backward-compatible alias,
3. build a single canonical `normalized_overrides` map,
4. apply precedence so explicit `mechanism_overrides` wins over alias values,
5. drop neutral UI autofill values matching defaults unless explicit-preservation flags are set.

Flags that preserve neutral entries:
- `mechanism_overrides_explicit`
- `preserve_neutral_mechanism_overrides`

### Construct/mechanism precedence
Effective precedence is resolved in `ConstructMapper.resolve_mechanisms` (`modules/construct_mapping.py`):
1. baseline mechanism defaults,
2. construct-to-mechanism perturbations,
3. normalized canonical `mechanism_overrides`.

### Runtime storage on agent
In `SimulationState.__init__` (`modules/simulation.py`):
- `agent.construct_values` stores resolved construct vector,
- `agent.mechanism_overrides` stores canonical normalized overrides,
- `agent.mechanism_profile` stores the effective resolved mechanism profile,
- scalar fields (`communication_propensity`, etc.) are copied from `agent.mechanism_profile` onto the `Agent` object.

Manifest output rows from `_agent_manifest_row` serialize both:
- canonical `mechanism_overrides`,
- effective `mechanism_profile`.

## B) Closure repair handoff path (including `source_pointer`)

### Where validation blockers are computed
- Validation blockers are computed via `_construction_action_blockers` and classified with `_classify_validation_blockers` in `modules/agent.py`.
- Closure signatures are built by `_closure_blocker_signature`, which includes both blocker tokens and normalized missing-rule IDs from `_closure_repair_missing_rules`.

### Where closure repair request is generated
- Closure communication requests are shaped in `generate_message` and gated by `_should_send_closure_repair_request`.
- Retry/escalation category selection is handled by `_closure_repair_strategy_for` across:
  - `exact_rule`
  - `precursor_info`
  - `source_pointer`
  - `recheck_commitment`
  - `teammate_redirect`

### Where response categories are assigned
- On `TKRQ` receive path in `receive_message`, response category assignment is performed and emitted in `TPS` payload (`response_category`).

### Where pointer state is stored
- On `TPS` with `response_category == "source_pointer"`, owner calls `_set_closure_source_pointer`.
- `_set_closure_source_pointer` writes `project_closure_state["source_pointer"]` and emits `closure_source_pointer_committed`.

### Where pointed reinspection is selected
- `_active_closure_source_pointer` validates active pointer (project/signature/ttl).
- `_apply_policy_pivots` consults this pointer and rewrites decisions toward `INSPECT_INFORMATION_SOURCE` for the pointed source.
- `_resolve_inspect_target` and closure relevance checks keep inspect targets blocker-focused.

### Where DIK-triggered refresh is called
- DIK updates call `_trigger_epistemic_update_pipeline`.
- That calls `_refresh_relevant_project_state_after_dik_change` → `_recompute_project_state_after_dik_change`.
- Non-DIK but blocker-relevant closure signals in `receive_message` also call `_refresh_relevant_project_state_after_dik_change` (e.g., `recheck_commitment` / `teammate_redirect` `TPS` categories).

### Where blocker shrinkage / return-to-validation is detected
- `_recompute_project_state_after_dik_change` reevaluates closure blockers against active closure project.
- If epistemic blockers clear while in repair mode, it emits `closure_episode_returned_to_validation` and resets repair-mode signature counters.

## C) Remaining risks (current seams)

1. **Legacy naming still exposed for compatibility in some outputs**
   - Some payloads/events still include compatibility `traits` aliases (e.g., brain context / certain metrics outputs), even though runtime control path is canonicalized through `mechanism_overrides` and `mechanism_profile`.

2. **Neutral override preservation can still be forced by flags**
   - `mechanism_overrides_explicit` / `preserve_neutral_mechanism_overrides` intentionally keep neutral values. This is expected but can mask construct-driven perturbations if enabled indiscriminately.

3. **Pointer TTL/exhaustion remains bounded by current heuristics**
   - Pointer deactivation still depends on TTL, attempt count, and no-change rules in existing logic. This is intentional in current system but remains sensitive to cadence/timing.

4. **Blocker recomputation is tied to existing validation blocker classifier**
   - If blocker classifier granularity is too coarse for some edge cases, shrinkage detection can lag despite correct refresh triggering.

5. **Retry/exhaustion bookkeeping is signature-scoped**
   - Signature stability directly affects request escalation memory (`repair_retry_state`, `unsatisfiable_exact`). If signatures are noisy, escalation quality can degrade.

