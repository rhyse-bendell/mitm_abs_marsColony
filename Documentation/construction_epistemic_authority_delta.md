# Construction Epistemic Authority Delta

## Authority boundary changed

Construction action translation in `Agent._translate_brain_decision_to_legacy_action(...)` now enforces runtime gating for construction actions (`start`, `continue`, `repair`, `validate`) instead of only logging advisory readiness failures.

- `start_construction` / `continue_construction` now hard-block on epistemic readiness blockers and ungrounded low-trust method notes.
- `repair_or_correct_construction` now requires both readiness and a detected mismatch signal.
- `validate_construction` now requires relevant expected-rule knowledge and is intentionally independent of transport-logistics prerequisites.

## Existing machinery promoted to enforcement

The runtime gate reuses existing components (rather than creating a parallel system):

- readiness blockers from `_build_readiness_blockers(...)`
- existing plan grounding trust + missing-prerequisite notes (`plan_method_status`, `validation_notes`)
- construction template expected rules (now canonicalized to `R_*` IDs)

## Completion semantics tightened

Construction now separates resource progress from mission-significant completion:

- `resource_complete` (and `structurally_complete`) indicate resource thresholds reached.
- `validated_complete` indicates validated completion.
- `status=complete` now requires `validated_complete`, not only delivered bricks.

`validate_construction` explicitly transitions eligible projects to validated completion; repair can restore correctness before validation.

## Namespace assumptions resolved

A small normalization layer maps legacy expected-rule tokens (for example `rule:house_enclosed`) onto canonical `R_*` identifiers at task load time.

Authoritative namespace is canonical rule IDs (`R_*`).

## Intentionally still advisory / soft

- planner action selection itself remains advisory (RuleBrain/LLM still recommend actions)
- low-trust plans can still be adopted, but construction execution now enforces runtime gates before simulator execution effects are applied
- fallback decision behavior remains unchanged

## Follow-up items

- add dedicated task-package aliases table if additional legacy rule tokens appear
- consider splitting validation outcomes into richer states (`needs_repair`, `validated_incorrect`, `validated_correct`) in GUI summaries
