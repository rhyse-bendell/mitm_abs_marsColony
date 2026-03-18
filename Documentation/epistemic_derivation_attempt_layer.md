# Epistemic derivation-attempt layer (minimal mechanism-driven slice)

## What changed

Epistemic transitions now follow an **eligibility -> attempt -> success/failure** path instead of guaranteed firing once prerequisites are present.

- Packet uptake (`absorb_packet`) now emits explicit attempt/failure events and uses mechanism-driven success probability.
- Task derivations now emit attempts, can fail, and remain retryable on later updates.
- Successful derivations still produce canonical output elements exactly as before.

## Mechanisms/hooks now driving transitions

The runtime now uses existing `dik_update` hook targets for transition success:

- `absorb_packet`
- `transform_data_to_information`
- `transform_information_to_knowledge`

`rule_accuracy` remains the main competence mechanism feeding those hook probabilities.

## What remains deterministic

The following remain deterministic and authoritative:

- canonical DIK derivation graph (task package truth)
- prerequisite ID checks (`required_inputs`)
- `min_required_count` eligibility checks
- validator/construction truth semantics

These determine **whether an attempt is allowed**, not whether it must succeed.

## Retry semantics

- Failed derivations are **not** added to `executed_derivations`.
- They can be attempted again whenever prerequisites remain satisfied.
- A small retry bonus is applied for repeated attempts to support recoverability without hard lockouts.

## Event observability

The runtime emits explicit attempt/failure/success style events:

- `derivation_attempted`
- `derivation_failed`
- `derivation_succeeded`
- `packet_absorption_attempted`
- `packet_absorption_failed`

Event payloads include probability and stochastic roll for analysis.

## Follow-up ideas (not in this slice)

- Add communication uptake hooks once communication acceptance mechanisms are enabled.
- Add richer context modifiers for contradiction-repair and artifact-supported reasoning where state is explicit.
- Add metrics rollups for per-agent derivation success rates by mechanism profile.
