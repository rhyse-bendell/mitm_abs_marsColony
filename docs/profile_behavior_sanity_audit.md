# Mars Colony Profile Behavioral Sanity Audit

## Audit plan
1. Inspect how construct/profile values are loaded and mapped into mechanism/hook effects.
2. Confirm existing metrics and event streams relevant to epistemic, team, and task outcomes.
3. Add minimal summary instrumentation for missing audit-facing counters.
4. Run a controlled 2x2 profile matrix:
   - High Task / High Team
   - High Task / Low Team
   - Low Task / High Team
   - Low Task / Low Team
5. Compare directional effects (high-low) for task-sensitive and team-sensitive outcomes.

## Files inspected
- `modules/simulation.py`
- `modules/agent.py`
- `modules/metrics.py`
- `modules/brain_context.py`
- `modules/brain_provider.py`
- `modules/construct_mapping.py`
- `interface.py`
- `config/tasks/mars_colony/agent_defaults.csv`
- `config/constructs.csv`
- `config/construct_to_mechanism.csv`
- `config/mechanism_to_hook.csv`
- `modules/aggregate_measures.py`
- `tests/test_metrics_outputs.py`
- `tests/test_construct_mapping.py`
- `tests/test_epistemic_derivation_attempts.py`

## Answers before coding
1. **Profile settings and application**
   - Active baseline construct settings are `teamwork_potential` and `taskwork_potential`, loaded from task defaults and applied per agent in simulation initialization.
   - Constructs map to mechanism values (e.g., communication, help, rule accuracy) and then into hooks (utility, duration, DIK success probability, validation sensitivity).
2. **Existing epistemic metrics**
   - Existing summaries already track derivation attempts/success/no-output and DIK counts, plus many inspect/readiness diagnostics.
   - Packet absorption was logged at event level but not summarized into explicit success/failure counts in run summaries.
3. **Missing metrics for this audit**
   - Packet absorption attempt/success/failure summary counts.
   - Explicit D->I and I->K split attempt/failure/success counts in run summary.
   - Communication/externalization/consultation attempts (not only successful events).
4. **Smallest controlled comparison**
   - Reuse `SimulationState` with per-condition agent configs overriding only two construct values and run headless for fixed steps.
5. **Batch/summary machinery vs dedicated harness**
   - Existing aggregation utility is too generic for this targeted profile matrix, so a tiny dedicated harness was added under `scripts/`.

## Conditions evaluated
- High Task / High Team (`taskwork_potential=0.9`, `teamwork_potential=0.9`)
- High Task / Low Team (`taskwork_potential=0.9`, `teamwork_potential=0.1`)
- Low Task / High Team (`taskwork_potential=0.1`, `teamwork_potential=0.9`)
- Low Task / Low Team (`taskwork_potential=0.1`, `teamwork_potential=0.1`)

Each condition was run with:
- 6 runs per condition
- 40 steps per run
- `dt=0.5`
- `rule_brain` backend

## Metrics collected
### Epistemic process
- packet absorption attempts/successes/failures
- derivation attempts/successes/failures
- D->I attempts/successes/failures
- I->K attempts/successes/failures

### Team/process behavior
- communication attempts/successes
- artifact externalization attempts/creations
- artifact consultation attempts/successes
- artifact adoptions
- mismatch detections, repair attempts, repair successes

### Task outcomes
- validated structures completed
- repaired structures count
- colony survivability proxy (`validated_structure_ratio`)
- phase objective completion count

## Short results summary
- Teamwork manipulation produced a very strong communication split (team-high >> team-low).
- Taskwork manipulation increased packet absorption and derivation success rates on average.
- Externalization/adoption stayed at zero in this short-horizon run regime, so team-profile effects are not yet visible on artifact pathways.
- Final mission outcomes saturated (all conditions reached validated structures ratio ~1.0), so current horizon is not discriminative for mission-level differences.

Observed condition means (from `artifacts/profile_behavior_sanity_audit.json`):
- HT/HT: packet_sr=0.000, deriv_sr=0.000, comm=211, validated=3, survival_ratio=1.0
- HT/LT: packet_sr=0.939, deriv_sr=0.248, comm=61, validated=3, survival_ratio=1.0
- LT/HT: packet_sr=0.000, deriv_sr=0.000, comm=211, validated=3, survival_ratio=1.0
- LT/LT: packet_sr=0.167, deriv_sr=0.195, comm=61, validated=3, survival_ratio=1.0

Directional deltas (high - low):
- task_high_minus_low:
  - packet_absorption_success_rate: +0.3862
  - derivation_success_rate: +0.0262
  - colony_validated_ratio: +0.0000
- team_high_minus_low:
  - communication_successes: +150.0
  - externalization_created: +0.0
  - artifact_adoptions: +0.0

## Calibration concerns
1. **Team-high profiles can suppress epistemic packet throughput** in this short horizon because communication dominates behavior early.
2. **Mission outcomes are too easy/saturated** under the chosen run length and rules; profile differences do not express in final survivability proxy.
3. **Artifact pathway weakly activated** in this setup (externalization/adoption mostly zero), limiting interpretability for team knowledge sharing beyond direct communication.

## Recommended next small step
Run the same harness with a slightly longer horizon and/or constrained early communication (without changing core decision logic) to force stronger inspect->derive->artifact pathways, then re-check monotonicity for:
- externalization/consult/adoption metrics,
- mismatch/repair dynamics,
- mission proxy separation.
