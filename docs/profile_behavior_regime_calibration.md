# Mars Colony Profile Regime Calibration Audit (Narrow Pass)

## Scope and constraints followed
- Reused existing 2x2 profile matrix and existing harness (`scripts/profile_behavior_sanity_audit.py`).
- Did **not** change core simulation architecture or cognitive mechanisms.
- Added only a small harness-level regime sweep + mild pressure override capability.

## Files inspected
- `scripts/profile_behavior_sanity_audit.py`
- `modules/metrics.py`
- `modules/simulation.py`
- `modules/agent.py`
- `modules/environment.py`
- `modules/construction.py`
- `interface.py`
- `config/tasks/mars_colony/agent_defaults.csv`
- `config/tasks/mars_colony/goal_definitions.csv`
- `config/tasks/mars_colony/phase_definitions.csv`
- `config/tasks/mars_colony/construction_templates.csv`
- `config/tasks/mars_colony/resource_nodes.csv`
- `config/tasks/mars_colony/task_manifest.json`
- `artifacts/profile_behavior_sanity_audit.json`
- `docs/profile_behavior_sanity_audit.md`

## Audit questions answered before code changes
1. **Cleanly variable run parameters (without architecture edits):**
   - run horizon (`steps`),
   - run duration granularity (`dt`),
   - harness-level scenario pressure via config-like runtime overrides (e.g., required construction resources),
   - seed and replication count.
2. **Best candidates for this calibration goal:**
   - Stage 1: horizon sweep (40/60/80) to probe saturation sensitivity and profile interpretability.
   - Stage 2: single mild pressure (construction required resource scaling) to reduce easy completion while preserving interpretation.
3. **Can this be done without core logic edits?**
   - Yes. The harness now applies an optional per-run required-resource scaling to construction templates after simulation initialization.
4. **Smallest practical matrix tested:**
   - 2x2 profiles × 3 horizons (40/60/80) plus 2 mild-pressure variants at 80 steps (`resource_requirement_scale` 1.15 and 1.30), with 2 replications per condition.

## Regimes evaluated
### Stage 1 (horizon only)
- `horizon_40`
- `horizon_60`
- `horizon_80`

### Stage 2 (mild pressure)
- `horizon_80_resource_scale_1.15`
- `horizon_80_resource_scale_1.30`

## Metrics compared (per condition and per regime)
- Epistemic/process:
  - packet absorption attempts/success/rate
  - derivation attempts/success/rate
  - D→I attempts/success/rate
  - I→K attempts/success/rate
- Team/process:
  - communication attempts/successes
  - externalization attempts/created
  - consult attempts/successes
  - adoptions
  - mismatch detections
  - repair attempts/successes
- Outcomes:
  - validated structures
  - repaired structures
  - colony validated ratio
  - phase objectives completed

## Key results
### Stage 1 observations
- **Task profile remained interpretable on epistemics** across horizons:
  - task-high minus task-low packet success delta stayed positive (≈ +0.36 to +0.39).
- **Team profile remained strongly interpretable on communication** and increased with horizon:
  - team-high minus team-low communication successes grew from +150.0 (40) to +310.75 (80).
- **Outcome saturation persisted:**
  - `colony_validated_ratio` remained 1.0 for all 4 conditions in all horizons.
- **Artifact/repair pathways stayed inactive:**
  - externalization created = 0,
  - artifact adoptions = 0,
  - repair successes = 0.

### Stage 2 observations (mild pressure)
- Mild pressure (`resource_requirement_scale` 1.15 / 1.30) **did not break saturation**:
  - `colony_validated_ratio` stayed 1.0 across all conditions.
- Artifact/externalization/adoption and repair pathways remained effectively zero.
- Profile interpretability remained (task epistemic separation + team communication separation), but mission/outcome discrimination did not emerge.

## Best candidate regime for near-term experiments
- From this small sweep, `horizon_80_resource_scale_1.15` ranked slightly best by the harness score (preserving profile separations), but it **still fails the key non-saturation requirement**.
- Practical recommendation: if one regime must be chosen from this pass, use `horizon_80_resource_scale_1.15` as the interim baseline for follow-on calibration because it keeps strongest separability signal while matching current constraints.

## Remaining calibration concerns
- The current mission proxy (`validated_structure_ratio`) appears too easy under tested horizons and mild resource pressure.
- Artifact/externalization pathway remains weakly incentivized in this run setup.
- Repair events are rare/nonexistent because mismatches are not getting induced under these settings.

## Recommended next small step
- Run one additional **single-axis** calibration pass (still harness/config-level) using one stronger-but-still-simple pressure at a time, e.g.:
  - increase required resources further **or**
  - shorten effective simulated time (smaller `dt` with fixed step budget) **or**
  - tighten phase support requirement through config override,
  and stop at the first regime where `colony_validated_ratio` no longer saturates while task/team separations remain interpretable.

## Artifacts produced
- `artifacts/profile_behavior_regime_calibration.json`
- `artifacts/profile_behavior_regime_calibration_probe.json`
- `artifacts/calibration_runs/calibration_stdout.log`
- `artifacts/calibration_runs/calibration_probe_stdout.log`
