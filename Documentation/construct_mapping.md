# Construct → Mechanism → Hook Mapping

This simulator now supports a conservative mapping pipeline:

1. **Constructs** (researcher-facing manipulations)
2. **Mechanisms** (stable simulator-facing latent behavior variables)
3. **Hooks** (implementation-facing effect points in runtime logic)

## Concepts

- **Constructs** are values researchers set (e.g., `teamwork_potential`, `taskwork_potential`).
- **Mechanisms** are the resolved, per-agent behavioral profile (e.g., `communication_propensity`, `build_speed`, `rule_accuracy`).
- **Hooks** are where mechanisms influence live code paths (e.g., action utility bias, action durations, DIK fidelity, mismatch detection).

## Config files

- `config/constructs.csv`
  - Defines construct IDs, bounds, defaults, and whether each construct is enabled.
- `config/construct_to_mechanism.csv`
  - Defines enabled mapping rows from construct values to mechanism values via named transforms.
- `config/mechanism_to_hook.csv`
  - Defines enabled hook rows from mechanisms to runtime hook effects via named formulas.

## Runtime flow

At simulation initialization:

1. `ConstructMapper` loads and validates the three CSV files.
2. For each agent, the simulation resolves:
   - normalized construct profile
   - resolved mechanism profile
   - resolved hook effects
3. The agent receives these values as:
   - `agent.construct_values`
   - `agent.mechanism_profile`
   - `agent.hook_effects`

The resolved mechanism values are also assigned to existing mechanism-like agent fields for compatibility.

## Teamwork / Taskwork compatibility

The GUI still exposes **Teamwork Potential** and **Taskwork Potential** as before.

- GUI High/Low selections are converted to construct values:
  - High → `1.0`
  - Low → `0.0`
- Existing trait sliders are preserved as **mechanism overrides** for backward compatibility.

This preserves the Experiment tab workflow while routing internally through construct → mechanism → hook.

## Extending with a new construct

To add a new construct without changing core code:

1. Add a row to `config/constructs.csv` with bounds/default and `enabled=true`.
2. Add one or more rows in `config/construct_to_mechanism.csv` from that construct to mechanism IDs.
3. Ensure transforms are named from the transform registry (`linear`, `linear_centered`).
4. Optionally add mechanism hook rows in `config/mechanism_to_hook.csv` using supported formulas.
5. Enable rows (`enabled=true`) and run tests.

## Safety and interpretability

- No dynamic eval is used.
- CSV behavior is constrained to named transform/formula registries.
- Bounds are clamped at construct, mechanism, and hook levels.
