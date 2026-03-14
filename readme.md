# Mars Colony ABS

Mars Colony ABS is an agent-based simulation platform for studying **macrocognition in teams** during a collaborative Mars-colony construction task.

The project is designed for research-oriented experimentation, not just visualization: it couples role-based agents, DIK-oriented cognition (Data / Information / Knowledge), team externalization dynamics, and configurable behavioral mechanisms with session-scoped logging and measures.

## Project Overview

The simulator models a small team (Architect, Engineer, Botanist) working across mission phases to gather information, coordinate, build structures, detect mismatches, and repair/validate outcomes.

Research framing in this repository emphasizes:
- collaborative taskwork and teamwork variation
- DIK transformation and quality
- communication and shared artifact use
- externalization (including construction state) as a team cognition mechanism
- interpretable experiment design through explicit construct-to-mechanism mappings

## Current Capabilities

Implemented capabilities in the current codebase include:

- **GUI-based experiment setup and run control** with Start / Pause / Stop lifecycle controls and an Experiment tab for condition setup.
- **Live environment rendering** and dashboard views (environment overview, agent state summary, construction summary, event monitor, and system log).
- **Role-based agents** (Architect, Engineer, Botanist) with role-specific packet access constraints.
- **DIK-oriented internal state and updates** (packet absorption plus data→information→knowledge progression).
- **Communication and team interaction logic** with event logging.
- **Artifact externalization logic** including plan externalization and consultation/adoption behavior.
- **Construction tracked as externalization-aware team artifacts** via the team knowledge manager.
- **Construct/mechanism-driven behavioral differences** applied at runtime (decision utility bias, action durations, DIK fidelity, construction fidelity, mismatch sensitivity, plan persistence effects).
- **Session-scoped outputs** (manifest, logs, events, run/phase/agent/team measures).
- **Headless-testable simulation components** with unit/integration tests for simulation, interface lifecycle/dashboard behavior, construct mapping, metrics outputs, and brain scaffolding.

## Experimental Design Support

The simulator currently supports condition-building and comparison workflows in several ways:

- **Researcher-facing manipulations in GUI**:
  - Teamwork Potential (High/Low)
  - Taskwork Potential (High/Low)
  - trait sliders (mechanism overrides)
  - role enable/disable and packet access configuration
- **Current condition translation flow**:
  1. Experiment tab settings are read by `build_agent_configs()`.
  2. Teamwork/Taskwork High/Low are converted to construct values (`1.0` / `0.0`).
  3. Trait sliders are applied as mechanism overrides.
  4. `SimulationState` resolves per-agent construct/mechanism/hook profiles using `ConstructMapper`.
- **Comparison-oriented output support**:
  - per-session measures (`run_summary.json`, `phase_summary.json`, `agent_summary.csv`, `team_summary.json`)
  - event logs and agent state logs
  - optional aggregate post-processing utility (`modules/aggregate_measures.py`) that combines run summaries across sessions

This structure enables controlled variation of conditions across runs while preserving a consistent output schema.

## Architecture Overview

Major runtime components:

- **GUI / Interface** (`interface.py`)
  - experiment configuration
  - simulation lifecycle controls
  - dashboard/environment/event displays
- **Simulation orchestration** (`modules/simulation.py`)
  - environment + agents initialization
  - update loop
  - construct mapping resolution at startup
  - metrics and logging integration
- **Agents and behavior logic** (`modules/agent.py`)
  - DIK memory and transformation
  - communication/externalization/build behavior
  - runtime hook effect usage
- **Environment and task flow** (`modules/environment.py`, `modules/phase_definitions.py`, `modules/construction.py`)
  - spatial/zone model, interaction targets, phased mission constraints
- **Team knowledge / artifacts** (`modules/team_knowledge.py`)
  - artifact externalization, consultation/adoption tracking, construction artifact upsert
- **Brain/context/provider scaffolding** (`modules/brain_context.py`, `modules/brain_provider.py`)
  - structured context packets
  - default RuleBrain
  - local/cloud stub/fallback scaffolding for future backend extension
- **Construct mapping system** (`modules/construct_mapping.py`, `config/*.csv`)
  - explicit construct→mechanism→hook resolution with validation
- **Logging and measures** (`modules/logging_tools.py`, `modules/metrics.py`, `modules/aggregate_measures.py`)
  - session directories, event streams, run summaries, aggregation support

## Repository Structure

Key paths:

- `interface.py` - GUI entrypoint and experiment/dashboard interface
- `modules/` - simulation runtime modules (agents, environment, mapping, logging, metrics, brain scaffolding)
- `config/` - construct/mechanism/hook mapping CSV configuration
- `tests/` - unit/integration tests (headless simulation + mapping + outputs + interface behavior)
- `Documentation/` - theory docs, architecture notes, and roadmap material
- `Outputs/` - session output root (created/populated at runtime)
- `launch_interface.sh` / `launch_interface.bat` - platform launch helpers

## Configuration and Construct Mapping

The construct mapping system is implemented and active at runtime.

Configuration files:
- `config/constructs.csv`
- `config/construct_to_mechanism.csv`
- `config/mechanism_to_hook.csv`

Conceptual layers:
- **Constructs** = researcher-facing manipulated variables (e.g., teamwork/taskwork potential)
- **Mechanisms** = simulator-facing latent capacities/dispositions (e.g., communication propensity, build speed, rule accuracy)
- **Hooks** = implementation-facing effect points in runtime logic (e.g., action utility bias, duration scaling, fidelity/threshold effects)

Current Experiment-tab Teamwork/Taskwork settings now flow through this mapping pipeline, while trait sliders remain available as direct mechanism overrides.

See `Documentation/construct_mapping.md` for additional details.

## Running the Simulator

### GUI launch

- Directly:
  - `python interface.py`
- Launcher helpers:
  - Windows: `launch_interface.bat`
  - Linux/macOS: `./launch_interface.sh`

Launcher notes:
- The Windows launcher probes `py -3`, `python`, then `python3`, and can attempt `matplotlib` installation if missing.
- `tkinter` is required for GUI display.

### Running tests

Use:

```bash
python -m pytest -q
```

or run targeted suites from `tests/` as needed.

## Outputs

Each run creates a timestamped session folder under `Outputs/` with session-scoped artifacts such as:

- `session_manifest.json`
- `logs/`
  - agent state CSV logs
  - `events.csv`
- `measures/`
  - `run_summary.json`
  - `phase_summary.json`
  - `agent_summary.csv`
  - `team_summary.json`
- `snapshots/` (reserved for snapshot-style outputs)

The output model is intended to support run-to-run comparison and downstream analysis.

## Current Limitations / Active Development Areas

The repository has substantial implemented functionality, but several areas remain active development priorities:

- continued refinement of reasoning/decision quality and plan robustness
- navigation/pathing sophistication and movement realism
- broader metric coverage for richer experimental analysis
- expanded phase/task realism and scenario depth
- fuller local/cloud brain backend integration beyond stubs/fallback scaffolding

## Development / Contribution Notes

When extending this codebase:

- Keep **simulator truth authoritative** (the simulator remains final arbiter of world legality/state).
- Preserve separation between **constructs**, **mechanisms**, and **hooks**.
- Maintain headless testability where possible (avoid coupling core logic to GUI-only flows).
- Avoid silent changes that break experiment-facing configs or output schemas.
- Prefer explicit, interpretable mappings for theory-driven experimentation.

## Why this architecture?

This architecture is meant to balance:
- **research interpretability** (explicit mappings and logged outputs)
- **experimental flexibility** (condition controls and configurable profiles)
- **engineering extensibility** (modular runtime + backend scaffolding)

It supports both immediate simulation studies and iterative extension toward richer cognitive/team models.
