# MITM Mars Colony Agent-Based Simulation Testbed for Macrocognition and Epistemic Teamwork

## Executive Overview

`mitm_abs_marsColony` is a research-oriented agent-based simulation (ABS) testbed for studying **macrocognition in teams** using a Mars colony construction scenario as the current task package. It models not only movement and task execution, but also distributed information access, DIK (Data-Information-Knowledge) progression, communication, externalization, and validation under role constraints.

The core scientific purpose is to treat teams as **epistemic systems**: outcomes depend on what agents can access, infer, share, externalize, and repair—not just whether they can physically perform actions. In this repository, colony construction is coupled to knowledge support, explicit readiness gates, and validation logic, so “success” is constrained by both taskwork and cognition.

Compared with simpler scripted simulations, this platform emphasizes: (a) task-package-driven configuration (`config/tasks/mars_colony/*`), (b) explicit construct→mechanism mappings for experiments (`config/construct*.csv` and `modules/construct_mapping.py`), and (c) modular decision backends (`rule_brain`, local OpenAI-compatible endpoints, and fallback routing).

The Mars Colony scenario is useful because it naturally induces role interdependence (Architect / Engineer / Botanist), shared-resource bottlenecks, staged mission goals, and conditions where teams can fail for epistemic reasons (missing support, mismatch, weak coordination) even when physical actions are available.

---

## Why This Testbed Exists

Many high-value team problems (mission planning, emergency response, complex operations) involve:

- bounded individual knowledge,
- distributed expertise,
- epistemic dependence,
- uncertainty and changing constraints,
- coordination and regulation under pressure.

This testbed exists to support controlled experiments on those processes with repeatable instrumentation. The codebase is designed so researchers can manipulate team/task conditions and decision architecture while keeping environment and task semantics stable.

In practical terms, this repository supports questions such as:

- How do shared mental models emerge (or fail) when information starts distributed by role?
- When do teams over-communicate, under-communicate, or communicate without effect?
- How do construction errors emerge from epistemic gaps, and how are they repaired?
- How do different “brains” (RuleBrain vs LLM-backed planners) behave under identical constraints?

---

## Core Design Philosophy

The simulator is structured so outcomes emerge from interacting layers (environment, task semantics, epistemic state, team coordination, and decision policy), not from hidden shortcuts.

Design intent in current implementation:

- avoid magical omniscience (role-scoped sources and packet access),
- avoid guaranteed scripted wins (validation gates and mismatch handling),
- avoid treating completion as independent from cognition (epistemic support is part of readiness/validation),
- keep simulator state authoritative (agents propose, simulator adjudicates legality and state transitions).

---

## Conceptual Layers of the Simulator

```text
Task Package (CSV/JSON) -> Environment + Agents + Construction + DIK Rules
                           -> Planner/Brain decision proposals
                           -> Action translation/execution in simulator
                           -> Logs/metrics/artifacts -> analysis
```

### 1) Environment Layer

- Spatial world with zones, interaction targets, resource nodes, and spawn points loaded from task files.
- Pathing and target reachability gate what actions are actually executable.
- Phases introduce time-structured constraints and objectives.

Key code/files:
- `modules/environment.py`
- `modules/grid_manager.py`
- `config/tasks/mars_colony/zones.csv`
- `config/tasks/mars_colony/interaction_targets.csv`
- `config/tasks/mars_colony/resource_nodes.csv`
- `config/tasks/mars_colony/phase_definitions.csv`

### 2) Task / Construction Layer

- Construction sites, resource piles, bridges, project templates, staged resources, build progress, and connection logic.
- Projects track both physical status and epistemic/validation-relevant state.
- Readiness and validation are explicit transition points, not implicit completion.

Key code/files:
- `modules/construction.py`
- `config/tasks/mars_colony/construction_templates.csv`
- `config/tasks/mars_colony/construction_parameters.json`

### 3) Epistemic Layer (DIK)

- Task package defines DIK elements and derivation rules.
- Agents absorb source contents, perform derivations, and maintain data/information/knowledge memory.
- Static validator checks reachability and grounding of DIK/rules/goals/methods.

Key code/files:
- `modules/knowledge.py`
- `modules/task_model.py`
- `modules/task_validation.py`
- `config/tasks/mars_colony/dik_elements.csv`
- `config/tasks/mars_colony/dik_derivations.csv`
- `config/tasks/mars_colony/rule_definitions.csv`

### 4) Team Coordination Layer

- Communication intents/actions, help requests, artifact consultation/adoption, and role interdependence.
- Team knowledge manager stores shared artifacts and validated knowledge updates.
- Whiteboard/plan artifacts and construction artifacts function as shared external memory.

Key code/files:
- `modules/team_knowledge.py`
- `modules/action_schema.py`
- `config/tasks/mars_colony/communication_catalog.csv`

### 5) Metacognitive / Regulatory Layer

- RuleBrain policy includes mode/method/step selection with dwell and switching controls.
- Agents track stagnation/no-effect patterns and run repair/closure loops around blocked construction.
- Planner cadence policies include triggered replanning and degraded-mode handling.

Key code/files:
- `modules/brain_provider.py` (policy config and rule methods)
- `modules/agent.py` (closure, repair, cadence, no-effect tracking)
- `modules/goal_manager.py`

Status note: commitment/focus dynamics exist in partial form (e.g., dwell windows, closure/repair state) and remain under active tuning.

### 6) Decision / Brain Layer

- Modular backend routing via `create_brain_provider(...)`.
- Deterministic-ish RuleBrain baseline plus OpenAI-compatible local HTTP pathway and fallback behavior.
- Brain context packets expose structured world/team/cognitive snapshot and legal affordances.

Key code/files:
- `modules/brain_provider.py`
- `modules/brain_context.py`
- `modules/brain_contract.py`
- `modules/simulation.py`

---

## Agent Roles in the Mars Colony Scenario

Default active roles:

| Role | Initial source scope | Typical contribution |
|---|---|---|
| Architect | `SRC_TEAM_SHARED` + `SRC_ARCHITECT_BRIEF` | Shelter validity/support constraints, housing-focused planning/building |
| Engineer | `SRC_TEAM_SHARED` + `SRC_ENGINEER_BRIEF` | Water/connectivity constraints, distribution and infrastructure dependencies |
| Botanist | `SRC_TEAM_SHARED` + `SRC_BOTANIST_BRIEF` | Greenhouse/food constraints and support dependencies |

The task package intentionally distributes role briefs so no single role starts with full solution support. Team-level success therefore depends on communication/integration, not isolated optimization.

---

## The Epistemic Process in This Simulator

The implemented flow is approximately:

1. **Encounter data** via observation/inspection of sources.
2. **Contextualize into information** through packet absorption and role/task interpretation.
3. **Combine into knowledge** via DIK derivations and rule grounding.
4. **Externalize claims/plans** as team artifacts (e.g., whiteboard, construction artifact updates).
5. **Compare and repair discrepancies** when mismatches/blockers are detected.
6. **Unlock readiness for action** when epistemic and material preconditions are sufficient.
7. **Validate against support + structure state** rather than assuming completion from materials alone.
8. **Update shared understanding** through communication, consultation, and artifact uptake.

This means failures can arise from epistemic causes (missing prerequisites, weak integration, poor repair loops) even if agents can move and act physically.

---

## Construction as More Than Building

### Physical taskwork

- Transport resources.
- Start/continue structure assembly.
- Build connectors and infrastructure elements.
- Progress toward phase support targets.

### Epistemic externalization

- Built state encodes design assumptions (e.g., structure type, expected rules, support dependencies).
- Construction artifacts capture provenance and epistemic workspace snapshots.
- Validation outcomes provide inspectable evidence of team reasoning quality.

Why this matters: construction state becomes analyzable evidence of team cognition, not only final task output.

---

## Validation and Closure

Validation philosophy in current implementation:

- Materials alone are insufficient for success.
- Hidden/latent knowledge alone is insufficient for success.
- Project status depends on material state **and** epistemic/rule support consistency.

Typical closure/repair loop:

1. Detect blocker or mismatch.
2. Diagnose missing support/rules or state inconsistency.
3. Inspect additional sources and/or request team input.
4. Externalize revised plan or perform repair action.
5. Re-validate and update artifact status.

---

## How Agents Decide What to Do

At a high level:

1. Goals and support-goals are represented/managed (`GoalManager`, goal stack/registry).
2. Brain context builder packages world, team, DIK state, and legal affordances.
3. Selected brain backend proposes a structured plan/action decision.
4. Action schema validation + translation maps decision into executable simulator action.
5. Environment/construction subsystems adjudicate effects and state transitions.
6. Logging/metrics capture events, traces, and aggregate outcomes.

This supports transparent rule-based execution today while preserving a stable contract for alternative planners.

---

## Brain Modularity and Experimental Control

Backend selection is an explicit experimental variable:

- **RuleBrain**: baseline policy with explicit methods/modes and interpretable mechanics.
- **LLM pathways**: OpenAI-compatible local endpoint support and configurable fallback.
- **Fallback controls**: degraded-mode timing, retries, and productive safe-action fallbacks.

Example questions enabled by this architecture:

- Do backends differ in communication timing/usefulness?
- Which policies fail gracefully under uncertainty or latency?
- Which produce fast construction but weak epistemic justification?

---

## Experimental Parameters

Current experiment/control surfaces include:

- taskwork potential and teamwork potential manipulations,
- construct→mechanism hook mapping profiles,
- trait/mechanism overrides,
- role activation and packet access settings,
- planner cadence/timeout/fallback settings,
- backend type/options,
- construction parameters,
- run count and timestep budget.

Together, these enable controlled between-condition comparisons with consistent output schemas.

---

## Outputs, Logs, and Analysis

Each run writes a session folder under `Outputs/`, typically including:

- session manifest and execution metadata,
- event stream/log files,
- per-run/phase/agent/team summaries,
- planner trace artifacts (when enabled),
- construction and validation-related event history.

Analysis support exists through in-repo loaders/widgets/plots and aggregation utilities (`modules/analysis_*`, `modules/aggregate_measures.py`).

---

## Example Research Uses

- Shared mental model emergence across phases.
- Communication policy effects on convergence and repair.
- Closure dynamics under epistemic uncertainty.
- RuleBrain vs local LLM planning comparisons.
- Over-/under-regulation dynamics in mode switching.
- Team resilience under constrained information distribution.
- Construction artifacts as cognitive externalization evidence.

---

## Current Status

### Implemented / active

- Task-package-driven Mars Colony model (DIK, rules, goals, methods, environment).
- Role-based constrained information access.
- Construction system with readiness/validation state progression.
- Team artifact/knowledge externalization manager.
- Modular brain routing with RuleBrain baseline + local OpenAI-compatible support.
- GUI and headless execution pathways.
- Rich runtime logging, metrics, and test coverage across core subsystems.

### In progress / active refinement

- Commitment/focus and metacognitive regulation tuning.
- Policy calibration for improved convergence and fewer no-effect loops.
- Richer construction semantics and repair dynamics.
- Expanded scenario/task packages beyond current Mars configuration.
- Further backend robustness and comparative evaluation workflows.

---

## Repository Structure

| Path | Purpose |
|---|---|
| `interface.py` | Tk GUI entrypoint and experiment controls |
| `modules/` | Core simulation, agent logic, decision/brain contracts, metrics, analysis tooling |
| `config/tasks/mars_colony/` | Task package data (DIK/rules/goals/environment/actions/roles) |
| `config/` | Construct/mechanism mapping and experimental config assets |
| `scripts/` | Preflight checks, audits, and validation helpers |
| `tests/` | Unit/integration tests across runtime, UI logic, and observability |
| `docs/` + `Documentation/` | Audits, theory notes, architecture/roadmap docs |
| `Outputs/` | Runtime-generated output artifacts |

---

## How to Run

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) GUI run

```bash
python interface.py
```

Platform helpers:

- Windows: `launch_interface.bat`
- Linux/macOS: `./launch_interface.sh`

### 3) Preflight check (recommended)

```bash
python scripts/preflight_check.py
```

Optional repair flow:

```bash
python scripts/preflight_check.py --repair
```

### 4) Tests

```bash
python -m pytest -q
```

### 5) Headless batch execution (programmatic)

Use `modules.headless_runner.run_batch_experiment(settings, ...)` from Python to execute repeated runs without GUI.

---

## Why This Matters

This repository is a reusable **macrocognitive testbed**, not only a scenario demo. It operationalizes cognitive/team science concepts in executable form, enabling controlled, instrumented comparisons of how team outcomes emerge from interacting epistemic, coordination, and construction processes.

The Mars Colony package is the current instantiation, but the architecture (task-package loading, modular decision backends, explicit validation/logging) is intended to support broader research programs on epistemic teamwork and adaptive collective problem solving.
