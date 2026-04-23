# Modeling Macrocognition in Teams Through an Agent-Based Mars Colony Testbed: Theory, Architecture, Measurement, and Experimental Use

## Front Matter

### Title Page
**Working title:** Modeling Macrocognition in Teams Through an Agent-Based Mars Colony Testbed: Theory, Architecture, Measurement, and Experimental Use.

**Authors and affiliations:** Placeholder author block; repository currently maintained as an engineering/research codebase under `mitm_abs_marsColony`.

**Repository / version / commit reference:** `mitm_abs_marsColony`, task package `mars_colony` manifest version `2.0`, branch `work`, commit `496f05d`.

**Date and document version:** April 22, 2026 (UTC), Document version 0.1.

**Corresponding author:** Placeholder; update before publication.

### Abstract
This document specifies the Mars Colony MITM agent-based testbed as an executable scientific environment for studying team macrocognition. The platform operationalizes distributed DIK (Data–Information–Knowledge), role-constrained access, communication, artifact externalization, construction taskwork, and validation loops inside a dynamic environment. It combines theory-grounded abstractions (MITM, shared mental models, transactive memory) with concrete software architecture (task packages, modular brain providers, event logging, and metrics), enabling controlled manipulations and causal analysis.

As implemented in this repository, the simulator is explicitly an executable research testbed: it links theory, runtime behavior, and measurement in a single instrumentation surface; supports controlled experimentation; and allows backend-policy comparisons between RuleBrain and modular external planner providers under shared task and environment constraints. It is intended for both engineering debugging and scientific experimentation.

**Status:** Implemented baseline with active refinement.

### Background and motivation
Team cognition research often captures static snapshots; this simulator captures process. It is designed to expose temporal emergence, misalignment, repair, and adaptation under realistic coordination constraints.

### Theoretical foundation in MITM
The system treats cognition as distributed across agents, interactions, and artifacts. Internalized knowledge and externalized knowledge jointly influence outcomes.

### Why Mars Colony is the current task package
Mars Colony introduces interdependent roles (Architect, Engineer, Botanist), phased mission demands, shared resources, and embodied logistics that naturally stress teamwork-taskwork coupling.

### Simulation architecture and key layers
Environment, agent, cognition/DIK, communication/artifact, construction/project, metacognitive regulation, brain backend, and logging/metrics layers interact on each simulation tick.

### What is modeled
Movement, source inspection, DIK derivation, communication acts, planning/externalization, resource transport, construction progress, validation, and repair.

### What is measured
Event traces, DIK transitions, communication patterns, alignment/convergence proxies, construction/validation progress, and phase-level outcomes.

### Current capabilities and intended uses
Supports deterministic RuleBrain baselines and local OpenAI-compatible backends, headless/GUI runs, parameter manipulations, and comparative experimental analysis.

### Keywords
Macrocognition in Teams; team cognition; collaborative problem solving; agent-based simulation; shared mental models; transactive memory; external cognition; externalization; taskwork; teamwork; human-AI teaming.

---

## How to Read This Document

Sections in this document intentionally mix four content categories:
1. **Implemented behavior** (what executes in the current codebase).
2. **Partially implemented scaffolds** (present in code but still being tuned/validated).
3. **Conceptual mapping** (theory-to-code alignment where implementation is incomplete).
4. **Planned roadmap features** (not yet implemented).

The simulator is under active development, so some sections are deliberately dual-purpose: they serve as both documentation of current behavior and specification guidance for near-term improvements.

Status labels are used where helpful: **Implemented**, **Partially Implemented**, **Experimental**, **Planned**, and **Conceptual Mapping Only**.

---
## Documentation Governance

This file serves both as:

1. **Technical documentation** for current repository behavior.
2. **Scientific specification** linking implementation to MITM theory.
3. **Design control artifact** for future development.
4. **Interpretation guide** for researchers using simulator outputs.

Because the simulator evolves actively, documentation changes should classify content into one of the following states:

- **Implemented** — Present and functioning in current codebase.
- **Partially Implemented** — Present in scaffolded or incomplete form.
- **Experimental** — Available but under active tuning or unstable semantics.
- **Planned** — Intended future functionality.
- **Conceptual Mapping Only** — Theory mapping not yet operationalized.

### Update Rule
When code behavior changes in ways that alter:
- runtime logic,
- metrics semantics,
- task rules,
- planner interfaces,
- file outputs,
- subsystem ownership,

this document should be updated in the same change cycle.

### Preferred Truth Hierarchy
When documentation and code disagree:

1. Authoritative runtime behavior in code  
2. Task package configuration files  
3. Tests validating behavior  
4. This documentation text  
5. Historical planning notes

## Quickstart: Running the Simulator

### Typical Workflow

1. Select the `mars_colony` task package.
2. Choose a brain backend:
   - `rule_brain` (deterministic baseline)
   - local/OpenAI-compatible provider
3. Configure run parameters:
   - team size
   - tick speed
   - planner cadence
   - seed (if applicable)
4. Launch via GUI or headless execution path.
5. Inspect outputs:
   - run summary
   - event logs
   - metrics rollups
   - planner traces
   - project state snapshots

### Recommended First Run
Use:

- backend: `rule_brain`
- default task package
- default agent count
- logging enabled

This establishes baseline behavior before testing alternative planners.

### Recommended Debug Sequence

If a run fails:

1. Read summary artifact  
2. Inspect dominant counters  
3. Review first major divergence in event log  
4. Inspect planner traces  
5. Patch subsystem  
6. Re-run same condition

## Part I. Motivation, Scope, and Scientific Positioning

### 1. Introduction
#### 1.1 Why team cognition needs executable testbeds
- **Limits of static outcome studies:** End-state success obscures pathways and failure signatures.
- **Need to model process, not only success/failure:** Cognitive progression and regulation happen over time.
- **Need to observe emergence, repair, and misalignment over time:** Temporal event logs allow sequence-aware diagnosis.

#### 1.2 Why MITM is the right organizing framework
- MITM frames team problem solving as distributed knowledge building.
- It exceeds narrow shared-mental-model framing by including interaction and externalization.
- It fits CPS because CPS requires iterative, negotiated, role-interdependent reasoning.

#### 1.3 Why agent-based simulation now
- Provides manipulable, theory-consistent, reproducible experiments.
- Connects cognitive constructs to runtime events and engineering mechanisms.
- Extends prior MITM-inspired ABS by explicitly representing DIK pipelines, artifacts, and validation.

#### 1.4 Why Mars Colony
- Distributed expertise is mandatory for valid structures.
- Teamwork (coordination/communication) and taskwork (build execution) are both essential.
- Cognition is visible in movement, communications, artifacts, and built outputs.
- The task is embodied CPS, not just abstract planning.

#### 1.5 What this paper/document is and is not
- Not just a README.
- Not just a design note.
- It is a scientific-technical specification for experimentation and interpretation.

### 2. Problem Statement and Research Gap
#### 2.1 The black box of team knowledge emergence
How individual partial knowledge becomes collective actionable knowledge remains under-instrumented.

#### 2.2 Limits of traditional empirical methods alone
- Static elicitation misses transitions.
- Post hoc coding often loses fine temporal causality.
- Temporal granularity is limited.
- Knowledge lineage is difficult to reconstruct.

#### 2.3 Why a simulation is needed
- Manipulate asymmetry, communication, structure, and timing.
- Test causal propositions about macrocognitive dynamics.
- Support both exploratory discovery and confirmatory theory tests.

---

## Part II. Theoretical Foundations

### 3. Team, Teamwork, Taskwork, and Collaborative Problem Solving
#### 3.1 What counts as a team
A team here is interdependent, goal-directed, role-differentiated, and embedded in a shared mission environment.

#### 3.2 Taskwork vs teamwork
- **Definitions:** Taskwork = direct production actions; teamwork = interactional integration/regulation.
- **Why it matters:** Different mechanisms drive each; failures can occur even with strong individual skill.
- **Failure insight:** Many run failures are integration/coordination breakdowns, not pure execution inability.

#### 3.3 Collaborative problem solving as phased, iterative, and negotiated
CPS cycles through problem analysis, criteria setting, option generation, evaluation/revision, and re-commitment.

### 4. Team Cognition, Interactive Team Cognition, and Macrocognition
#### 4.1 In-the-head vs between-the-heads approaches
The testbed models both private internal states and interaction-mediated cognition.

#### 4.2 Interactive team cognition and communication as cognition
Communication is not just output; it updates shared state and future decisions.

#### 4.3 Why the simulator must support both internalized and interactional cognition
Without both, convergence/divergence and repair dynamics are misrepresented.

#### 4.4 Why artifacts and environment must be treated as cognitive substrates
Whiteboards, plans, and construction states store memory that agents can revisit and reinterpret.

#### 4.5 Context, adaptation, and naturalistic constraints in macrocognition
Phases, movement costs, access bottlenecks, and resource constraints force adaptive regulation.

### 5. The Macrocognition in Teams Model (MITM)
#### 5.1 Original MITM components
Individual knowledge building; team knowledge building; internalized knowledge; externalized knowledge; problem-solving outcomes.

#### 5.2 DIK transformation
Data are observed, contextualized into information, integrated into knowledge, and used to form rules/actionable plans.

#### 5.3 MITM collaborative problem-solving phases
Knowledge construction; team problem model development; consensus formation; evaluation/revision.

#### 5.4 Why MITM is useful for simulation
It enables traceable transitions, measurable proxies, and decomposition into implementable subsystems.

### 6. Expanded MITM for Simulation
#### 6.1 Environmental DIK as a formal input channel
Environmental sources provide structured data inputs; environment is modeled as active information infrastructure.

#### 6.2 Agent knowledge building processes to internalized DIK
Agents encode, select, and integrate inputs under bounded attention; internalized knowledge is partial and error-prone.

#### 6.3 Recursive agent ↔ team knowledge building loops
Agent updates feed team state; team signals/artifacts feed agent updates, recursively.

#### 6.4 Externalized agent DIK vs externalized team DIK
Agent-level externalization can be public but not shared; team externalization implies co-validation/convergence.

#### 6.5 Externalized DIK becoming environmental DIK
Externalizations persist in environment, can be re-consumed, reinterpreted, and adopted later.

#### 6.6 Agent taskwork as an explicit MITM component
Taskwork is explicit as world-transforming action and as materialized cognition.

#### 6.7 Why this expanded MITM matters
It prevents under-modeling environment/artifacts/taskwork and strengthens empirical interpretability.

---

## Part III. Scientific Goals and Research Questions

### 7. What the Testbed Is Designed to Study
#### 7.1 Knowledge emergence
How fragments combine into role-aware actionable knowledge.
#### 7.2 Knowledge divergence and repair
How contradictions emerge and how teams recover.
#### 7.3 Shared mental model development
When and how internal models align.
#### 7.4 Transactive memory and expertise routing
How agents learn who knows what and route requests.
#### 7.5 Teamwork–taskwork coupling
How communication/planning quality affects construction execution.
#### 7.6 Externalization and artifact-mediated cognition
How artifacts improve or degrade coordination.
#### 7.7 Metacognitive regulation and adaptation
How teams detect stagnation and retune behavior.
#### 7.8 Human-like versus alternative decision architectures
Comparisons between deterministic and LLM-like policy backends.
#### 7.9 Cognitive lineage of taskwork and outcomes
How final builds can be traced to prior knowledge/communication events.

### 8. Example Hypotheses / Research Programs
#### 8.1 Knowledge asymmetry and convergence
Greater asymmetry delays convergence unless routing is strong.
#### 8.2 Communication structure and performance
Directed acknowledged exchanges outperform unfocused broadcasts.
#### 8.3 Externalization conditions
Persistent shared artifacts improve delayed coordination.
#### 8.4 Cognitive repair under contradiction
Repair-focused communication improves validation success after mismatches.
#### 8.5 Brain backend comparisons
Backend choice changes planning quality, timing, and failure modes.
#### 8.6 Planning quality versus execution quality
Fast execution can still fail under weak epistemic grounding.
#### 8.7 Effects of environmental stressors
Scarcity/time pressure increase coordination brittleness.
#### 8.8 Path dependence and phase transitions
Early misalignment can lock in later inefficiencies.

---

## Part IV. Mars Colony Task Domain

### 9. The Human Task That Grounds the Simulation
#### 9.1 Experimental task overview
Agents construct colony-critical structures under phased demand and role-distributed knowledge.
#### 9.2 Why construction is a strong CPS environment
Construction requires design validity, resource logistics, and inter-role reasoning.
#### 9.3 Role packets and distributed expertise
Role briefs and team-shared sources intentionally partition knowledge.
#### 9.4 Phases, waves of colonists, and changing constraints
Phase 1 and 2 demands alter requirements and site accessibility.
#### 9.5 Whiteboard use and cognitive artifacts
Whiteboard captures shared plans/summaries as persistent public memory.
#### 9.6 Resource carts, bridge, and spatial logistics as embodied constraints
Transport timing, carrying limits, and bridge-gated site access shape feasible strategies.

### 10. Environment and Task Layout
#### 10.1 Spatial map
- **resource piles:** Main bricks node in resource field.
- **stations:** Role info stations plus team info.
- **bridge:** Existing A-B bridge and B-C unlock logic.
- **build sites:** Sites A/B/C with templates and capacities.
- **whiteboard:** Central shared artifact location.
- **construction zones:** Zone-indexed build regions.
#### 10.2 Zones, slots, and access points
Zone definitions assign coordinates; targets map to object IDs and role scopes.
#### 10.3 Obstacles and movement affordances
Boundary blocked segments and blocked A-C path enforce route constraints.
#### 10.4 Information source locations
Architect/Engineer/Botanist/Team sources are spatially separated.
#### 10.5 Construction footprints and site constraints
Templates specify structure type, target, required resources, and expected rules.

### 11. Colony Objectives and Constraints
#### 11.1 Shelter, food, and water as interdependent colony functions
Housing, greenhouse, and water generation support survivability jointly.
#### 11.2 Bridge decision and second build zone
Site C becomes buildable when bridge condition is satisfied.
#### 11.3 Resource competition among structures
Shared bricks pool requires prioritization.
#### 11.4 Multi-phase colony demands
Phase populations alter support requirements.
#### 11.5 Scoring / survival framing and colony viability
Completion quality is interpreted through structure validity and mission support.

### 12. Roles and Distributed Expertise
#### 12.1 Architect
Primary housing constraints: enclosure/spacing/airlocks/capacity rules.
#### 12.2 Botanist
Greenhouse and food-related constraints, including support dependencies.
#### 12.3 Engineer
Water generation/distribution and infrastructure logic.
#### 12.4 Shared information
Common team packet contains baseline mission framing/objectives.
#### 12.5 Why no single role has the full solution
Role briefs are complementary by design to induce interdependence.
#### 12.6 Two-person adaptation and shared roles, if modeled
Architecture supports altered role activation for reduced team-size experiments.

---

## Part V. Overall Simulator Architecture

### 13. High-Level System Architecture
#### 13.1 Environment layer
Spatial world, phase context, movement/path constraints.
#### 13.2 Agent layer
Per-agent state machines, goals, plans, and action execution.
#### 13.3 Knowledge / cognition layer
DIK memory, derivations, and rule adoption.
#### 13.4 Communication / artifact layer
Message passing, team knowledge manager, shared artifacts.
#### 13.5 Construction / project layer
Material staging, build progress, readiness, and validation.
#### 13.6 Metacognitive regulation layer
Stagnation detection, repair/regrounding, cadence and switching logic.
#### 13.7 Brain / backend layer
RuleBrain and modular provider routing for alternative planners.
#### 13.8 Logging / metrics / analysis layer
Event logger, rollups, summaries, and downstream analysis modules.

### 14. Runtime Control Flow
**Status:** Implemented baseline with active tuning for asynchronous planner operation and degraded-mode handling.

#### 14.1 Tick-level lifecycle
Current runtime behavior follows this authoritative sequence each tick:
1. Phase/time update inputs are prepared (`dt` scaled by speed multiplier).
2. For non-RuleBrain backends, inflight planner and DIK-integration requests are polled.
3. Planner barrier state is refreshed; simulation can pause when barrier policy requires it.
4. Environment updates phase context and emits `phase_transition` when index changes.
5. Construction projects are synchronized into team artifacts (`upsert_construction_artifact`).
6. Agent-local context is built; each agent decides whether planning should fire based on cadence/trigger state.
7. RuleBrain or external provider is queried.
8. Provider output is validated against schema and normalized decision contracts.
9. Decision intent is translated into simulator-executable action types/targets.
10. Movement/inspection/communication/construction/validation/repair actions execute.
11. Structured events are emitted for transitions, errors, and outcomes.
12. Readiness reconciliation can be triggered by DIK/construction/phase events.
13. Metrics rollups are updated from event stream and authoritative snapshots.
14. Tick time advances and periodic log persistence occurs.

#### 14.2 Cadence controls
Cadence includes step and wall-clock intervals, trigger masks, retries, degraded mode intervals, and backend-specific timeout/token settings.

#### 14.3 Fallback logic and failure recovery
When provider calls timeout/fail/return invalid payloads, fallback behavior can demote to rule-based execution and emit backend/fallback telemetry.

#### 14.4 Action translation and execution
Brain decisions are advisory. Simulator-side legality checks prevent direct world mutation by planner output and protect against malformed or impossible actions.

#### 14.5 Event emission and reconciliation
Subsystems emit structured events; readiness and project status are recomputed on trigger events rather than guessed from narrative assumptions.

#### 14.6 Where authority resides in the simulator
Simulator is authoritative; brains propose actions, engine adjudicates legality/effects.

### 15. Repository / Module Organization

##### Action catalog and diagnostics update (2026-04)

The canonical action vocabulary and action-loss diagnostics are now documented in `Documentation/action_system.md`. The authoritative machine-readable source is `modules/action_catalog.py`, which maps each `ExecutableActionType` to planner visibility, alias normalization, translation destination, execution owner, and implementation status.


## 15.0 Ownership and interaction framing
For each module below, read descriptions as ownership boundaries:
- *Owns state* means authoritative data should be maintained there.
- *Should not own* means avoid hidden cross-layer side effects.
- Cross-module interactions should flow through explicit simulator actions/events.

#### 15.1 `modules/simulation.py`
- **Core responsibility:** authoritative runtime orchestration (`SimulationState`), including per-tick update ordering, planner barrier handling, backend-state refresh, and subsystem wiring.
- **Owns state:** simulation time, configured/effective backend state, planner executor/barrier state, logger and metrics lifecycle control.
- **Should never own:** role-specific domain content (task rules/templates), or private agent cognitive truth beyond explicit agent snapshots.
- **Critical interactions:** `Agent.update`, `Environment.update`, `MetricsCollector.on_step`, `TeamKnowledgeManager.upsert_construction_artifact`.

#### 15.2 `modules/agent.py`
- **Core responsibility:** bounded local cognition, plan/goal management, action selection, communication behavior, and metacognitive regulation.
- **Owns state:** DIK memories, goal stack/registry, planner request state, support/stagnation counters, teammate model approximations, commitment/closure state.
- **Should never own:** direct mutation of global world/project truth outside simulator-executed action pathways.
- **Critical interactions:** brain contract/provider, environment affordances, goal manager/state utilities, task-model rule normalization.

#### 15.3 `modules/construction.py`
- **Core responsibility:** project/site/resource/bridge lifecycle and readiness/validation-relevant project bookkeeping.
- **Owns state:** site capacities/buildability, resource piles, bridge progress, project dictionaries and template instantiation.
- **Should never own:** agent-level policy logic or message semantics.
- **Critical interactions:** environment embedding, transport/build/validate actions, team-knowledge artifact projection.

#### 15.4 `modules/environment.py`
- **Core responsibility:** spatial map, interaction targets/zones, movement and access affordances, phase transitions.
- **Owns state:** geometry/blocked regions, object metadata, source access constraints, task-model-backed layout overrides.
- **Should never own:** planner policy decisions.

#### 15.5 `modules/team_knowledge.py`
- **Core responsibility:** shared artifact and validated-knowledge ledger.
- **Owns state:** externalized artifacts, uptake counts, recent team-level update stream.
- **Should never own:** movement/construction execution logic.

#### 15.6 `modules/brain_provider.py`
- **Core responsibility:** RuleBrain deterministic baseline plus pluggable provider pathways for local/OpenAI-compatible planning.
- **Owns state:** backend configuration, response parsing/normalization, provider-level fallback strategy.
- **Should never own:** authoritative legality enforcement or world mutation.

#### 15.7 `modules/metrics.py`
- **Core responsibility:** event-driven accumulation of run/phase/agent metrics and reason distributions.
- **Owns state:** breakdown counters, phase rollups, planner fallback summaries, movement/construction/communication tallies.
- **Should never own:** primary event-truth generation.

#### 15.8 `interface.py`, `tests/*`, `scripts/*`, and `config/tasks/mars_colony/*`
- **`interface.py` (Implemented):** Tk control/observability surface for interactive experiments.
- **`tests/*` (Implemented):** regression checks for runtime, contracts, backend fallback, construction/readiness behavior, interface resilience, and analysis tools.
- **`scripts/*` (Implemented):** operational validation and audit utilities (`preflight_check.py`, bottleneck/consistency auditors).
- **`config/tasks/mars_colony/*` (Implemented task package):** role defaults, rules, source contents, zones/targets, construction templates, action availability, and manifest defaults.

### 15.8a Task Package Contract

A task package is the mechanism for adapting the simulator to new domains while preserving core architecture.

A complete task package should define:

- environment layout
- zones / coordinates / access rules
- information sources
- role definitions
- structure templates
- rule catalog
- scoring logic
- phase timing and transitions
- action affordances
- default planner parameters
- metrics overlays (optional)

### Design Intent

Core simulator code should remain domain-general.  
Task-specific truth belongs in task packages whenever feasible.

#### 15.9 Architectural invariants
- Simulator remains authoritative over world state mutations.
- Brains are advisory policy layers and do not directly mutate environment/project state.
- Task packages contain domain-specific truth tables and scenario content.
- Metrics derive from events or authoritative snapshots.
- Logging should represent actual transitions, not inferred guesses.

### 15.10 Runtime Invariants

The following assumptions should remain true unless intentionally redesigned:

- Agents do not teleport.
- Brains do not directly mutate world state.
- Construction progress must occur through simulator actions.
- Validation is determined by simulator-side rules.
- Metrics should derive from events or authoritative snapshots.
- Communication emission is not equivalent to communication success.
- Shared knowledge requires uptake, not mere broadcast.
- Task packages define domain truth; planners adapt within those rules.
- Failure states are analytically meaningful and should not be hidden.
---

## Part VI. Agent Cognition and Internal Architecture

### 16. Agent State
**Status:** Implemented core state model with partially implemented/experimental regulation tuning.

#### 16.1 Role identity
Named role-bound agents with role-scoped source access, display labels/template IDs, and per-agent brain/communication parameter overlays.

#### 16.2 Position and movement state
Continuous position/orientation with target/detour tracking, stall counters, blocked movement diagnostics, and path mode settings.

#### 16.3 Inventory / carrying / transport state
Carry state and transport occupancy constraints gate simultaneous action possibilities and affect logistics throughput.

#### 16.4 Current target state
Action target IDs/site bindings/current inspect target state prevent ambiguous execution intent.

#### 16.5 Goal stack / hierarchical goals
Mission/project/support/repair goals are represented through goal stack plus goal registry/order and goal-manager transitions.

#### 16.6 Current plan / adopted plan
Planner state tracks request lifecycle (queued/inflight/completed), adopted method/step, fallback counters, and invalidation events.

#### 16.7 DIK memory
Data/information/knowledge memory sets coexist with source inspection memory, source exhaustion state, known gaps, and post-inspect handoff bookkeeping.

#### 16.8 Rule beliefs
Rule tokens/rule candidates are maintained as action constraints and readiness/validation support evidence.

#### 16.9 Teammate model / expertise beliefs
Theory-of-mind and communication history act as transactive-memory approximations for request routing and deferral.

#### 16.10 Commitment state
Project closure state tracks focused project, commit windows, repair mode, blocker signatures, retry history, and support-focus fatigue.

#### 16.11 Stagnation counters / recovery triggers
Epistemic/support/closure/wait stagnation counters trigger regrounding, support-loop breaks, and controlled switching.

#### 16.12 Trait parameters
Taskwork/teamwork-relevant trait parameters (e.g., help tendency, build speed, communication propensity, rule accuracy) shape behavior and downstream metrics.

#### 16.13 Bounded rationality statement
Agents are non-omniscient: behavior emerges from constraints + planner outputs + available local knowledge + artifact/communication uptake.

### 17. DIK Representation
**Status:** Implemented with active refinement of semantic quality and lineage analytics.

#### 17.1 Data objects
Discrete observed facts from source packets, environment observations, project states, messages, and action outcomes.

#### 17.2 Information objects
Contextualized relevance-bearing interpretations of data relative to role, phase, and current goals.

#### 17.3 Knowledge objects
Integrated action-relevant understanding sufficient to guide communication, planning, and execution choices.

#### 17.4 Rule structures
Canonical `R_*` rule IDs used as operational decision constraints/procedures and validation expectations.

#### 17.5 Confidence / uncertainty / freshness
Confidence/freshness concepts are represented indirectly via planning cadence, fallback/reassessment behavior, and stale-loop diagnostics.

#### 17.6 Provenance and lineage tracking
Events and project provenance snapshots should allow tracing source→derivation→sharing→adoption→construction outcomes.

#### 17.7 DIK failure points explicitly modeled
- missed data acquisition,
- incorrect contextualization,
- weak integration,
- stale beliefs,
- contradictory rules,
- unshared knowledge with no team uptake.

### 18. Internalized Knowledge and Mental Models
#### 18.1 Task mental models
Agent representation of task constraints and structure dependencies.
#### 18.2 Team interaction mental models
Beliefs about communication norms and coordination states.
#### 18.3 Teammate models
Role expertise beliefs used for routing requests/deferrals.
#### 18.4 Problem model representations
Current team challenge framing and candidate solution states.
#### 18.5 Theory-of-mind / transactive memory approximations
Operationalized through teammate model updates and expertise requests.
#### 18.6 Accuracy, partiality, and error
Knowledge can be incomplete, misapplied, delayed, or inconsistent.

### 19. Individual Knowledge Building
#### 19.1 Data selection / attention
Agents choose sources/actions under access and timing constraints.
#### 19.2 Encoding / uptake
Source access adds candidate DIK to local memory.
#### 19.3 Contextualization into information
Data are interpreted relative to role, phase, and goals.
#### 19.4 Integration into knowledge
Derivations/rule adoption integrate multiple prerequisite elements.
#### 19.5 Individual synthesis and self-talk / self-management
Reassessment actions and planner invocations support self-regulation.
#### 19.6 Knowledge object development
Knowledge objects mature via new evidence and correction episodes.

---

## Part VII. Team Knowledge Building and Communication

### 20. Team Knowledge Building Processes
**Status:** Implemented baseline with partial scaffolds for richer negotiation semantics.

#### 20.1 Team information exchange
Role-specific content is communicated through explicit message intents and logged as causal runtime events.
#### 20.2 Team knowledge sharing
Rule/knowledge payload sharing supports convergence when recipients actually integrate received content.
#### 20.3 Team solution option generation
Plan proposals and externalized artifacts create explicit alternative pathways.
#### 20.4 Team evaluation and negotiation of alternatives
Agreement/repair acts and artifact updates negotiate plans; multi-turn deep negotiation remains partially implemented.
#### 20.5 Team process and plan regulation
Meta-level coordination actions guide sequencing and commitment windows.
#### 20.6 Consensus and co-validation
Consensus is indicated by joint artifact adoption, coordinated execution, and validated outcomes (not message volume alone).
#### 20.7 Knowledge interoperability and repair
Mismatch detection triggers clarification/repair loops with possible fallback to source reinspection.

### 21. Communication System
**Status:** Implemented core executable communication intents; extended discourse taxonomy partly analytic; richer discourse planned.

#### 21.1 Why communication is modeled as cognitive processing
Messages can alter DIK state, teammate models, and subsequent action selection.
#### 21.2 Communication as observable and causal
Communication events are explicit in logs and can be linked to downstream execution outcomes.
#### 21.3 Conditions for speaking, listening, acknowledging, adopting
Action legality, proximity, timing, and state determine opportunities for send/receive/adopt behavior.
#### 21.4 Communication timing, proximity, and access
Spatial and temporal context mediates message production and uptake lag.
#### 21.5 Directed versus broadcast communication
System supports sender/recipient targeting while preserving team-level artifact pathways.
#### 21.6 One-way, acknowledged, and integrated communication
Not all messages are adopted; uptake and integration determine whether communication changes team cognition.

#### 21.7 Communication failure modes
- message not sent,
- message ignored,
- delayed uptake,
- misunderstood payload,
- not integrated into plan/action policy.

### 22. Full Communication Coding Taxonomy
#### 22.1 Interpersonal and communication management
- **SR:** social regulation to maintain interaction.
- **AM:** acknowledgement management.
- **DCM:** dialogue control management.
- **MCB:** metacommunication boundary-setting.
- **OFF:** off-task speech marker.
- **MISC:** miscellaneous interpersonal content.
- **OTH:** other uncategorized communication.
#### 22.2 Knowledge management
- **IKM:** individual knowledge management acts.
- **IDG:** individual data generation.
- **IIG:** individual information generation.
- **IIS:** individual information sharing/support.
- **TDP/TIP/TKP:** team data/information/knowledge provision (implemented).
- **TDR/TIR:** team data/information requests.
- **TKR/TKE:** team knowledge request/elaboration.
#### 22.3 Goal/task orientation and problem framing
- **TGTO:** goal/objective framing (implemented).
- **TIPS:** team interpretation/problem structuring.
#### 22.4 Planning and solution development
- **TSOG:** team solution-option generation.
- **TENA:** team evaluation/negotiation of alternatives.
#### 22.5 Coordination and monitoring
- **TC:** explicit coordination.
- **TTM:** time/tempo management.
- **TU:** task update.
- **TUR:** task update request.
#### 22.6 Reflection and outcomes
- **TPR:** team process reflection.
- **ER:** error reflection/repair reflection.
- **PR:** performance reflection.
- **TPSO:** team problem-solving outcome summary.
#### 22.7 Mapping codes to simulation functions
Core implemented communication intents are `TDP`, `TIP`, `TKP`, `TGTO`, `TKRQ`, `TCR`, `TPP`, and `TPA`; additional taxonomy codes are a measurement-facing extension scaffold. Core codes change DIK/team state directly; optional codes enrich discourse realism and analytic granularity.

**Status split:**
- **Implemented:** core executable communication intents tied to simulator state transitions and measurable uptake effects.
- **Partially Implemented:** extended analytic taxonomy labels with limited direct execution semantics.
- **Planned:** richer multi-turn discourse and negotiation state tracking.

Communication quality should be interpreted by uptake/integration outcomes, not emission counts alone.

### 23. Shared Mental Models and Convergence
#### 23.1 What counts as convergence
Increased overlap of task constraints, project priorities, and teammate expectations.
#### 23.2 Task mental model similarity
Similarity in structure-rule-goal understanding across agents.
#### 23.3 Team interaction knowledge similarity
Similarity in expectations about coordination/communication patterns.
#### 23.4 Teammate knowledge similarity
Alignment in beliefs about who holds which expertise.
#### 23.5 Shared situation awareness
Common understanding of phase demands, bottlenecks, and project status.
#### 23.6 Temporal emergence of shared models
Convergence appears through repeated exchange, artifact uptake, and validation feedback.
#### 23.7 When misalignment is adaptive vs harmful
Temporary divergence can support exploration; persistent divergence harms readiness/validation.

### 24. Transactive Memory and Expertise Routing
#### 24.1 Metaknowledge of who knows what
Role-linked beliefs and interaction history form expertise maps.
#### 24.2 Request routing
Knowledge requests are directed to likely experts.
#### 24.3 Deferral to expertise
Agents can defer decisions/actions to role-appropriate teammates.
#### 24.4 Updating teammate expertise beliefs
Successful responses and validation outcomes update expertise confidence.
#### 24.5 Role authority and knowledge ownership
Role briefs establish initial authority boundaries that can later be revised by evidence.

---

## Part VIII. Externalization, Artifacts, and Environmental Embedding

### 25. Externalized Agent Knowledge
#### 25.1 What counts as agent-level externalization
Single-agent notes/messages/actions made public but not yet co-validated.
#### 25.2 Externalized-but-not-yet-shared knowledge
Public output may remain unadopted.
#### 25.3 Uptake lag and failed uptake
Time gaps or non-adoption are modeled as meaningful process variables.
#### 25.4 Importance for modeling incomplete sharing
This captures realistic communication inefficiency and miscoordination.

### 26. Externalized Team Knowledge
#### 26.1 Co-authored / co-validated artifacts
Team artifacts become shared when multiple agents adopt/revise/act on them.
#### 26.2 Shared plans, layouts, summaries, protocols
Whiteboard plans and project-level records represent team-level cognition.
#### 26.3 Why team externalization is not just many individual externalizations
Collective endorsement and reuse distinguish team-level artifacts.
#### 26.4 Indicators of genuine convergence and collective authorship
Multi-agent consultation, agreement, and consistent execution trajectories.

### 27. Whiteboards, Plans, and Knowledge Objects
#### 27.1 Whiteboard affordances in the human task
Central, persistent, revisable coordination medium.
#### 27.2 Notes, diagrams, and spatial plans
Supports task decomposition and spatial coordination.
#### 27.3 Artifact authorship and persistence
Artifacts have history, persistence, and revision trajectories.
#### 27.4 Artifact consultation and update
Consultation and update actions are explicit and measurable.
#### 27.5 Artifact lineage and re-use
Artifact-derived decisions can be traced later in event logs.

### 28. Externalized DIK as Environmental DIK
#### 28.1 Reclassification after externalization
Public externalizations become environment-available DIK.
#### 28.2 Persistent environmental knowledge objects
Artifacts and build state persist as cognitive resources.
#### 28.3 Re-consumption and reinterpretation
Agents can revisit and reinterpret old artifacts under new context.
#### 28.4 Delayed coordination and artifact-mediated repair
Artifacts allow asynchronous alignment and delayed repair.
#### 28.5 DIK transience and persistence rules
Some events are transient signals; artifacts/project states are persistent.

---

## Part IX. Taskwork, Construction, and Embodied Cognition

### 29. Construction as Taskwork
#### 29.1 Resource transport
Transport actions move bricks from resource node to build sites.
#### 29.2 Material staging
Projects track staged materials.
#### 29.3 Build steps
Start/continue construction actions accumulate build progress.
#### 29.4 Connection logic
Bridges/infrastructure alter site accessibility and dependencies.
#### 29.5 Project states
Projects move through created, in-progress, ready, completed, validated, repair states.
#### 29.6 Validation states
Validation checks expected rules and state sufficiency.

### 30. Construction as Epistemic Externalization
#### 30.1 Why construction is not only output
Build choices reveal internal assumptions.
#### 30.2 Taskwork as expression of mental models
Incorrect sequencing/configuration indicates model mismatch.
#### 30.3 Incorrect construction as evidence of knowledge divergence
Misbuilds are diagnostic, not just failures.
#### 30.4 Construction outputs as environmental knowledge objects
Built artifacts become shared evidence for future decisions.
#### 30.5 Construction-triggered diagnosis and repair
Validation failures trigger epistemic repair loops.

### 31. Project Model
#### 31.1 What a project is
A template-grounded structure objective tied to site/constraints.
#### 31.2 Project identity
`project_id` plus structure type and target site.
#### 31.3 Project dependencies
Expected rules and unlock constraints.
#### 31.4 Material satisfaction
Required resources must be staged.
#### 31.5 Physical completeness
Construction progress must meet completion criteria.
#### 31.6 Epistemic sufficiency
Required rule support/knowledge must be available.
#### 31.7 Readiness
Project marked ready when prerequisites are met.
#### 31.8 Validation
Explicit validation confirms acceptable completion.
#### 31.9 Repair episodes
Mismatches trigger correction iterations.

### 32. Construction / Validation Loop
#### 32.1 Material delivery
Gather/transport/stage materials.
#### 32.2 Start/continue construction
Initiate and advance build.
#### 32.3 Detect mismatch
Check readiness/validation for inconsistencies.
#### 32.4 Seek/repair knowledge
Inspect sources, communicate, revise plan/artifact.
#### 32.5 Retry build or validate
Resume execution then re-attempt validation.
#### 32.6 Final completion and colony contribution
Validated projects contribute to mission viability.

---

## Part X. Goals, Plans, and Metacognitive Regulation

### 33. Goal System
#### 33.1 Mission goals
Colony survivability and phase requirement goals.
#### 33.2 Project goals
Structure-specific completion/validation goals.
#### 33.3 Support goals
Knowledge acquisition, communication, and repair subgoals.
#### 33.4 Goal activation and suppression
Context-dependent goal promotion/demotion.
#### 33.5 Goal persistence and demotion
Unproductive goals can be deprioritized after evidence of stagnation.

### 34. Planning System
#### 34.1 Local plans
Per-agent step sequences for current objective.
#### 34.2 Team plans
Shared externally visible plans.
#### 34.3 Plan externalization
Plans can be published to team artifacts.
#### 34.4 Plan adoption
Agents can adopt own or teammate plans.
#### 34.5 No-active-plan recovery
Fallback/reassessment logic restores actionable state.

### 35. Problem-Solving Phases in the Simulator
#### 35.1 Knowledge construction
Inspect/derive/share foundational DIK.
#### 35.2 Team problem model
Represent dependencies, bottlenecks, sequence constraints.
#### 35.3 Consensus
Achieve enough alignment for coordinated execution.
#### 35.4 Evaluation/revision
Use outcomes to revise assumptions/plans.
#### 35.5 Phase transitions as time-based or behavior-triggered
Both clock progression and event triggers can drive replanning.

### 36. Metacognitive and Regulatory Layer
#### 36.1 Monitoring
Track progress, failures, and stale plans.
#### 36.2 Stagnation detection
Identify no-effect loops and repeated blockers.
#### 36.3 Regrounding
Force fresh information acquisition or model updates.
#### 36.4 Execution windows / commitment
Dwell windows stabilize commitment before switching.
#### 36.5 Project switching
Controlled switching prevents fixation and thrash.
#### 36.6 Tempo management
Cadence controls planner frequency and responsiveness.
#### 36.7 Plan/process regulation
Policies manage transitions across planning/execution/repair.
#### 36.8 Known regulation failure modes
- overfiring,
- churn,
- weak successor policies,
- premature switching.

### 37. Closure, Repair, and Validation
#### 37.1 Closure requirements
Closure requires material and epistemic sufficiency.
#### 37.2 Blockers
Can be physical, informational, or coordination-based.
#### 37.3 Knowledge repair
Repair obtains missing support and resolves contradictions.
#### 37.4 Uncertainty resolution
Agents reduce uncertainty via inspection, requests, and artifact checks.
#### 37.5 Return to execution/validation
After repair, execution resumes and validation is retried.
#### 37.6 Rigid-rule pitfalls and semantic sufficiency
Over-rigid checks can reject semantically acceptable states; balance is required.

---

## Part XI. Brain Modularity and Experimental Control

### 38. Brain Layer
#### 38.1 RuleBrain baseline
Deterministic/interpretable default policy for reproducible comparisons.
#### 38.2 Why a deterministic baseline matters
Supports attribution and debugging without model stochasticity confounds.
#### 38.3 Modular provider pathway for alternative models
`create_brain_provider` routes backend implementations behind a stable contract.
#### 38.4 Local model / OpenAI-compatible backends
Supports local HTTP and OpenAI-compatible model endpoints with fallback controls.
#### 38.4a Planner Decision Contract

External planners should return structured decisions rather than free-form prose.

Preferred decision fields include:

- `action_type`
- `target_id`
- `goal_id`
- `reason`
- `confidence`
- `requested_followup`
- `notes`

Example:

```json
{
  "action_type": "inspect_source",
  "target_id": "Engineer_Info",
  "goal_id": "acquire_water_rules",
  "reason": "Need missing constraints for generator readiness",
  "confidence": 0.82
}
#### 38.5 Why backend is an experimental variable
Backend differences affect planning latency, robustness, and cognitive behavior.

### 39. Multi-Level Entry Points for Alternative Brains
#### 39.1 Metacognitive regulation entry point
Alternative controllers can tune switching, repair, and cadence logic.
#### 39.2 Taskwork planning entry point
Alternative planners can choose transport/build sequencing.
#### 39.3 Teamwork planning entry point
Alternative planners can optimize communication and coordination choices.
#### 39.4 Epistemic interpretation entry point
Alternative models can reinterpret DIK and uncertainty handling.
#### 39.5 Why these seams matter experimentally
They enable controlled substitutions without rewriting simulator physics/state authority.

---

## Part XII. Parameters, Manipulations, and Experimental Design

### 40. Agent-Level Parameters
#### 40.1 Taskwork potential
Baseline execution strength for physical progress.
#### 40.2 Teamwork potential
Baseline coordination/communication contribution.
#### 40.3 Build speed
Affects construction action throughput.
#### 40.4 Rule accuracy / inference quality
Affects correctness of derived interpretations.
#### 40.5 Goal alignment
Bias toward team mission consistency.
#### 40.6 Help tendency
Propensity to assist/respond.
#### 40.7 Persistence
Tendency to continue current strategy.
#### 40.8 Attention allocation
Source/target prioritization behavior.
#### 40.9 Planning cadence
Frequency/timing of replanning.
#### 40.10 Uncertainty response
Whether uncertainty drives inspection/request/deferral.
#### 40.11 Drift susceptibility / maintenance tendency
Likelihood of losing alignment without maintenance communication.

### 41. Team-Level and Contextual Manipulations
#### 41.1 Information asymmetry
Change source distribution/access.
#### 41.2 Role distribution
Alter role counts/activation.
#### 41.3 Communication constraints
Throttle, filter, or delay communications.
#### 41.4 Environmental constraints
Modify pathing, access, or site availability.
#### 41.5 Time pressure
Shorten phase durations.
#### 41.6 Resource scarcity
Reduce resource availability/capacity.
#### 41.7 Dynamic disruptions
Inject unexpected failures or changing conditions.
#### 41.8 Stressors and uncertainty
Increase ambiguity and perturbation frequency.

### 42. Experiment Design Uses
#### 42.1 Profile comparisons
Compare trait/construct profiles.
#### 42.2 RuleBrain vs LLM backend comparisons
Evaluate architecture effects under identical tasks.
#### 42.3 Externalization manipulations
Vary artifact affordances and persistence.
#### 42.4 Shared-vs-fragmented knowledge manipulations
Control overlap of initial packet access.
#### 42.5 Metacognitive support conditions
Add/remove regulation aids.
#### 42.6 Construction-design tradeoffs
Study planning quality vs build tempo.
#### 42.7 Bridge/zone access strategy comparisons
Compare unlock-centric vs near-term throughput strategies.

---

## Part XIII. Measurement, Logging, and Analysis

### 43. Measurement Philosophy
#### 43.1 Need for integrative measurement
No single metric captures macrocognitive quality.
#### 43.2 Individual-level, team-level, artifact-level, and outcome-level measurement
Use multilevel measures to avoid reductionism.
#### 43.3 Static vs sequential / temporal measurement
Combine end-state summaries with time-ordered traces.
#### 43.4 Why logs and traces matter for causal diagnosis
Ordered events reveal mechanism pathways and failure points.

### 44. Event Logging Architecture
**Status:** Implemented event architecture with ongoing semantic tightening of labels and derived metrics.

#### 44.1 Event philosophy
Every meaningful state transition should emit inspectable events.

#### 44.2 Categories of events
movement; source access; DIK; communication; planning; artifact creation; construction; readiness reconciliation; validation; repair; metacognition; backend events; errors/failures.

#### 44.3 Run artifacts

Typical run outputs include:

- **run_summary.json / csv**  
  Final outcomes, counters, durations, reason distributions.

- **events.jsonl / csv**  
  Time-ordered structured event stream.

- **metrics_rollup.json / csv**  
  Aggregated agent/team/process metrics.

- **planner_trace.jsonl**  
  Planner requests, responses, retries, fallback outcomes.

- **backend_trace.jsonl**  
  Provider latency, parse failures, invalid payloads, timeout telemetry.

- **project_state_dump.json**  
  Final and intermediate project states.

- **config_snapshot.json**  
  Effective settings used for reproducibility.

#### 44.4 Metric separation for analysis
1. **Already implemented:** direct counters/rollups in summaries and metrics collector.
2. **Derivable now:** sequence/network/window analyses from existing event traces.
3. **Conceptual future metrics:** richer semantic and discourse quality metrics not yet fully operationalized.

#### 44.5 Event trace consolidation and multimodal compatibility
Outputs are structured to support cross-tool integration and downstream fusion.

### 45. Measurement Framework by MITM Component
#### 45.1 Individual knowledge building measures
Source access counts, DIK events, derivation success/failure traces.
#### 45.2 Internalized team knowledge measures
Task/interaction/teammate model similarity and shared awareness proxies.
#### 45.3 Team knowledge building measures
Exchange volume, option generation, alternative evaluation, regulation acts, temporal communication flow.
#### 45.4 Externalized team knowledge measures
Artifact quality, uptake, uncertainty-resolution traces.
#### 45.5 Team problem-solving outcomes
Plan quality, planning efficiency, execution efficiency, colony viability outcomes.

### 46. Communication Analytics
#### 46.1 Static content measures
Type frequencies and distributions.
#### 46.2 Static flow measures
Sender/receiver counts and centrality-like summaries.
#### 46.3 Sequential content/flow measures
Transition probabilities and sequence motifs.
#### 46.4 Moving-window analyses
Windowed rates for dynamic regime detection.
#### 46.5 Anticipation ratios
Leading vs trailing information exchanges.
#### 46.6 Following/dominance patterns
Turn-taking and directional influence.
#### 46.7 Generate/clarify/evaluate/reduce patterns
Problem-solving speech function progression.
#### 46.8 Communication-network analysis possibilities
Construct temporal conversation graphs per phase.

### 47. Knowledge and Alignment Metrics
#### 47.1 Overlap indices
Set overlaps on DIK/rules/goals across agents.
#### 47.2 Convergence/divergence
Distance trajectories over time.
#### 47.3 Knowledge lineage
Trace output states back to source/derivation events.
#### 47.4 Uptake lag
Delay between externalization and adoption.
#### 47.5 Recognition of expertise
Accuracy of who-knows-what beliefs.
#### 47.6 Shared-problem-model quality
Consistency and sufficiency of cross-agent problem framing.

### 48. Movement and Embodiment Metrics
#### 48.1 Path efficiency
Distance/time vs shortest plausible route.
#### 48.2 Zone dwell time
Time spent by functional zone.
#### 48.3 Site access success/failure
Successful arrivals vs blocked attempts.
#### 48.4 Construction-site occupancy
Presence patterns at build sites.
#### 48.5 Embodied coordination measures
Co-location timing, handoff timing, interference patterns.

### 49. Construction and Validation Metrics
#### 49.1 Material progress
Resources staged by project over time.
#### 49.2 Build-step progress
Construction action completion counts.
#### 49.3 Connection completion
Bridge/connection milestones.
#### 49.4 Repair episodes
Count/duration of repair cycles.
#### 49.5 Validation attempts
Validation trial frequency.
#### 49.6 Validation success/failure
Outcome distributions and reasons.
#### 49.7 Colony survivability outcomes
Whether required support structures are achieved by phase.

### 50. Process and Emergent-State Metrics
#### 50.1 Coordination
Indices from synchronization/handoff consistency.
#### 50.2 Communication
Quality and relevance-adjusted communication metrics.
#### 50.3 Adaptation
Response quality to disruptions/contradictions.
#### 50.4 Planning quality
Plan completeness/coherence and update quality.
#### 50.5 Mood/affect/motivation if modeled
Optional extension variables.
#### 50.6 Viability / willingness to remain coordinated
Persistence of cooperative behavior under strain.
#### 50.7 Cohesion / confidence / conflict if modeled
Optional social-state extensions.

---

## Part XIV. Verification, Validation, and Debugging

### 51. What Counts as a Working System
#### 51.1 Theoretical fidelity
Mechanisms map correctly to intended MITM constructs.
#### 51.2 Behavioral plausibility
Agent/team trajectories look credible under constraints.
#### 51.3 Technical correctness
No schema/runtime violations; deterministic baselines stable.
#### 51.4 Experimental usefulness
System supports controlled manipulation and interpretable outputs.

### 52. Verification Strategy
#### 52.1 Unit tests
Component-level correctness checks.
#### 52.2 subsystem tests
Cross-module interaction checks.
#### 52.3 scenario tests
End-to-end task package execution checks.
#### 52.4 run-level audits
Post-run consistency and artifact audit scripts.

### 53. Validation Strategy
#### 53.1 Face validity with the human task
Task package mirrors intended human CPS constraints.
#### 53.2 Alignment with MITM constructs
Observed runtime events correspond to conceptual constructs.
#### 53.3 Comparison to known team-cognition patterns
Check qualitative plausibility against established findings.
#### 53.4 Comparison across conditions
Expected condition effects should appear in metrics.
#### 53.5 Avoiding magical success shortcuts
No hidden omniscience or direct completion bypasses.

### 54. Common Failure Modes
#### 54.1 Startup failures
Task package/config/backend initialization issues.
#### 54.2 Project-binding failures
Action-to-project linkage errors.
#### 54.3 Transport churn
Repeated non-productive logistics loops.
#### 54.4 Readiness/validation handoff failures
Ready states not transitioning cleanly to validated outcomes.
#### 54.5 Over-regulation / churn
Excessive switching without progress.
#### 54.6 Movement/pathing bottlenecks
Blocked or inefficient movement trajectories.
#### 54.7 Artifact underuse / externalization failures
Plans/messages not adopted or consulted.
#### 54.8 Misleading metrics semantics
Metric labels interpreted beyond supported meaning.

### 55. How to Diagnose Runs
#### 55.1 Read run summary first
Establish top-level outcome signature.
#### 55.2 Inspect major counters and reason distributions
Use rollups to locate dominant failure classes before line-by-line trace reading.
#### 55.3 Then event log timeline
Locate first divergence point and subsequent blocker propagation.
#### 55.4 Then planner/backend traces
Inspect decision-level causes and malformed/fallback patterns.
#### 55.5 Classify failure type
Separate epistemic, coordination, execution, environment/infrastructure failures.
#### 55.6 Patch responsible subsystem and re-run controlled comparison
Confirm fixes by rerunning same condition with minimal parameter drift.

---

## Part XV. Current Status, Open Issues, and Roadmap

### 56. Current Implemented Capabilities

#### 56.1 Stable Core (**Status: Implemented**)
- Task-driven Mars package ingestion and environment/construction instantiation.
- Authoritative simulator tick loop with event emission.
- DIK acquisition/derivation scaffolding and core communication intents.
- Construction/readiness/validation loop with artifact publication.
- RuleBrain baseline and pluggable brain provider pathway.
- Logging/metrics/test scaffolding sufficient for repeatable experimentation.

#### 56.2 Actively Tuned (**Status: Partially Implemented**)
- Metacognitive switching thresholds and successor-policy handoffs.
- Async planner cadence/degraded mode behavior under high latency.
- Readiness reconciliation timing and semantic sufficiency balancing.
- Artifact uptake quality and communication-effectiveness interpretation.

#### 56.3 Experimental (**Status: Experimental**)
- Extended communication taxonomy as executable behavior (beyond core intents).
- Richer convergence and discourse quality metrics.
- Some advanced commitment and closure heuristics under diverse stressors.

#### 56.4 Planned (**Status: Planned**)
- Richer dialogue/negotiation state machines.
- Additional CPS task packages beyond Mars Colony.
- Stronger semantic validation and cross-domain transfer utilities.
- Human-in-the-loop and adaptive support-agent workflows.

### 57. Current Bottlenecks
#### 57.1 Successor policy handoffs
Switching between methods/modes can still induce inefficiency.
#### 57.2 Construction commitment
Premature switching can interrupt build completion.
#### 57.3 Readiness reconciliation
Edge cases remain around trigger timing and state reconciliation.
#### 57.4 Movement/path execution issues
Some runs exhibit unnecessary path churn.
#### 57.5 Shared knowledge convergence limits
Uptake delays can leave teams fragmented.
#### 57.6 Balancing epistemic richness with completion
Richer cognition can increase runtime complexity and slower completion.
#### 57.7 Fallback overuse patterns
Under unstable provider conditions, repeated fallback can mask deeper planner-contract issues.

### 58. Future Directions
#### 58.1 Stronger dialogue and negotiation
Add richer conversational act handling and negotiation state.
#### 58.2 Better construction semantics
Increase semantic validity checks beyond template compliance.
#### 58.3 Better multi-agent commitment dynamics
Improve durable joint commitments and handoff policies.
#### 58.4 Additional task packages beyond Mars Colony
Generalize to new CPS domains via task package architecture.
#### 58.5 Hybrid human-AI teams
Inject human-in-the-loop controls and mixed-agent runs.
#### 58.6 Adaptive tutoring/support agents
Add facilitative agents for coaching/repair nudges.
#### 58.7 Real-time analytics and intervention systems
Stream diagnostics for live experiment steering.

---

## Part XVI. Example Studies and Practical Applications

### 59. Example Research Studies
#### 59.1 Shared mental model emergence
Track convergence trajectories across phases.
#### 59.2 Knowledge asymmetry and expertise routing
Vary asymmetry and observe request-routing quality.
#### 59.3 Communication coding and pattern analysis
Map message sequences to outcomes.
#### 59.4 Repair and uncertainty resolution
Study repair loops after contradiction events.
#### 59.5 Taskwork as epistemic externalization
Use build traces as cognitive evidence.
#### 59.6 LLM vs rule-based decision systems
Compare planning robustness and interpretability.
#### 59.7 Construction-domain collaborative learning
Evaluate how repeated runs improve policy quality.

### 60. Practitioner / Applied Uses
#### 60.1 Training design
Prototype team training scenarios around coordination failure points.
#### 60.2 Measurement system prototyping
Evaluate candidate team-cognition metrics before field deployment.
#### 60.3 Intelligent teammate design
Test assistant policies for communication and repair support.
#### 60.4 Team diagnostics
Identify bottlenecks in coordination and information flow.
#### 60.5 Human-AI teaming evaluation
Assess mixed-policy compatibility and trust/workload implications.
#### 60.6 Construction / mission planning analogs
Apply patterns to mission engineering and constrained operations.

---

## Part XVII. Discussion and Conclusion

### 61. Integrating Theory, Measurement, and Simulation
#### 61.1 Why this testbed matters scientifically
It enables mechanistic tests of macrocognition rather than only correlational snapshots.
#### 61.2 Why it matters methodologically
It unifies manipulations, traces, and metrics under reproducible control.
#### 61.3 Why it matters for intelligent systems
It provides a benchmark for coordination-capable AI decision architectures.

### 62. Limitations
#### 62.1 Abstraction limits
Simulator necessarily simplifies human cognition/social behavior.
#### 62.2 fidelity tradeoffs
Higher fidelity can reduce tractability and reproducibility.
#### 62.3 current implementation limits
Some conceptual components are scaffolds rather than fully realized modules.
#### 62.4 domain specificity vs generalizability
Mars construction is one CPS domain; transfer requires cautious adaptation.
#### 62.5 What is simulated vs abstracted

The simulator models many constructs functionally rather than biologically or phenomenologically.

Examples:

Modeled directly:
- movement
- access constraints
- communication events
- planning decisions
- resource transport
- build progress
- validation outcomes

Abstracted proxies:
- motivation
- trust
- stress
- confidence
- shared understanding
- theory of mind
- affect

These constructs are represented through state variables, tendencies, event patterns, or decision consequences rather than literal human cognition.

### 63. Conclusion
MITM can function as executable science; Mars Colony offers an embodied epistemic teamwork testbed; simulation bridges theory, measurement, and intervention.

---

## Back Matter / Appendices

### Appendix A. Full Expanded MITM Mapping
- **original MITM:** Individual/team knowledge building, internalized/externalized knowledge, outcomes.
- **expanded MITM:** Adds environmental DIK channels, recursive loops, explicit taskwork, and externalized→environmental feedback.
- **simulator mappings:** `task_model` (inputs), `agent`/`knowledge` (internalized processing), `team_knowledge` (shared state), `construction` (taskwork externalization), `metrics/logging` (measurement).

### Appendix B. Full Communication Codebook
- **all code definitions:** Core executable codes (`TDP`, `TIP`, `TKP`, `TGTO`, `TKRQ`, `TCR`, `TPP`, `TPA`) plus extended analytic taxonomy families.
- **examples:** knowledge request→knowledge provision→plan proposal→agreement sequence.
- **simulation effects:** adoption updates DIK/shared state; non-adoption contributes to divergence.

### Appendix C. Full Measurement Matrix
- **constructs:** individual/team/artifact/outcome/macrocognitive constructs.
- **measures:** event counts, ratios, sequence motifs, convergence indices.
- **event sources:** runtime event stream, planner traces, project state summaries.
- **possible indices:** overlap, lag, centrality, efficiency, viability.
- **aggregation level:** per-event, per-agent, per-phase, per-run, cross-condition.

### Appendix D. Event Taxonomy
- **simulation events by category:** movement, DIK, communication, planning, construction, validation, regulation, errors.
- **key log fields:** event type, timestamp, actor, targets, payload reason/context.
- **example traces:** source access → derivation → communication → build → validation → repair/retry.

### Appendix E. Parameter Catalog
- **all agent and system parameters:** construct/mechanism/hook parameters, planner cadence/timeout/fallback, construction parameters.
- **defaults:** loaded from task manifest and config files.
- **experimental meaning:** each parameter maps to behavioral hypotheses and expected process effects.

### Appendix F. Mars Colony Task Rules
- **role briefs:** Architect/Engineer/Botanist scoped sources.
- **phase objectives:** phase population targets and required structures.
- **structure requirements:** template-based resource/rule expectations.
- **cart requirements:** carrying capacity and movement-time scaling.
- **whiteboard use:** shared artifact externalization/consultation.
- **bridge/resource logic:** Site C unlock via bridge condition; shared brick resource competition.

### Appendix G. Example Annotated Run Walkthrough
- **selected event trace:** identify a run segment with acquisition, planning, construction, validation.
- **interpretation by subsystem:** map each segment to environment/agent/knowledge/team/construction layers.
- **failure mode classification:** classify as epistemic, coordination, execution, or mixed.

### Appendix H. Reproducibility / Repo Use Guide
- **repository structure:** `modules/`, `config/tasks/mars_colony/`, `tests/`, `scripts/`, `Documentation/`.
- **setup:** install requirements and run preflight checks.
- **configuration:** choose task package, backend, planner, and manipulation profiles.
- **running experiments:** GUI or headless batch paths.
- **reading outputs:** session summaries, event logs, rollups, and planner/backend traces.

## Glossary

**Agent** — Autonomous simulated team member.  
**Artifact** — Persistent external representation used cognitively.  
**Backend** — Decision provider used by planner layer.  
**DIK** — Data, Information, Knowledge.  
**Goal** — Desired future state driving behavior.  
**MITM** — Macrocognition in Teams Model.  
**Project** — Structure objective with requirements and lifecycle state.  
**Readiness** — Condition indicating a project may be validated.  
**RuleBrain** — Deterministic baseline planner.  
**Task Package** — Domain-specific configuration set.  
**Tick** — One simulator update cycle.  
**Validation** — Rule-based confirmation of acceptable completion.