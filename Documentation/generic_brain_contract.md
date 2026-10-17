# Generic Brain Contract (Simulator-Owned Execution)

## Responsibilities
- **Simulator owns truth/execution**: world state, legal action filtering, action execution, cadence/trigger schedule, memory persistence, DIK derivations, construction rules, logging, metrics.
- **Brain provider owns cognition**: prioritize goals/subgoals, produce short-horizon ordered actions, propose immediate `next_action`, optional uncertainty and optional explanation.

## Request / Response contract
- Request model: `AgentBrainRequest` (`modules/brain_contract.py`).
- Response model: `AgentBrainResponse` with `AgentPlan` containing:
  - `ordered_goals`
  - `ordered_actions`
  - `next_action`
  - `confidence`
- JSON schemas are exported as `BRAIN_REQUEST_JSON_SCHEMA` and `BRAIN_RESPONSE_JSON_SCHEMA`.

## Explanation scheduling
Configured simulator-side in planner config:
- `explanation_mode`: `never | every_n_calls | probability | always`
- `explanation_every_n_calls`
- `explanation_probability`

Simulator sets `request_explanation` on each `AgentBrainRequest`; simulator does not require explanation to execute.

## Generic agents from data
Agents are now assembled from task data + optional template inheritance:
- `agent_id`, `display_name`, `label`
- `template_id`
- `accessible_packet_ids`
- `initial_goal_seeds`
- `planner_config`, `brain_config`, communication params

Mars Architect/Engineer/Botanist behavior remains represented via templates/default rows in task package data.

## Per-agent controllers (`modules/controller_spec.py`)
Controller assignment is **per agent**, never per team/session. Each role resolves independently to a `ControllerSpec`:
`{"controller_type": "procedural_baseline" | "local_ollama", "settings": {...}}`. The experimental profile (gestalts, seven constituents) is configured
separately and is not part of controller identity. Only `procedural_baseline` is enabled; `local_ollama` is a declared placeholder whose reserved settings are
`model, temperature, controller_generation_seed, max_context_tokens, max_output_tokens, timeout_s, prompt_version, inference_parameters` (a procedural spec carries none).
A declared-but-unimplemented controller can be described, serialised and recorded in a manifest but raises `ControllerNotEnabledError` if a simulation tries to instantiate it.

Resolution path (the Environment and Middleware contain no team-level controller conditionals): `SimulationState(controller_assignments=...)` (or `experimental_settings["controller_assignments"]`,
or an agent config's `controller`) -> per-agent `ControllerSpec` -> that agent's `brain_config` backend -> its own provider + `PilotAdapter` runtime -> the canonical Pilot contract.
Headless: `--controller Botanist=procedural_baseline` (repeatable) or `controller_assignments` in a condition JSON / batch condition. Controller composition is part of the condition identity
(`condition_identity`), of `session_manifest.json -> controller_assignments` (per agent: agent_id, role, controller + implementation/version, experimental profile + resolved seven, seeds) and of every
`pilot_selection_via_adapter` event (`controller_type`, `controller_agent_id`).
Seeds: the world seed (`SimulationState.seed`) is separate from the reserved per-agent `controller_generation_seed`; resolving controller settings never touches the world RNG.
GUI: each agent card has its own Controller selector (Procedural Baseline only for now) plus a reserved per-agent frame for that controller's settings; convenience buttons reset/apply to all.

## Local Ollama controller (`controller_type = local_ollama`; local inference only)
Modules: `ollama_client.py` (loopback-only transport), `ollama_prompt.py` (canonical prompt + deterministic budget), `ollama_pilot.py` (PilotAdapter + preflight), `controller_spec.py` (settings).
- **Single selection path.** Controller identity comes only from `ControllerSpec`. A `brain_backend` string can no longer build an Ollama provider (`create_brain_provider` raises `LegacyOllamaBackendRetiredError`);
  the legacy aliases (`local_http`, `openai_compatible_local`, `ollama_local`, `ollama`) given globally or per agent are ROUTED by `SimulationState` to an equivalent `local_ollama` spec (event
  `legacy_ollama_backend_routed_to_controller_spec`; a non-loopback legacy endpoint is rejected). The GUI no longer offers them.
- **Connection.** Plain http to 127.0.0.0/8, `::1` or `localhost` only (checked before any socket; env proxies ignored; redirects refused; no credentials). Only `GET /api/version`, `GET /api/tags`, `POST /api/chat`
  are ever called: no pull/delete/create, no service start, no API key. Preflight (service reachable + exact installed model tag) runs before the simulation is constructed; failure raises `ControllerUnavailableError`
  (headless) or blocks Start with a dialog (GUI).
- **Settings (per agent).** `model` (required), `base_url`, `temperature`, `controller_generation_seed` (Ollama `options.seed`), `max_context_tokens` (`num_ctx`), `max_output_tokens` (`num_predict`), `prompt_budget_tokens`,
  `timeout_s`, `prompt_version`, `max_retries`, `failure_policy` (`stop_run` | `idle_and_log`), `think` (default false), `inference_parameters` (safe options only: top_p, top_k, min_p, repeat_penalty, ...).
- **Prompt.** Built only from `canonical_context`: system message (role, rules, response format) + one `CONTEXT:` JSON object (identity, task, goals, observations, known gaps, team_plan, legal_actions, held_dik, communications_and_artifacts,
  recent_outcomes). Each legal action gets an `affordance_id` (`a0..`) and a curated set of its own fields (no utility scores). Hidden truth, other agents' DIK, the live team store, paths/logs and Python objects never appear.
- **Budget.** Deterministic tiers: tier 1 (identity, task, goals, observations, gaps, plan assignment, legal actions) is always sent (only optional detail may shrink, in a fixed order, and the affordance list never shrinks);
  tier 2 (held DIK, artifacts/communications, blockers and recent outcomes) and tier 3 (older shared updates, construction updates, older events) fill the remaining budget item by item, most relevant/recent first.
  Token estimate: starts at `chars/2.8` and adapts per model tag from real `prompt_eval_count` (densest recent ratio minus a 7% margin, clamped to [2.0, 4.0]; `modules/token_estimator.py`); the configured budget is never raised, and estimated vs actual counts plus the estimator state are recorded per decision and in the manifest. Omitted counts per section, shrink steps and `tier1_over_budget` are stored in the decision trace.
- **Not-currently-executable actions.** REPAIR (`repair_ready`), VALIDATE (`validation_ready`, the ActionGate's own readiness) and INSPECT of another role's private source (`source_accessible`, `private_source_inaccessible`) are listed under `blocked_actions` with blockers and no affordance id; the ActionGate still refuses a forced attempt. No substitute source is inspected for ANY Pilot (access is controller-neutral).
- **Resources.** Local inference is guarded by `modules/resource_guard.py` in both the GUI (governor) and headless (synchronous, waits before each call) paths: RAM floor, optional NVIDIA headroom/contention/utilisation checks; unsafe = wait, never advance simulated time, never kill/unload anything, never fall back to another controller. Failures carry a `failure_class` (transport timeout vs resource-contention timeout vs illegal action vs malformed output).
- **Response.** `{"affordance_id": "<id>", "arguments": {}, "decision_note": "<=200 chars, optional>"}` enforced with an Ollama JSON-schema `format` (affordance_id is an enum of the offered ids) and re-validated. `arguments` may only repeat
  the chosen affordance's own `action_args`. The note is not executable. The choice becomes an ordinary `AgentBrainResponse` and goes through the same grounding, ActionGate and translation as every Pilot.
- **Failure.** Unparsable/unknown/illegal replies and inference errors get a concise correction listing the legal ids and up to `max_retries` retries. After that the controller raises `ControllerFailure`: the failure is traced and, with
  `stop_run` (default), the run halts with `terminal_state = controller_error`; with `idle_and_log` that agent idles one tick, explicitly logged. The Procedural Baseline is never consulted for an Ollama agent.
- **Mecha behaviour.** Ollama agents are decided in the synchronous Pilot path (no async planner, DIK-integration, bootstrap fallback, policy pivots or project rebinding). Translation-time `action_legality_failed` outcomes are fed back to a generative Pilot
  as blocker names in `recent_action_outcomes` (unchanged for the Procedural Baseline). Inference wall-clock never advances simulated time.
- **Records.** Manifest: `controller_assignments.per_agent[agent].ollama` (model, resolved tag, service version, installed models at start, all settings, implementation version). Events: `controller_decision`, `controller_failure`,
  `controller_error_run_halted`. Durable trace: `logs/controller_decisions.jsonl` (exact system/user prompt, hash, options, budget metadata, every attempt's raw reply/error/latency/Ollama token metadata, parsed affordance/action,
  validation outcome, retry count, wall-clock latency, plus a `translation_outcome` row per decision).

## Canonical controller context (`modules/controller_context.py`, schema `controller_context/1`)
Every request carries `task_context["canonical_context"]`, a whitelist projection of legitimate agent state: identity, task, goals, observations, held DIK, known gaps,
communications/artifacts, `team_plan_context`, recent outcomes/blockers and the legal action candidates. Hidden world truth (e.g. construction correctness), other agents' private DIK,
the Environment/SimulationState, logs/filesystem and the live team-knowledge store are never copied; `assert_canonical_context_clean` rejects forbidden keys at any depth.
The Procedural Baseline receives the same object (it keeps using the richer packet internally; behaviour is unchanged).

## Team-plan context (controller-neutral)
Every Pilot (Procedural Baseline, LLM-backed, future) receives the same legitimate team-plan facts from the middleware; none reads `TeamKnowledgeManager`.
The facts are built by `Agent.team_plan_context()` and exposed as `team_state["team_plan_context"]` (schema `team_plan_context/1`), mirrored in the
provider request as `task_context["team_plan_context"]`:
- `known_plan`: the plan **as this agent read, received or authored it** (`status_when_read`, `read_at`, `read_via`, `age_s`, targets, assignments, `review_at`, `expires_at`, `expired`).
  Never the live store: a plan the agent has not acquired appears only as `unread_live_plan_artifact: true` (existence of an unread version on the workspace, no content).
- `adopted`, `adopted_plan_id`.
- `my_assignment`: the agent's own assignment, `status` (`applied | started | ungrounded | completed | blocked | invalidated | expired | superseded`), `grounded`, `grounding_reason`,
  the assigned `action_types`/`target`, the task-level `success_criterion` (`kind`, `status: open|met|indeterminate`, `detail`), `progress` (`aligned_actions`, `deviations`, `started`),
  `recognized_gap_tokens` (what is missing when it cannot be grounded; nothing is granted) and `matching_affordances`.
- `review`: `next_review_at`, `review_due`, `expires_at`, `expired`, `revision_due`.
Matching legal affordances additionally carry `plan_assignment_plan_id` and `plan_assignment` (`plan_id`, `criterion_kind`, `aligned_actions`, `assignment_status`). The marker is a stated fact,
not a command: a Pilot may follow, defer or deviate, and deviation is logged. Shared-store lifecycle chatter (`team_plan_*` updates, unread plan summaries/status) is removed from
`recent_shared_updates`, `team_shared_knowledge` and the artifact summaries. Agents also pass on (message) only the plan copy they hold.

## Ollama provider integration
`OllamaLocalBrainProvider` is a pluggable provider in `modules/brain_provider.py` using local HTTP endpoint configuration and fallback-safe behavior. Provider errors, malformed output, or invalid plans safely degrade to deterministic legal fallback.
