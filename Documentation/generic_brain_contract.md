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

## Ollama provider integration
`OllamaLocalBrainProvider` is a pluggable provider in `modules/brain_provider.py` using local HTTP endpoint configuration and fallback-safe behavior. Provider errors, malformed output, or invalid plans safely degrade to deterministic legal fallback.
