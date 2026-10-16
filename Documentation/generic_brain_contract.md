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
