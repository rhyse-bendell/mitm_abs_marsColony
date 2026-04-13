# Experimental Mechanism Audit (construct → mechanism → hook)

Scope: enabled mechanisms from `config/experimental/mechanism_to_hook.csv` (enabled=true), traced through resolver and runtime behavior entry points.

## Resolution path (shared by all rows)

1. `SimulationState` resolves per-agent profiles via `ConstructMapper.resolve_agent_profile(construct_values, mechanism_overrides, mechanism_defaults)`.  
2. Resolution precedence is: mechanism defaults → construct-derived effects → normalized overrides (`traits` alias merged first, then explicit `mechanism_overrides` overwrite key-by-key).  
3. Runtime receives canonical `agent.construct_values`, `agent.mechanism_overrides`, `agent.mechanism_profile`, and `agent.hook_effects`.

## Enabled mechanism table

| mechanism_id | source construct(s) | default / override path | hook target(s) | behavior entry point(s) | affects_action_choice | affects_state_change | affects_outcome_quality | affects_timing | current_experimental_strength | issues / risks | changes made in this PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| communication_propensity | teamwork_potential | defaults + construct + override | `action_utility:communicate`, `action_utility:request_assistance` | `Agent._apply_trait_bias_to_decision` | yes (communication/externalization choice) | moderate (via communication side-effects) | moderate | low | moderate | `meeting` hook target remains unresolved in runtime | none; documented unresolved `meeting` |
| goal_alignment | teamwork_potential | defaults + construct + override | `action_utility:consult_team_artifact`, `externalize_plan`, `reassess_plan` | `_apply_trait_bias_to_decision`, `_evaluate_team_plan_fit`, `_adopt_committed_team_plan` | yes (consult/reassess/team-plan responses) | yes (team-plan adoption + goals) | yes | low | strong | previously team-plan deference did not strongly condition committed assignment acceptance | committed team-plan uptake redirection + committed-plan acceptance path + alignment-scaled commitment windows |
| help_tendency | teamwork_potential | defaults + construct + override | `request_assistance`, `repair_or_correct_construction`, `decision_bias:assist_stalled_teammate` | `_apply_trait_bias_to_decision`, `_update_goal_states_from_runtime`, `compare_and_repair_construction` | yes | yes (support goals, repair) | yes | low | strong | assist hook was mostly superficial before | stalled-teammate detection now sets assist support-goal priority and intent commitment windows; repair probability includes assist/repair hooks |
| build_speed | taskwork_potential | defaults + construct + override | action durations (`transport/start/continue/repair/validate`) | `_duration_scale`, `_translate_brain_decision_to_legacy_action` | no | no | moderate (faster completion) | yes | strong | none | none |
| rule_accuracy | taskwork_potential | defaults + construct + override | `dik_update:*`, `construction_fidelity:start_construction`, `validation_check:detect_mismatch` | `_epistemic_success_probability`, externalization fidelity in `_apply_externalization_and_construction_effects`, `compare_and_repair_construction` | indirect | yes | yes | low | strong | none | none |
| artifact_externalization_tendency | teamwork_potential | defaults + construct + override | `action_utility:externalize_plan`, `start_construction:externalization_weight` | `_apply_trait_bias_to_decision`, `_apply_externalization_and_construction_effects` | yes | yes (team-plan/artifact created/updated) | moderate | low | moderate→strong | `externalization_weight` was not materially used | externalization signal now combines mechanism + hooks and influences team-plan artifact lifespan |
| artifact_consultation_tendency | teamwork_potential | defaults + construct + override | `action_utility:consult_team_artifact` | `_apply_trait_bias_to_decision`, `_update_goal_states_from_runtime`, consult path in `_apply_externalization_and_construction_effects` | yes | yes (artifact uptake/adoption events) | yes | low | moderate→strong | previously mostly a choice nudge | support-goal priority now includes goal-alignment+consult signals; committed team-plan uptake bias added |
| artifact_adoption_tendency | (default unless future constructs enabled) | defaults + override | `artifact_use:adopt_externalized_knowledge` | consult branch in `_apply_externalization_and_construction_effects` | no | yes (artifact uptake count, consulted_by) | moderate | low | moderate | construct mapping currently disabled by default (`trust_bias` row disabled) | adoption probability now combines adoption tendency + consult tendency + adoption hook + goal alignment; committed team-plan bonus |
| teammate_model_accuracy | teamwork_potential | defaults + construct + override | `tom_update:update_teammate_model` | `_tom_update_success_probability`, perception/ToM updates | indirect | yes (theory_of_mind freshness) | moderate | low | moderate | depends on teammate observability opportunities | none |
| build_readiness_sensitivity | taskwork_potential | defaults + construct + override | `decision_threshold:start_construction` | `_readiness_threshold` gates construction readiness | yes (build start eligibility) | yes | yes | low | strong | none | none |
| mismatch_detection_sensitivity | taskwork_potential | defaults + construct + override | `validation_check:detect_mismatch:sensitivity` | `compare_and_repair_construction` | yes (trigger correction path) | yes | yes | low | strong | none | none |
| validation_thoroughness | taskwork_potential | defaults + construct + override | `action_utility:validate_construction` | `_apply_trait_bias_to_decision` | yes | yes (validation actions) | yes | low | moderate | still mostly trigger-gated preference | none (kept bounded within current architecture) |
| plan_persistence | (default unless future constructs enabled) | defaults + override | `plan_control:continue_current_plan:persistence_weight` | `_plan_trigger_reason` (plan expiry) | yes (defer replanning) | no | moderate | yes | strong | none | none |
| replanning_tendency | (default unless future constructs enabled) | defaults + override | `plan_control:reassess_plan:utility_weight` | `_apply_trait_bias_to_decision` | yes | no | moderate | low | thin→moderate | previously resolved but not consumed | now actively used to trigger `REASSESS_PLAN` under contradiction/stall/readiness uncertainty |

## Observability and inspectability updates

- Per-agent manifest rows now include canonical `mechanism_overrides` and serialized `hook_effects` values (not only hook keys).  
- Runtime logs now emit `legacy_traits_alias_normalized` whenever old `traits` payloads are provided and normalized into canonical `mechanism_overrides`.  
- Runtime hook logs include concrete hook key→value maps for post hoc tracing.  
- Brain context now exposes canonical `individual_cognitive_state.mechanism_profile` (with `traits` retained as legacy alias for compatibility).

## Honest limitations

- `action_utility:meeting` has no direct runtime consumer in current action space; this remains a known thin/unwired hook target.  
- Some mechanisms (e.g., validation_thoroughness) remain preference-level rather than direct world-mutation hooks by design; strengthening further would require broader controller policy changes.
