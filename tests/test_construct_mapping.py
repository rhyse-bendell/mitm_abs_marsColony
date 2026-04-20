import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construct_mapping import ConstructMapper
from modules.experimental_config import normalize_mechanism_override_inputs
from modules.simulation import SimulationState


class TestConstructMapping(unittest.TestCase):
    def test_default_config_loads(self):
        mapper = ConstructMapper()
        self.assertIn("teamwork_potential", mapper.constructs)
        self.assertIn("taskwork_potential", mapper.constructs)
        self.assertGreater(len(mapper.construct_to_mechanism), 0)
        self.assertGreater(len(mapper.mechanism_to_hook), 0)

    def test_invalid_transform_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "constructs.csv").write_text(
                "construct_id,label,description,scale_min,scale_max,default_value,construct_group,enabled,notes,citation\n"
                "teamwork_potential,Teamwork Potential,d,0,1,0.5,baseline,true,n,c\n",
                encoding="utf-8",
            )
            (cfg / "construct_to_mechanism.csv").write_text(
                "construct_id,mechanism_id,effect_weight,transform,intercept,min_output,max_output,phase_scope,condition_group,enabled,notes\n"
                "teamwork_potential,communication_propensity,0.5,not_real,0,0,1,all,default,true,n\n",
                encoding="utf-8",
            )
            (cfg / "mechanism_to_hook.csv").write_text(
                "mechanism_id,hook_type,hook_target,operator,parameter,formula_name,min_effect,max_effect,enabled,notes\n"
                "communication_propensity,action_utility,communicate,add,utility_weight,bounded_add,0,1,true,n\n",
                encoding="utf-8",
            )
            mapper = ConstructMapper(config_dir=cfg)
            self.assertTrue(any("Unknown transform" in issue for issue in mapper.validation_issues))
            self.assertEqual(mapper.construct_to_mechanism, [])

    def test_teamwork_and_taskwork_resolve_expected_directions(self):
        mapper = ConstructMapper(config_dir="config")
        low = mapper.resolve_mechanisms({"teamwork_potential": 0.0, "taskwork_potential": 0.0}, mechanism_overrides={})
        high = mapper.resolve_mechanisms({"teamwork_potential": 1.0, "taskwork_potential": 1.0}, mechanism_overrides={})

        self.assertGreater(high["communication_propensity"], low["communication_propensity"])
        self.assertGreater(high["help_tendency"], low["help_tendency"])
        self.assertGreater(high["build_speed"], low["build_speed"])
        self.assertGreater(high["rule_accuracy"], low["rule_accuracy"])

    def test_disabled_rows_do_not_apply(self):
        mapper = ConstructMapper(config_dir="config")
        mechanisms = mapper.resolve_mechanisms({"conscientiousness": 1.0}, mechanism_overrides={})
        self.assertNotIn("plan_persistence", mechanisms)

    def test_precedence_override_then_construct_then_default(self):
        mapper = ConstructMapper()
        mechanisms = mapper.resolve_mechanisms(
            {"teamwork_potential": 0.75},
            mechanism_overrides={"communication_propensity": 0.1},
            mechanism_defaults={"communication_propensity": 0.9, "goal_alignment": 0.2},
        )
        self.assertAlmostEqual(mechanisms["communication_propensity"], 0.1, places=4)
        self.assertGreater(mechanisms["goal_alignment"], 0.2)

    def test_simulation_traits_alias_normalizes_to_mechanism_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                agent_configs=[
                    {
                        "name": "Architect",
                        "role": "Architect",
                        "constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5},
                        "traits": {"help_tendency": 0.91},
                    }
                ],
            )
            agent = sim.agents[0]
            self.assertAlmostEqual(agent.mechanism_overrides.get("help_tendency", 0.0), 0.91, places=4)
            self.assertAlmostEqual(agent.mechanism_profile.get("help_tendency", 0.0), 0.91, places=4)

    def test_traits_alias_merges_with_explicit_mechanism_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                agent_configs=[
                    {
                        "name": "Architect",
                        "role": "Architect",
                        "traits": {"help_tendency": 0.2, "goal_alignment": 0.9},
                        "mechanism_overrides": {"help_tendency": 0.8},
                    }
                ],
            )
            agent = sim.agents[0]
            self.assertAlmostEqual(agent.mechanism_overrides["help_tendency"], 0.8, places=4)
            self.assertAlmostEqual(agent.mechanism_overrides["goal_alignment"], 0.9, places=4)
            self.assertAlmostEqual(agent.mechanism_profile["help_tendency"], 0.8, places=4)
            self.assertAlmostEqual(agent.mechanism_profile["goal_alignment"], 0.9, places=4)
            self.assertNotIn("traits", agent.__dict__)

    def test_construct_profile_applies_when_mechanism_overrides_are_neutral_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                agent_configs=[
                    {
                        "name": "Architect",
                        "role": "Architect",
                        "constructs": {"teamwork_potential": 0.75, "taskwork_potential": 0.75},
                        "mechanism_overrides": {
                            "communication_propensity": 0.5,
                            "goal_alignment": 0.5,
                            "help_tendency": 0.5,
                            "build_speed": 0.5,
                            "rule_accuracy": 0.5,
                        },
                    }
                ],
            )
            agent = sim.agents[0]
            self.assertEqual(agent.mechanism_overrides, {})
            self.assertGreater(agent.mechanism_profile["communication_propensity"], 0.5)
            self.assertGreater(agent.mechanism_profile["help_tendency"], 0.5)

    def test_explicit_mechanism_override_wins_over_construct_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                agent_configs=[
                    {
                        "name": "Architect",
                        "role": "Architect",
                        "constructs": {"teamwork_potential": 0.75, "taskwork_potential": 0.75},
                        "mechanism_overrides": {"communication_propensity": 0.17},
                    }
                ],
            )
            agent = sim.agents[0]
            self.assertAlmostEqual(agent.mechanism_profile["communication_propensity"], 0.17, places=4)

    def test_legacy_traits_alias_normalizes_to_single_override_path(self):
        normalized, legacy_alias, explicit_overrides = normalize_mechanism_override_inputs(
            {
                "traits": {"help_tendency": 0.25},
                "mechanism_overrides": {"help_tendency": 0.8, "goal_alignment": 0.5},
            },
            mechanism_defaults={"goal_alignment": 0.5},
        )
        self.assertEqual(legacy_alias, {"help_tendency": 0.25})
        self.assertEqual(explicit_overrides["help_tendency"], 0.8)
        self.assertEqual(normalized["help_tendency"], 0.8)
        self.assertNotIn("goal_alignment", normalized)

    def test_invalid_numeric_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "constructs.csv").write_text(
                "construct_id,label,description,scale_min,scale_max,default_value,construct_group,enabled,notes,citation\n"
                "teamwork_potential,Teamwork Potential,d,0,1,0.5,baseline,true,n,c\n",
                encoding="utf-8",
            )
            (cfg / "construct_to_mechanism.csv").write_text(
                "construct_id,mechanism_id,effect_weight,transform,intercept,min_output,max_output,phase_scope,condition_group,enabled,notes\n"
                "teamwork_potential,communication_propensity,not_a_number,linear,0,0,1,all,default,true,n\n",
                encoding="utf-8",
            )
            (cfg / "mechanism_to_hook.csv").write_text(
                "mechanism_id,hook_type,hook_target,operator,parameter,formula_name,min_effect,max_effect,enabled,notes\n"
                "communication_propensity,action_utility,communicate,add,utility_weight,bounded_add,0,1,true,n\n",
                encoding="utf-8",
            )
            mapper = ConstructMapper(config_dir=cfg)
            self.assertTrue(any("Invalid numeric field 'effect_weight'" in issue for issue in mapper.validation_issues))
            self.assertEqual(mapper.construct_to_mechanism, [])

    def test_hooks_affect_action_utility_bias_direction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            base = BrainDecision(
                selected_action=ExecutableActionType.WAIT,
                reason_summary="baseline",
                confidence=1.0,
            )
            context = SimpleNamespace(team_state={"plan_readiness": "validated_shared_plan"})
            # Trait gate open; hook drives probability via midpoint average in _apply_trait_bias_to_decision.
            agent.goal_alignment = 0.9
            agent.help_tendency = 0.0
            agent.communication_propensity = 0.0

            import modules.agent as agent_module
            original_random = agent_module.random.random
            try:
                agent.hook_effects[("action_utility", "consult_team_artifact", "utility_weight")] = 1.0
                agent_module.random.random = lambda: 0.95
                unchanged = agent._apply_trait_bias_to_decision(base, context, sim, "no_active_plan")
                self.assertEqual(unchanged.selected_action, ExecutableActionType.WAIT)

                decision2 = BrainDecision(
                    selected_action=ExecutableActionType.WAIT,
                    reason_summary="baseline",
                    confidence=1.0,
                )
                agent.hook_effects[("action_utility", "consult_team_artifact", "utility_weight")] = 1.0
                agent_module.random.random = lambda: 0.94
                changed = agent._apply_trait_bias_to_decision(decision2, context, sim, "no_active_plan")
                self.assertEqual(changed.selected_action, ExecutableActionType.CONSULT_TEAM_ARTIFACT)
            finally:
                agent_module.random.random = original_random

    def test_hooks_affect_duration_and_fidelity_directions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]

            decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES)
            agent.hook_effects[("action_duration", "transport_resources", "duration_scale")] = 0.6
            fast = agent._translate_brain_decision_to_legacy_action(decision, sim.environment)[0]["duration"]
            agent.hook_effects[("action_duration", "transport_resources", "duration_scale")] = 1.4
            slow = agent._translate_brain_decision_to_legacy_action(decision, sim.environment)[0]["duration"]
            self.assertLess(fast, slow)

            agent.hook_effects[("construction_fidelity", "start_construction", "fidelity_score")] = 0.1
            agent.rule_accuracy = 0.2
            low_fidelity = (
                agent._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5)
                + agent._trait_value("rule_accuracy")
            ) / 2.0
            agent.hook_effects[("construction_fidelity", "start_construction", "fidelity_score")] = 0.9
            agent.rule_accuracy = 0.9
            high_fidelity = (
                agent._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5)
                + agent._trait_value("rule_accuracy")
            ) / 2.0
            self.assertLess(low_fidelity, high_fidelity)

    def test_gui_compatibility_flow_and_headless_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                agent_configs=[
                    {
                        "name": "Architect",
                        "role": "Architect",
                        "constructs": {"teamwork_potential": 1.0, "taskwork_potential": 1.0},
                        "mechanism_overrides": {
                            "communication_propensity": 0.8,
                            "goal_alignment": 0.8,
                            "help_tendency": 0.8,
                            "build_speed": 0.9,
                            "rule_accuracy": 0.9,
                        },
                        "packet_access": ["Team_Packet", "Architect_Packet"],
                    }
                ],
            )
            self.assertIn("teamwork_potential", sim.agents[0].construct_values)
            self.assertIn("communication_propensity", sim.agents[0].mechanism_profile)
            self.assertIn("communication_propensity", sim.agents[0].mechanism_overrides)
            sim.update(0.2)
            self.assertGreater(sim.time, 0.0)

    def test_stalled_teammate_bias_creates_assist_support_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            teammate = sim.agents[1]
            teammate.loop_counters["action_repeats"] = 4
            agent.help_tendency = 0.95
            agent.hook_effects[("decision_bias", "assist_stalled_teammate", "priority_weight")] = 0.95
            agent._update_goal_states_from_runtime(sim, sim.environment)
            labels = [g.label for g in agent.goal_registry.values()]
            self.assertIn("assist_teammate", labels)
            self.assertGreater(agent.active_intent.get("min_commit_until", 0.0), sim.time)

    def test_replanning_tendency_hook_can_shift_decision_to_reassess_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            context = SimpleNamespace(team_state={"plan_readiness": "unvalidated"})
            decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="base", confidence=1.0)
            agent.hook_effects[("plan_control", "reassess_plan", "utility_weight")] = 0.95
            import modules.agent as agent_module
            original_random = agent_module.random.random
            try:
                agent_module.random.random = lambda: 0.1
                changed = agent._apply_trait_bias_to_decision(decision, context, sim, "contradiction_detected")
            finally:
                agent_module.random.random = original_random
            self.assertEqual(changed.selected_action, ExecutableActionType.REASSESS_PLAN)

    def test_committed_team_plan_fit_accepts_on_alignment_even_when_help_low(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            agent.goal_alignment = 0.95
            agent.help_tendency = 0.05
            agent.hook_effects[("action_utility", "consult_team_artifact", "utility_weight")] = 0.95
            response, reason = agent._evaluate_team_plan_fit(
                sim,
                {
                    "plan_id": "p1",
                    "status": "committed",
                    "assignments_by_role": {agent.role: {"task": "repair", "project_target": "greenhouse_build"}},
                },
            )
            self.assertEqual(response, "assignment_accept")
            self.assertEqual(reason, "committed_plan_alignment")


if __name__ == "__main__":
    unittest.main()
