import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construct_mapping import ConstructMapper
from modules.simulation import SimulationState


class TestConstructMapping(unittest.TestCase):
    def test_default_config_loads(self):
        mapper = ConstructMapper(config_dir="config")
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

            project = sim.environment.construction.projects["Build_Table_B"]
            agent.active_actions = [{"type": "construct", "progress": 0, "duration": 1.0, "priority": 1}]
            agent.hook_effects[("construction_fidelity", "start_construction", "fidelity_score")] = 0.1
            import modules.agent as agent_module
            original_random = agent_module.random.random
            try:
                agent_module.random.random = lambda: 0.8
                agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            finally:
                agent_module.random.random = original_random
            self.assertFalse(project["correct"])

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
                        "traits": {
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
            sim.update(0.2)
            self.assertGreater(sim.time, 0.0)


if __name__ == "__main__":
    unittest.main()
