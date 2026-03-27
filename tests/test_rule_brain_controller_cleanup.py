import unittest

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState


class TestRuleBrainControllerCleanup(unittest.TestCase):
    def test_rule_brain_runtime_uses_local_controller_not_async_submit(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            agent = sim.agents[0]
            calls = {"controller": 0}

            def _fail_submit(*_args, **_kwargs):
                raise AssertionError("async planner submit should not run for rule_brain runtime")

            def _count_controller(*_args, **_kwargs):
                calls["controller"] += 1
                return True

            agent._submit_planner_request_async = _fail_submit
            agent._run_rule_brain_controller = _count_controller
            sim.update(0.2)
            self.assertGreaterEqual(calls["controller"], 1)
        finally:
            sim.stop()

    def test_stale_inspect_latch_does_not_execute_without_inspect_step(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            agent = sim.agents[0]
            agent.current_inspect_target_id = "Team_Info"
            agent.active_method_step = "move_to_role_source"
            seen = {"inspect": 0}

            def _inspect_stub(*_args, **_kwargs):
                seen["inspect"] += 1
                return True

            agent._inspect_source = _inspect_stub
            agent._run_rule_brain_controller = lambda *_args, **_kwargs: False
            agent.update(0.1, sim.environment, sim_state=sim, planner_lifecycle_already_polled=True)
            self.assertEqual(seen["inspect"], 0)
            self.assertIsNone(agent.current_inspect_target_id)
        finally:
            sim.stop()

    def test_team_info_exhaustion_handoffs_to_role_grounding(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            agent = sim.agents[1]  # Engineer
            agent.source_exhaustion_state["Team_Info"] = {
                "inspect_count": 3,
                "last_dik_changed": False,
                "no_new_dik_streak": 3,
                "inspected": True,
                "exhausted": True,
            }
            agent.known_gaps.add("Engineer specific gap")
            ok = agent._run_rule_brain_controller(sim, sim.environment, "unit_test")
            self.assertTrue(ok)
            self.assertEqual(agent.active_method_id, "AcquireRoleSpecificGrounding")
            actions = list(agent.current_action or [])
            self.assertTrue(actions)
            inspect_targets = {a.get("source_target_id") for a in actions if a.get("source_target_id")}
            if inspect_targets:
                self.assertIn("Engineer_Info", inspect_targets)
        finally:
            sim.stop()

    def test_no_autonomous_proximity_communication_in_simulation_update(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            a0, a1 = sim.agents[0], sim.agents[1]
            a0.position = (5.0, 5.0)
            a1.position = (5.1, 5.0)
            count = {"calls": 0}

            def _count_comm(*_args, **_kwargs):
                count["calls"] += 1

            a0.communicate_with = _count_comm
            a1.communicate_with = _count_comm
            sim.update(0.2)
            self.assertEqual(count["calls"], 0)
        finally:
            sim.stop()

    def test_derivation_pipeline_not_called_by_background_update_knowledge(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.objects["Team_Info"]["position"])
        calls = {"derivations": 0, "secondary": 0}

        def _count_derivations(*_args, **_kwargs):
            calls["derivations"] += 1

        def _count_secondary(*_args, **_kwargs):
            calls["secondary"] += 1

        agent._apply_task_derivations = _count_derivations
        agent._apply_secondary_rule_inference = _count_secondary
        agent.update_knowledge(env, full_packet_sweep=False, sim_state=None)
        self.assertEqual(calls["derivations"], 0)
        self.assertEqual(calls["secondary"], 0)

        packet = env.knowledge_packets["Team_Info"]
        agent.absorb_packet(packet, accuracy=1.0, sim_state=None, source_id="Team_Info")
        self.assertGreater(calls["derivations"], 0)
        self.assertGreater(calls["secondary"], 0)

    def test_update_knowledge_no_longer_runs_background_tag_rule_inference(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        packet = env.knowledge_packets["Architect_Info"]
        for info in packet.get("information", []):
            agent.mental_model["information"].add(info)

        calls = {"count": 0}
        original_try_infer = agent.mental_model["knowledge"].try_infer_rules

        def _count_try_infer(*args, **kwargs):
            calls["count"] += 1
            return original_try_infer(*args, **kwargs)

        agent.mental_model["knowledge"].try_infer_rules = _count_try_infer
        agent.update_knowledge(env, full_packet_sweep=False, sim_state=None)
        self.assertEqual(calls["count"], 0)

    def test_communication_transfer_still_triggers_epistemic_pipeline(self):
        env = Environment(phases=[])
        sender = Agent(name="Architect", role="Architect", position=(5.0, 5.0))
        receiver = Agent(name="Engineer", role="Engineer", position=(5.05, 5.0))
        env.agents = [sender, receiver]
        packet = env.knowledge_packets["Team_Info"]
        sender.absorb_packet(packet, accuracy=1.0, sim_state=None, source_id="Team_Info")

        seen = {"count": 0}

        def _count_trigger(*_args, **_kwargs):
            seen["count"] += 1

        receiver._trigger_epistemic_update_pipeline = _count_trigger
        sender.communicate_with(receiver, sim_state=None)
        self.assertGreater(seen["count"], 0)

    def test_legacy_wrappers_remain_callable_as_compatibility_shims(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        env.agents = [agent]

        agent.evaluate_goals()
        actions = agent.select_action()
        self.assertIsInstance(actions, list)

        agent.decide_next_action(env)
        self.assertIsInstance(agent.current_action, list)

        agent.decide(type("LegacySim", (), {"environment": env, "agents": [agent], "time": 0.0})())
        self.assertIsInstance(agent.current_action, list)

    def test_deprecated_wrappers_not_used_in_live_simulation_update(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            agent = sim.agents[0]

            def _fail(*_args, **_kwargs):
                raise AssertionError("deprecated wrapper should not drive live runtime")

            agent.decide = _fail
            agent.evaluate_goals = _fail
            agent.select_action = _fail
            agent.update_active_actions = _fail
            agent.decide_next_action = _fail
            agent._run_goal_management_pipeline = _fail
            sim.update(0.2)
        finally:
            sim.stop()

    def test_rule_brain_runtime_does_not_increment_llm_counters(self):
        sim = SimulationState(phases=[], brain_backend="rule_brain")
        try:
            for _ in range(3):
                sim.update(0.2)
            for agent in sim.agents:
                planner = agent.planner_state
                self.assertEqual(planner.get("llm_success_count"), 0)
                self.assertEqual(planner.get("llm_timeout_count"), 0)
                self.assertEqual(planner.get("llm_invalid_count"), 0)
                self.assertEqual(planner.get("llm_transport_error_count"), 0)
                self.assertGreater(planner.get("deterministic_rulebrain_decision_count", 0), 0)
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
