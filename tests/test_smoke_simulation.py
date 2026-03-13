import unittest
import random

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState


class TestSimulationSmoke(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_non_gui_simulation_step_runs(self):
        sim = SimulationState(phases=[])
        sim.update(0.1)
        self.assertGreater(sim.time, 0.0)
        self.assertEqual(len(sim.agents), 3)

    def test_packet_access_respected(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.objects["Architect_Info"]["position"])

        agent.allowed_packet = ["Team_Packet"]
        agent.update_knowledge(env)
        info_ids = {info.id for info in agent.mental_model["information"]}
        self.assertNotIn("I004", info_ids)

        agent.allowed_packet = ["Team_Packet", "Architect_Packet"]
        agent.update_knowledge(env)
        info_ids = {info.id for info in agent.mental_model["information"]}
        self.assertIn("I004", info_ids)


if __name__ == "__main__":
    unittest.main()


class TestUnifiedGoalPipeline(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_update_uses_authoritative_goal_pipeline(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        env.agents = [agent]

        calls = []

        def fake_pipeline(dt, environment):
            calls.append((dt, environment))

        agent._run_goal_management_pipeline = fake_pipeline

        def fail_decide_next_action(_environment):
            raise AssertionError("legacy decide_next_action path should not run in update")

        def fail_update_active_actions(_dt):
            raise AssertionError("legacy update_active_actions wrapper should not run in update")

        agent.decide_next_action = fail_decide_next_action
        agent.update_active_actions = fail_update_active_actions

        agent.update(0.25, env)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0.25)
        self.assertIs(calls[0][1], env)

    def test_authoritative_pipeline_advances_goal_seek_share_build(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.objects["Team_Info"]["position"])
        agent.allowed_packet = ["Team_Packet", "Architect_Packet"]

        # Step 1: no goal yet -> seek_info is pushed.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "seek_info")

        # Make I004 available in mental model to trigger share transition.
        architect_info = [i for i in env.knowledge_packets["Architect_Info"]["information"] if i.id == "I004"][0]
        agent.mental_model["information"].add(architect_info)

        # Step 2: seek_info -> share.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "share")

        # Step 3: share -> build with construction target.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "build")
        self.assertEqual(agent.current_goal()["target"], env.objects["Table_B"]["position"])
