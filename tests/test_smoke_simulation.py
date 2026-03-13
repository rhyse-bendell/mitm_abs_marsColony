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
