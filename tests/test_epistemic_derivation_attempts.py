import unittest
from unittest.mock import patch

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class EpistemicDerivationAttemptTests(unittest.TestCase):
    def _prime_phase_objectives_inputs(self, agent, env):
        team_packet = env.knowledge_packets["Team_Info"]
        needed = {
            "D_SHARED_PLANNING_REVIEW_ORDER",
            "D_SHARED_PHASE1_TARGET_50_CIV",
            "D_SHARED_PHASE2_TARGET_40_CIV_20_VIP",
        }
        for item in team_packet["data"]:
            if item.id in needed:
                agent.mental_model["data"].add(item)

    def test_data_to_information_derivation_uses_mechanism_probability(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Engineer", role="Engineer")
        agent.task_model = model
        self._prime_phase_objectives_inputs(agent, env)

        agent.rule_accuracy = 0.95
        agent.hook_effects[("dik_update", "transform_data_to_information", "success_probability")] = 0.95

        with patch("modules.agent.random.uniform", return_value=0.0), patch("modules.agent.random.random", return_value=0.2):
            agent._apply_task_derivations(sim_state=None)

        info_ids = {i.id for i in agent.mental_model["information"]}
        self.assertIn("I_SHARED_PHASE_OBJECTIVES", info_ids)

    def test_eligibility_does_not_guarantee_output_when_probability_is_low(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Engineer", role="Engineer")
        agent.task_model = model
        self._prime_phase_objectives_inputs(agent, env)

        agent.rule_accuracy = 0.05
        agent.hook_effects[("dik_update", "transform_data_to_information", "success_probability")] = 0.05

        with patch("modules.agent.random.uniform", return_value=0.0), patch("modules.agent.random.random", return_value=0.95):
            agent._apply_task_derivations(sim_state=None)

        info_ids = {i.id for i in agent.mental_model["information"]}
        self.assertNotIn("I_SHARED_PHASE_OBJECTIVES", info_ids)
        self.assertNotIn("DRV_PHASE_OBJECTIVES", agent.executed_derivations)

    def test_failed_derivation_is_retryable(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Engineer", role="Engineer")
        agent.task_model = model
        self._prime_phase_objectives_inputs(agent, env)

        agent.rule_accuracy = 0.6
        agent.hook_effects[("dik_update", "transform_data_to_information", "success_probability")] = 0.6

        with patch("modules.agent.random.uniform", return_value=0.0), patch(
            "modules.agent.random.random", side_effect=[0.95, 0.2]
        ):
            agent._apply_task_derivations(sim_state=None)
            agent._apply_task_derivations(sim_state=None)

        info_ids = {i.id for i in agent.mental_model["information"]}
        self.assertIn("I_SHARED_PHASE_OBJECTIVES", info_ids)
        self.assertIn("DRV_PHASE_OBJECTIVES", agent.executed_derivations)

    def test_packet_absorption_attempt_can_fail_with_weak_mechanism(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        team_packet = sim.environment.knowledge_packets["Team_Info"]

        agent.rule_accuracy = 0.05
        agent.hook_effects[("dik_update", "absorb_packet", "success_probability")] = 0.05

        with patch("modules.agent.random.uniform", return_value=0.0), patch("modules.agent.random.random", return_value=0.99):
            agent.absorb_packet(team_packet, sim_state=sim, source_id="Team_Info")

        self.assertEqual(len(agent.mental_model["data"]), 0)
        self.assertEqual(len(agent.mental_model["information"]), 0)
        event_types = [e["event_type"] for e in sim.logger.recent_events]
        self.assertIn("packet_absorption_attempted", event_types)
        self.assertIn("packet_absorption_failed", event_types)
        sim.stop()


if __name__ == "__main__":
    unittest.main()
