import json
import random
import tempfile
import unittest
from pathlib import Path

from modules.agent import Agent
from modules.brain_context import BrainContextBuilder
from modules.environment import Environment
from modules.simulation import SimulationState


class TestPhaseAwareProgression(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def _utility(self, packet, action_type):
        matches = [a for a in packet.action_affordances if a["action_type"] == action_type]
        self.assertTrue(matches, msg=f"missing affordance for {action_type}")
        return max(float(a.get("utility", 0.0)) for a in matches)

    def test_phase_transitions_change_affordance_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            builder = BrainContextBuilder()

            early_packet = builder.build(sim, agent)
            early_profile = early_packet.world_snapshot["phase_profile"]

            agent.mental_model["information"].update(sim.environment.knowledge_packets["Team_Info"]["information"][:2])
            agent.mental_model["knowledge"].rules.append("rule:house_enclosed")
            agent.source_inspection_state["Team_Info"] = "inspected"

            for _ in range(60):
                sim.update(1.0)

            late_packet = builder.build(sim, agent)
            late_profile = late_packet.world_snapshot["phase_profile"]

            self.assertNotEqual(early_profile["stage"], late_profile["stage"])
            early_inspect = self._utility(early_packet, "inspect_information_source")
            late_start = self._utility(late_packet, "start_construction")
            self.assertGreater(early_inspect, 0.5)
            self.assertGreater(late_start, 0.5)

    def test_build_is_not_eligible_before_minimum_readiness(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        builder = BrainContextBuilder()

        readiness = builder._build_readiness(agent, builder._summarize_structures(env), env, team_state={"externalized_artifacts": []})
        self.assertEqual(readiness["status"], "premature")
        self.assertIn("insufficient_information_inspection", readiness["blockers"])

    def test_planning_and_externalization_more_plausible_before_execution_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)

            inspect_u = self._utility(packet, "inspect_information_source")
            start_u = self._utility(packet, "start_construction")
            externalize_u = self._utility(packet, "externalize_plan")

            self.assertGreater(inspect_u, start_u)
            self.assertGreater(externalize_u, start_u)

    def test_execution_and_logistics_more_plausible_after_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            env = sim.environment

            agent.mental_model["information"].update(env.knowledge_packets["Team_Info"]["information"][:2])
            agent.mental_model["knowledge"].rules.append("rule:house_enclosed")
            agent.source_inspection_state["Team_Info"] = "inspected"

            for _ in range(70):
                sim.update(1.0)

            packet = BrainContextBuilder().build(sim, agent)
            start_u = self._utility(packet, "start_construction")
            transport_u = self._utility(packet, "transport_resources")
            communicate_u = self._utility(packet, "communicate")

            self.assertGreater(start_u, communicate_u)
            self.assertGreater(transport_u, communicate_u)

    def test_headless_run_produces_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            for _ in range(8):
                sim.update(0.5)
            sim.stop()

            outputs_root = Path(tmpdir) / "Outputs"
            session = next(path for path in outputs_root.iterdir() if path.is_dir())
            run_summary = session / "measures" / "run_summary.json"
            phase_summary = session / "measures" / "phase_summary.json"
            self.assertTrue(run_summary.exists())
            self.assertTrue(phase_summary.exists())

            parsed = json.loads(run_summary.read_text(encoding="utf-8"))
            self.assertIn("process", parsed)
            self.assertIn("events", parsed)


if __name__ == "__main__":
    unittest.main()
