import unittest
import random
import json
import re
import tempfile
from pathlib import Path

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState


class TestSimulationSmoke(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_non_gui_simulation_step_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            sim.update(0.1)
            self.assertGreater(sim.time, 0.0)
            self.assertEqual(len(sim.agents), 3)

    def test_session_output_structure_and_manifest_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                experiment_name="My Test Experiment",
                speed=1.5,
                flash_mode=True,
                project_root=tmpdir,
            )

            sim.update(0.1)
            sim.stop()

            outputs_root = Path(tmpdir) / "Outputs"
            sessions = [path for path in outputs_root.iterdir() if path.is_dir()]
            self.assertEqual(len(sessions), 1)

            session_dir = sessions[0]
            self.assertRegex(session_dir.name, r"^My_Test_Experiment_\d{8}_\d{6}$")

            logs_dir = session_dir / "logs"
            self.assertTrue(logs_dir.exists())
            self.assertTrue(any(path.suffix == ".csv" for path in logs_dir.iterdir()))

            manifest_path = session_dir / "session_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment_name"], "My Test Experiment")
            self.assertEqual(manifest["sanitized_prefix"], "My_Test_Experiment")
            self.assertTrue(re.match(r"^\d{8}_\d{6}$", manifest["timestamp"]))

            measures_placeholder = session_dir / "measures" / "final_measures_placeholder.json"
            self.assertTrue(measures_placeholder.exists())

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


class TestMovementAndInfoAccessRepairs(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_role_spawns_are_not_inside_blocking_geometry(self):
        env = Environment(phases=[])

        for role in ["Architect", "Engineer", "Botanist"]:
            pos = env.get_spawn_point(role)
            for name, obj in env.objects.items():
                if obj.get("type") in {"rect", "circle", "blocked"} and not obj.get("passable", False):
                    self.assertFalse(
                        env.is_near_object(pos, name, threshold=0.0),
                        msg=f"{role} spawn {pos} overlaps blocking geometry {name}",
                    )

    def test_info_packets_are_non_blocking_interaction_targets(self):
        env = Environment(phases=[])
        for packet_name in ["Team_Info", "Architect_Info", "Engineer_Info", "Botanist_Info"]:
            self.assertTrue(env.objects[packet_name].get("passable", False), msg=f"{packet_name} should be passable")

    def test_rectangular_info_access_uses_zone_proximity(self):
        env = Environment(phases=[])

        architect_station = env.objects["Architect_Info"]
        inside_point = (architect_station["position"][0] + 0.2, architect_station["position"][1] + 0.2)
        self.assertTrue(env.can_access_info(inside_point, "Architect_Info", role="Architect"))

        far_from_rect = (architect_station["position"][0] + 1.0, architect_station["position"][1] + 1.0)
        self.assertFalse(env.can_access_info(far_from_rect, "Architect_Info", role="Architect"))

    def test_roles_leave_spawn_and_are_not_immediately_stuck(self):
        sim = SimulationState(phases=[])
        starts = {a.role: a.position for a in sim.agents}

        for _ in range(4):
            sim.update(1.0)

        for agent in sim.agents:
            moved = ((agent.position[0] - starts[agent.role][0]) ** 2 + (agent.position[1] - starts[agent.role][1]) ** 2) ** 0.5
            self.assertGreater(moved, 0.2, msg=f"{agent.role} did not leave spawn")

            recent_log = " ".join(agent.activity_log[-8:])
            self.assertNotIn("Blocked while moving", recent_log, msg=f"{agent.role} became immediately stuck")

    def test_agent_can_reach_and_ingest_allowed_packet(self):
        env = Environment(phases=[])
        station = env.objects["Architect_Info"]
        center = (station["position"][0] + (station["size"][0] / 2.0), station["position"][1] + (station["size"][1] / 2.0))
        agent = Agent(name="Architect", role="Architect", position=center)
        agent.allowed_packet = ["Architect_Packet"]

        agent.update_knowledge(env)

        info_ids = {info.id for info in agent.mental_model["information"]}
        self.assertIn("I004", info_ids)
