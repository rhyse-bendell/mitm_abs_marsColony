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

    def test_authoritative_pipeline_defers_build_until_readiness_threshold(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.objects["Team_Info"]["position"])
        agent.allowed_packet = ["Team_Packet", "Architect_Packet"]

        # Step 1: no goal yet -> seek_info is pushed.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "seek_info")

        # A single information fragment is not enough to start build.
        architect_info = [i for i in env.knowledge_packets["Architect_Info"]["information"] if i.id == "I004"][0]
        agent.mental_model["information"].add(architect_info)

        # Step 2: seek_info -> share.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "share")

        # Step 3: share remains share until communication action is completed.
        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "share")

        # Complete communicate action; then share resolves to seek_info (not build) due to low readiness.
        for _ in range(12):
            agent._run_goal_management_pipeline(dt=0.2, environment=env)
            if agent.has_shared:
                break
        self.assertTrue(agent.has_shared)

        agent._run_goal_management_pipeline(dt=0.1, environment=env)
        self.assertEqual(agent.current_goal()["goal"], "seek_info")


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

    def test_team_info_rectangle_access_is_not_corner_only(self):
        env = Environment(phases=[])
        # Near top-right area of Team_Info rectangle; should still be valid interaction.
        point = (8.95, 6.85)
        self.assertTrue(env.can_access_info(point, "Team_Info", role="Engineer"))

    def test_viewport_bounds_tighten_to_playable_space(self):
        env = Environment(phases=[])
        (x_min, x_max), (y_min, y_max) = env.get_viewport_bounds()
        self.assertAlmostEqual(x_min, -0.2)
        self.assertAlmostEqual(x_max, 10.2)
        self.assertAlmostEqual(y_min, -0.2)
        self.assertAlmostEqual(y_max, 7.6)

    def test_team_info_zone_nearest_point_proximity_allows_walk_up_access(self):
        env = Environment(phases=[])
        # Just below team info zone by less than DEFAULT_INFO_ACCESS_RADIUS.
        point = (8.0, 6.1)
        self.assertTrue(env.can_access_info(point, "Team_Info", role="Engineer"))

    def test_table_interaction_is_possible_from_reasonable_adjacent_space(self):
        env = Environment(phases=[])
        point = (6.5, 2.7)
        self.assertTrue(env.can_interact_with_table(point, "Table_A"))

    def test_interaction_access_reason_reports_zone_vs_distance(self):
        env = Environment(phases=[])
        in_zone = env.get_interaction_access((8.0, 6.5), "Team_Info", role="Engineer")
        near_zone = env.get_interaction_access((8.0, 6.1), "Team_Info", role="Engineer")
        self.assertEqual(in_zone["reason"], "in_zone")
        self.assertEqual(near_zone["reason"], "near_zone")

    def test_blocked_regions_remain_impassable(self):
        env = Environment(phases=[])
        self.assertFalse(env.is_point_navigable((5.0, 3.0)))

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


    def test_startup_first_productive_action_reaches_translation(self):
        sim = SimulationState(phases=[])
        for _ in range(6):
            sim.update(1.0)
        events = sim.logger.get_recent_events(400)
        productive_started = [
            e for e in events
            if e["event_type"] == "first_productive_action_started"
        ]
        self.assertTrue(productive_started, msg="No first productive action startup event emitted")
        sim.stop()

    def test_run_summary_contains_startup_progression_measures(self):
        sim = SimulationState(phases=[])
        for _ in range(6):
            sim.update(1.0)
        summary = sim.metrics.finalize()
        startup = summary["process"]["startup_progression"]
        self.assertIn("agents_left_spawn_count", startup)
        self.assertIn("time_to_leave_spawn_by_agent", startup)
        self.assertIn("first_productive_action_time_by_agent", startup)
        sim.stop()

    def test_agent_can_reach_and_ingest_allowed_packet(self):
        env = Environment(phases=[])
        station = env.objects["Architect_Info"]
        center = (station["position"][0] + (station["size"][0] / 2.0), station["position"][1] + (station["size"][1] / 2.0))
        agent = Agent(name="Architect", role="Architect", position=center)
        agent.allowed_packet = ["Architect_Packet"]

        agent.update_knowledge(env)

        info_ids = {info.id for info in agent.mental_model["information"]}
        self.assertIn("I004", info_ids)

    def test_agent_can_route_around_central_blocked_zone_to_reachable_target(self):
        env = Environment(phases=[])
        start = env.get_spawn_point("Engineer")
        target = env.objects["Team_Info"]["position"]
        agent = Agent(name="Engineer", role="Engineer", position=start)

        for _ in range(40):
            agent.move_toward(target, dt=0.25, environment=env)

        self.assertTrue(
            env.can_access_info(agent.position, "Team_Info", role="Engineer"),
            msg=f"Agent failed to route around obstacle; final position: {agent.position}",
        )

    def test_early_simulation_steps_do_not_deadlock_all_agents_at_central_block(self):
        sim = SimulationState(phases=[])
        central_block = sim.environment.objects["Blocked_Zone_AC"]
        (x1, y1), (x2, y2) = central_block["corners"]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        def near_central_block(pos, margin=0.25):
            return (x_min - margin) <= pos[0] <= (x_max + margin) and (y_min - margin) <= pos[1] <= (y_max + margin)

        for _ in range(20):
            sim.update(1.0)

        clustered = [a for a in sim.agents if near_central_block(a.position)]
        self.assertLess(
            len(clustered),
            len(sim.agents),
            msg="All agents remained clustered around the central blocked zone",
        )


class TestBuildTargetAndProgress(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_build_target_uses_accessible_interaction_zone_not_table_center(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))

        target = agent._select_build_target(env)
        self.assertIsNotNone(target)
        self.assertNotEqual(target, env.objects["Table_B"]["position"])
        self.assertTrue(env.is_point_navigable(target), msg=f"Build target should be navigable: {target}")

    def test_agent_can_progress_toward_build_interaction_target(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        target = agent._select_build_target(env)
        self.assertIsNotNone(target)

        start_dist = ((agent.position[0] - target[0]) ** 2 + (agent.position[1] - target[1]) ** 2) ** 0.5
        for _ in range(30):
            agent.move_toward(target, dt=0.25, environment=env)
        end_dist = ((agent.position[0] - target[0]) ** 2 + (agent.position[1] - target[1]) ** 2) ** 0.5

        self.assertLess(end_dist, start_dist)
        self.assertTrue(env.is_point_navigable(agent.position), msg="Agent should remain on navigable terrain while pursuing build interaction")
        self.assertFalse(
            env.is_in_blocked_zone(agent.position),
            msg=f"Agent deadlocked into blocked center while pursuing build target: {agent.position}",
        )


class TestInspectSemanticsAndLegacySeparation(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_inspect_decision_without_explicit_target_resolves_to_source_target(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        decision = type("D", (), {"selected_action": None, "target_id": None})()
        from modules.action_schema import ExecutableActionType
        decision.selected_action = ExecutableActionType.INSPECT_INFORMATION_SOURCE

        actions = agent._translate_brain_decision_to_legacy_action(decision, env)
        self.assertEqual(actions[0]["type"], "move_to")
        self.assertIn("source_target_id", actions[0])
        self.assertIsNotNone(actions[0].get("target"))

    def test_fully_inspected_source_is_not_endlessly_reselected(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        agent.source_inspection_state["Team_Info"] = "inspected"

        class D:
            selected_action = None
            target_id = None

        from modules.action_schema import ExecutableActionType
        D.selected_action = ExecutableActionType.INSPECT_INFORMATION_SOURCE
        actions = agent._translate_brain_decision_to_legacy_action(D, env)
        self.assertNotEqual(actions[0].get("source_target_id"), "Team_Info")

    def test_contradiction_marks_source_revisitable(self):
        env = Environment(phases=[])
        agent = Agent(name="Botanist", role="Botanist", position=env.get_spawn_point("Botanist"))
        agent.source_inspection_state["Team_Info"] = "inspected"
        agent.current_inspect_target_id = "Team_Info"
        agent.activity_log.append("Mismatch with construction: reevaluating knowledge")

        agent.update_knowledge(env, full_packet_sweep=False)
        self.assertEqual(agent.source_inspection_state["Team_Info"], "revisitable_due_to_gap")

    def test_repeated_stalls_drive_conservative_retargeting(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=env.get_spawn_point("Architect"))
        agent.inspect_stall_counts["Team_Info"] = 4
        agent.source_inspection_state["Team_Info"] = "unseen"

        class D:
            selected_action = None
            target_id = None

        from modules.action_schema import ExecutableActionType
        D.selected_action = ExecutableActionType.INSPECT_INFORMATION_SOURCE
        actions = agent._translate_brain_decision_to_legacy_action(D, env)
        self.assertNotEqual(actions[0].get("source_target_id"), "Team_Info")

    def test_legacy_packet_sweep_failure_noise_is_suppressed_from_last_action(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        agent.update_knowledge(env, full_packet_sweep=True)

        # Old behavior spammed "Could not access packet ..."; this should no longer become status text.
        self.assertNotIn("Could not access packet", agent.status_last_action)
