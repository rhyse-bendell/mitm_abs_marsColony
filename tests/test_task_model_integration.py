import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import REQUIRED_TASK_FILES, TaskModelError, load_task_model


class TaskModelIntegrationTests(unittest.TestCase):
    def test_mars_task_package_loads(self):
        model = load_task_model("mars_colony")
        self.assertGreater(len(model.sources), 0)
        self.assertGreater(len(model.dik_elements), 0)
        self.assertGreater(len(model.derivations), 0)
        self.assertGreater(len(model.rules), 0)
        self.assertGreater(len(model.goals), 0)
        self.assertGreater(len(model.plan_methods), 0)
        self.assertGreater(len(model.artifacts), 0)
        self.assertGreater(len(model.environment_objects), 0)
        self.assertGreater(len(model.zones), 0)
        self.assertGreater(len(model.interaction_targets), 0)
        self.assertGreater(len(model.spawn_points), 0)
        self.assertGreater(len(model.resource_nodes), 0)
        self.assertGreater(len(model.phases), 0)
        self.assertGreater(len(model.roles), 0)
        self.assertGreater(len(model.agent_defaults), 0)
        self.assertGreater(len(model.action_availability), 0)
        self.assertGreater(len(model.action_parameters), 0)
        self.assertGreater(len(model.communication_catalog), 0)
        self.assertGreater(len(model.construction_templates), 0)

    def test_missing_required_file_raises_clear_error(self):
        temp_dir = tempfile.mkdtemp(prefix="task_model_missing_")
        try:
            base = Path(temp_dir) / "tmp_task"
            base.mkdir(parents=True, exist_ok=True)
            src = Path("config/tasks/mars_colony")
            for _, fname in REQUIRED_TASK_FILES.items():
                if fname == "rule_definitions.csv":
                    continue
                shutil.copy(src / fname, base / fname)

            with self.assertRaises(TaskModelError) as ctx:
                load_task_model("tmp_task", config_root=temp_dir)
            self.assertIn("rule_definitions.csv", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)

    def test_environment_initializes_from_task_package_content(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        self.assertIn("Team_Info", env.objects)
        self.assertIn("Zone_Table_B", env.zones)
        self.assertIn("Build_Table_B", env.interaction_targets)
        self.assertEqual(env.get_spawn_point("Architect"), (6.9, 1.2))

    def test_source_inspection_yields_task_backed_dik_elements(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Architect", role="Architect", position=(8.0, 6.6))
        agent.allowed_packet = ["Team_Info", "Architect_Info"]

        with patch("modules.agent.random.random", return_value=0.0):
            success = agent._inspect_source(env, "Team_Info")
        self.assertTrue(success)

        info_ids = {i.id for i in agent.mental_model["information"]}
        data_ids = {d.id for d in agent.mental_model["data"]}
        self.assertIn("I_SHARED_PHASE_OBJECTIVES", info_ids)
        self.assertIn("D_SHARED_TEAM_MISSION", data_ids)

    def test_data_to_information_derivation_executes(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Engineer", role="Engineer", position=(8.0, 6.6))
        agent.task_model = model
        agent.allowed_packet = ["Team_Info", "Engineer_Info"]

        team_packet = env.knowledge_packets["Team_Info"]
        wanted = {"D_SHARED_PLANNING_REVIEW_ORDER", "D_SHARED_PHASE1_TARGET_50_CIV", "D_SHARED_PHASE2_TARGET_40_CIV_20_VIP"}
        for d in team_packet["data"]:
            if d.id in wanted:
                agent.mental_model["data"].add(d)

        agent._apply_task_derivations(sim_state=None)
        info_ids = {i.id for i in agent.mental_model["information"]}

        self.assertIn("I_SHARED_PHASE_OBJECTIVES", info_ids)
        self.assertIn("DRV_PHASE_OBJECTIVES", agent.executed_derivations)

    def test_information_to_knowledge_derivation_executes(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Botanist", role="Botanist", position=(8.0, 6.6))
        agent.task_model = model
        agent.allowed_packet = ["Team_Info", "Botanist_Info"]

        phase_objective_info = next(i for i in env.knowledge_packets["Team_Info"]["information"] if i.id == "I_SHARED_PHASE_OBJECTIVES")
        agent.mental_model["information"].add(phase_objective_info)
        agent._apply_task_derivations(sim_state=None)

        self.assertIn("K_PHASE1_SUPPORT_TARGET", set(agent.mental_model["knowledge"].rules))
        self.assertIn("DRV_PHASE1_TARGET_KNOWLEDGE", agent.executed_derivations)

    def test_construction_artifacts_align_with_definitions(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        for project in env.construction.projects.values():
            artifact_type = project.get("artifact_type")
            self.assertIn(artifact_type, model.artifacts)

    def test_phases_roles_actions_and_construction_are_task_driven(self):
        sim = SimulationState(speed="Fast", flash_mode=True)
        self.assertEqual(sim.environment.phases[0]["name"], "Phase 1")
        self.assertEqual([a.role for a in sim.agents], ["Architect", "Engineer", "Botanist"])
        action_ids = sim.task_model.enabled_actions_for_role("Architect")
        self.assertIn("transport_resources", action_ids)
        self.assertIn("Build_Table_B", sim.environment.construction.projects)
        sim.stop()

    def test_headless_simulation_runs_with_task_model(self):
        sim = SimulationState(speed="Fast", flash_mode=True)
        for _ in range(3):
            sim.update(0.1)
        sim.stop()
        self.assertEqual(sim.task_model.task_id, "mars_colony")


if __name__ == "__main__":
    unittest.main()
