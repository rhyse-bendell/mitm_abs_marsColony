import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import TaskModelError, load_task_model


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

    def test_missing_required_file_raises_clear_error(self):
        temp_dir = tempfile.mkdtemp(prefix="task_model_missing_")
        try:
            base = Path(temp_dir) / "tmp_task"
            base.mkdir(parents=True, exist_ok=True)
            for csv_path in Path("config/tasks/mars_colony").glob("*.csv"):
                if csv_path.name != "rule_definitions.csv":
                    shutil.copy(csv_path, base / csv_path.name)

            with self.assertRaises(TaskModelError) as ctx:
                load_task_model("tmp_task", config_root=temp_dir)
            self.assertIn("rule_definitions.csv", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)

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

    def test_headless_simulation_runs_with_task_model(self):
        sim = SimulationState(speed="Fast", flash_mode=True)
        for _ in range(3):
            sim.update(0.1)
        sim.stop()
        self.assertEqual(sim.task_model.task_id, "mars_colony")


if __name__ == "__main__":
    unittest.main()
