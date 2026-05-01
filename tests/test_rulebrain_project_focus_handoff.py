import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class TestRuleBrainProjectFocusHandoff(unittest.TestCase):
    def test_material_incomplete_focus_prefers_transport(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            pid, _ = cm.create_project("site_b", structure_type="house")
            decision = BrainDecision(selected_action=ExecutableActionType.REASSESS_PLAN, target_id=pid, confidence=0.6)
            rewritten = agent._suppress_reassess_when_project_work_available(decision, sim.environment, sim_state=sim)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            self.assertEqual(rewritten.target_id, pid)
        finally:
            sim.stop()

    def test_material_complete_focus_prefers_build(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            pid, _ = cm.create_project("site_b", structure_type="house")
            p = cm.projects[pid]
            cm.deliver_resource(pid, "bricks", quantity=int(p["required_resources"]["bricks"]))
            decision = BrainDecision(selected_action=ExecutableActionType.REASSESS_PLAN, target_id=pid, confidence=0.6)
            rewritten = agent._suppress_reassess_when_project_work_available(decision, sim.environment, sim_state=sim)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.CONTINUE_CONSTRUCTION)
            self.assertEqual(rewritten.target_id, pid)
        finally:
            sim.stop()

    def test_non_reassess_action_passthrough(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            decision = BrainDecision(selected_action=ExecutableActionType.WAIT, target_id="missing_project", confidence=0.6)
            rewritten = agent._suppress_reassess_when_project_work_available(decision, sim.environment, sim_state=sim)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.WAIT)
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
