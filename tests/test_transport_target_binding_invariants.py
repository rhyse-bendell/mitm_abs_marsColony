import unittest

from modules.action_schema import ExecutableActionType
from modules.simulation import SimulationState


class TransportTargetBindingInvariantTests(unittest.TestCase):
    def _event_types(self, sim):
        return [e.get("event_type") for e in sim.logger.recent_events]

    def test_pickup_stage_project_bound_target_is_repaired_to_pile(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = sim.environment.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
        pile_pos = tuple(sim.environment.construction.resource_nodes["pile_a"].position)
        dropoff_pos = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)

        agent.transport_state.update(
            {
                "stage": "pickup",
                "carrying": {"resource_type": None, "quantity": 0},
                "pickup_source_id": "pile_a",
                "bound_project_id": project_id,
            }
        )
        action = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "target": dropoff_pos,
            "target_id": "pile_a",
            "target_kind": "pickup",
        }
        before = int(sim.environment.construction.projects[project_id]["delivered_resources"]["bricks"])
        agent.position = (5.6, 3.8)
        agent.active_actions = [action]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertEqual(tuple(action.get("target")), pile_pos)
        self.assertEqual(action.get("target_id"), "pile_a")
        self.assertEqual(action.get("target_kind"), "pickup")
        self.assertEqual(before, int(sim.environment.construction.projects[project_id]["delivered_resources"]["bricks"]))
        self.assertIn("transport_pickup_target_mismatch_repaired", self._event_types(sim))
        sim.stop()

    def test_dropoff_stage_pickup_bound_target_is_repaired_to_project(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = sim.environment.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
        pile_pos = tuple(sim.environment.construction.resource_nodes["pile_a"].position)
        dropoff_pos = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)

        agent.transport_state.update(
            {
                "stage": "in_transit",
                "carrying": {"resource_type": "bricks", "quantity": 1},
                "pickup_source_id": "pile_a",
                "bound_project_id": project_id,
            }
        )
        action = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "target": pile_pos,
            "target_id": "pile_a",
            "target_kind": "pickup",
        }
        before = int(sim.environment.construction.projects[project_id]["delivered_resources"]["bricks"])
        agent.position = pile_pos
        expected_dropoff = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
        agent.active_actions = [action]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertEqual(tuple(action.get("target")), tuple(expected_dropoff))
        self.assertEqual(action.get("target_id"), project_id)
        self.assertEqual(action.get("target_kind"), "dropoff")
        self.assertEqual(before, int(sim.environment.construction.projects[project_id]["delivered_resources"]["bricks"]))
        self.assertIn("transport_dropoff_target_mismatch_repaired", self._event_types(sim))
        sim.stop()

    def test_valid_transport_targets_remain_unchanged(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = sim.environment.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
        pile_pos = tuple(sim.environment.construction.resource_nodes["pile_a"].position)
        dropoff_pos = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)

        pickup_action = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "target": pile_pos,
            "target_id": "pile_a",
            "target_kind": "pickup",
        }
        agent.transport_state.update(
            {
                "stage": "pickup",
                "carrying": {"resource_type": None, "quantity": 0},
                "pickup_source_id": "pile_a",
                "bound_project_id": project_id,
            }
        )
        agent.position = (6.2, 4.6)
        agent.active_actions = [pickup_action]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(tuple(pickup_action.get("target")), pile_pos)

        expected_dropoff = sim.environment.get_interaction_target_position(project_id, from_position=pile_pos)
        dropoff_action = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "target": tuple(expected_dropoff),
            "target_id": project_id,
            "target_kind": "dropoff",
        }
        agent.transport_state.update(
            {
                "stage": "in_transit",
                "carrying": {"resource_type": "bricks", "quantity": 1},
                "pickup_source_id": "pile_a",
                "bound_project_id": project_id,
            }
        )
        agent.position = pile_pos
        agent.active_actions = [dropoff_action]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(tuple(dropoff_action.get("target")), tuple(expected_dropoff))
        self.assertNotIn("transport_target_binding_repaired", self._event_types(sim))
        sim.stop()

    def test_tenth_delivery_path_repairs_pickup_binding_and_advances(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id, _ = sim.environment.construction.create_project("site_b", structure_type="water_generator")
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        delivered_seed = max(0, required - 1)
        project["delivered_resources"]["bricks"] = delivered_seed
        sim.environment.construction.update()
        pile_pos = tuple(sim.environment.construction.resource_nodes["pile_a"].position)
        dropoff_pos = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)

        action = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "target": dropoff_pos,
            "target_id": "pile_a",
            "target_kind": "pickup",
        }
        agent.transport_state.update(
            {
                "stage": "pickup",
                "carrying": {"resource_type": None, "quantity": 0},
                "pickup_source_id": "pile_a",
                "bound_project_id": project_id,
            }
        )
        agent.position = (5.6, 3.8)
        agent.active_actions = [action]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertEqual(tuple(action.get("target")), pile_pos)
        self.assertEqual(agent.transport_state.get("stage"), "pickup")
        self.assertEqual(int(project["delivered_resources"]["bricks"]), delivered_seed)
        self.assertIn("transport_pickup_target_mismatch_repaired", self._event_types(sim))
        sim.stop()


if __name__ == "__main__":
    unittest.main()
