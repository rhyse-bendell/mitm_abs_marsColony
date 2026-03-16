import unittest

from modules.brain_provider import select_productive_fallback_action


class TestFallbackPolicy(unittest.TestCase):
    def test_prefers_build_family_actions_when_available(self):
        allowed = [
            {"action_type": "wait"},
            {"action_type": "inspect_information_source", "target_id": "Team_Info", "target_zone": "Zone_Team_Info"},
            {"action_type": "continue_construction", "target_id": "Habitat_A"},
        ]
        step = select_productive_fallback_action(allowed)
        self.assertEqual(step.action_type.value, "continue_construction")
        self.assertEqual(step.target_id, "Habitat_A")

    def test_prefers_inspect_when_build_actions_not_available(self):
        allowed = [
            {"action_type": "wait"},
            {"action_type": "observe_environment"},
            {"action_type": "inspect_information_source", "target_id": "Team_Info", "target_zone": "Zone_Team_Info"},
        ]
        step = select_productive_fallback_action(allowed)
        self.assertEqual(step.action_type.value, "inspect_information_source")
        self.assertEqual(step.target_id, "Team_Info")

    def test_falls_back_to_observe_or_wait_when_needed(self):
        observe_only = [{"action_type": "observe_environment"}]
        step = select_productive_fallback_action(observe_only)
        self.assertEqual(step.action_type.value, "observe_environment")

        none_allowed = []
        step2 = select_productive_fallback_action(none_allowed)
        self.assertEqual(step2.action_type.value, "wait")

    def test_prefers_reachable_productive_action_when_available(self):
        allowed = [
            {"action_type": "inspect_information_source", "target_id": "Blocked_Info", "reachable": False},
            {"action_type": "inspect_information_source", "target_id": "Team_Info", "reachable": True},
            {"action_type": "observe_environment"},
        ]
        step = select_productive_fallback_action(allowed)
        self.assertEqual(step.action_type.value, "inspect_information_source")
        self.assertEqual(step.target_id, "Team_Info")


if __name__ == "__main__":
    unittest.main()
