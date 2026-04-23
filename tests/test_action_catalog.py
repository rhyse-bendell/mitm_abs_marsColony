import unittest
from dataclasses import replace
from unittest import mock

from modules.action_catalog import (
    ACTION_CATALOG,
    ACTION_ALIASES,
    normalize_action_alias,
    validate_action_catalog_coverage,
    validate_rulebrain_action_references,
)
from modules.action_schema import ExecutableActionType
from modules.brain_provider import RuleBrain


class TestActionCatalog(unittest.TestCase):
    def test_every_executable_action_is_cataloged(self):
        errors = validate_action_catalog_coverage()
        self.assertFalse(errors, msg=f"coverage errors: {errors}")

    def test_planner_visible_actions_have_translation_or_partial_status(self):
        for action_id, entry in ACTION_CATALOG.items():
            if not entry.planner_visible:
                continue
            has_runtime_path = bool(entry.translation_destination.strip())
            self.assertTrue(
                has_runtime_path or entry.status in {"partial", "deprecated", "experimental"},
                msg=f"planner-visible action missing runtime path: {action_id}",
            )

    def test_catalog_keys_match_enum_values(self):
        for action_id, entry in ACTION_CATALOG.items():
            self.assertEqual(action_id, entry.action.value)

    def test_catalog_entry_identity_properties_are_derived_from_enum(self):
        for entry in ACTION_CATALOG.values():
            self.assertEqual(entry.action_id, entry.action.value)
            self.assertEqual(entry.enum_name, entry.action.name)

    def test_alias_normalization_supported(self):
        expected = {
            "inspect": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
            "inspect_information_source": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
            "communicate_with_team": ExecutableActionType.COMMUNICATE.value,
            "ask_for_help": ExecutableActionType.REQUEST_ASSISTANCE.value,
            "transport": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "deliver_resources": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "start_build": ExecutableActionType.START_CONSTRUCTION.value,
            "continue_build": ExecutableActionType.CONTINUE_CONSTRUCTION.value,
            "validate": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
            "repair": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
            "reassess": ExecutableActionType.REASSESS_PLAN.value,
            "wait": ExecutableActionType.WAIT.value,
            "observe": ExecutableActionType.OBSERVE_ENVIRONMENT.value,
        }
        for alias, canonical in expected.items():
            self.assertEqual(canonical, normalize_action_alias(alias), msg=f"alias failed: {alias}")

    def test_alias_normalization_rejects_ambiguous_or_unsupported(self):
        for alias in ("build", "do_something", "", "   "):
            self.assertIsNone(normalize_action_alias(alias), msg=f"alias should be rejected: {alias!r}")

    def test_alias_map_matches_normalized_aliases(self):
        for action_id, entry in ACTION_CATALOG.items():
            for alias in entry.aliases:
                self.assertEqual(action_id, ACTION_ALIASES[alias.strip().lower()])

    def test_catalog_validation_detects_duplicate_aliases(self):
        original_actions = tuple(ACTION_CATALOG.values())
        collision = replace(original_actions[1], aliases=(*original_actions[1].aliases, original_actions[0].aliases[0]))
        with mock.patch("modules.action_catalog.ACTION_CATALOG", {e.action_id: e for e in (*original_actions, collision)}):
            errors = validate_action_catalog_coverage()
        self.assertTrue(any(err.startswith("duplicate_alias:") for err in errors), msg=f"errors: {errors}")

    def test_rulebrain_action_maps_do_not_drift_from_catalog(self):
        errors = validate_rulebrain_action_references(
            RuleBrain.MODE_ACTION_PREFERENCES,
            RuleBrain.STEP_ACTION_MAP,
            RuleBrain.FALLBACK_ACTION_ORDER,
        )
        self.assertFalse(errors, msg=f"rulebrain drift errors: {errors}")

    def test_construction_action_traceability_in_catalog(self):
        required = {
            ExecutableActionType.START_CONSTRUCTION.value,
            ExecutableActionType.CONTINUE_CONSTRUCTION.value,
            ExecutableActionType.VALIDATE_CONSTRUCTION.value,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
        }
        for action_id in required:
            entry = ACTION_CATALOG[action_id]
            self.assertTrue(entry.planner_visible)
            self.assertTrue(entry.executable)
            self.assertIn("agent.translate", entry.translation_destination)
            self.assertIn("modules/agent.py", entry.execution_owner)


if __name__ == "__main__":
    unittest.main()
