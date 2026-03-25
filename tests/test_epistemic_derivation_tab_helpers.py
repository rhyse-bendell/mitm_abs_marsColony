import unittest
from types import SimpleNamespace

from interface import MarsColonyInterface


class EpistemicDerivationTabHelperTests(unittest.TestCase):
    def test_abbreviated_element_id_strips_type_prefix(self):
        self.assertEqual(MarsColonyInterface._abbreviated_element_id("D_ARCH_AIRLOCK_SIZE", "data"), "ARCH_AIRLOCK_SIZE")
        self.assertEqual(MarsColonyInterface._abbreviated_element_id("I_BUILD_PLAN", "information"), "BUILD_PLAN")
        self.assertEqual(MarsColonyInterface._abbreviated_element_id("K_RULE_CHAIN", "knowledge"), "RULE_CHAIN")

    def test_wrap_flow_rows_wraps_when_width_exceeded(self):
        rows = MarsColonyInterface._wrap_flow_rows([80, 90, 70], max_width=180, hgap=6)
        self.assertEqual(rows, [[0, 1], [2]])

    def test_derivation_groups_preserve_alternative_pathways(self):
        task_model = SimpleNamespace(
            derivations={
                "d1": SimpleNamespace(enabled=True, output_type="information", output_element_id="I_A"),
                "d2": SimpleNamespace(enabled=True, output_type="information", output_element_id="I_A"),
                "d3": SimpleNamespace(enabled=True, output_type="information", output_element_id="I_B"),
                "d4": SimpleNamespace(enabled=True, output_type="knowledge", output_element_id="K_A"),
            }
        )
        groups = MarsColonyInterface._build_derivation_groups(task_model, output_type="information")
        grouped = {output_id: len(paths) for output_id, paths in groups}
        self.assertEqual(grouped["I_A"], 2)
        self.assertEqual(grouped["I_B"], 1)
        self.assertNotIn("K_A", grouped)

    def test_pathway_satisfied_uses_selected_view_state(self):
        app = MarsColonyInterface.__new__(MarsColonyInterface)
        derivation = SimpleNamespace(required_inputs=["D_1", "I_2"], min_required_count=0)
        task_model = SimpleNamespace(
            dik_elements={
                "D_1": SimpleNamespace(element_type="data"),
                "I_2": SimpleNamespace(element_type="information"),
            }
        )
        state = {"data": {"D_1"}, "information": set(), "knowledge": set(), "rules": set()}
        self.assertFalse(app._pathway_satisfied(derivation, state, task_model))
        state["information"].add("I_2")
        self.assertTrue(app._pathway_satisfied(derivation, state, task_model))


if __name__ == "__main__":
    unittest.main()
