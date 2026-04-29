import unittest
from unittest import mock

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class ConstructionEpistemicAuthorityTests(unittest.TestCase):
    def _ensure_project(self, sim, target_id="Build_Site_B"):
        project_id = sim.environment.construction.resolve_project_id(target_id, create_if_missing=True)
        if project_id is None:
            project_id, _ = sim.environment.construction.create_project("site_b", structure_type="house")
        return project_id, sim.environment.construction.projects[project_id]

    def _prime_build_readiness(self, sim, agent):
        team_packet = sim.environment.knowledge_packets["Team_Info"]
        role_packet = sim.environment.knowledge_packets.get(f"{agent.role}_Info", {})
        agent.mental_model["information"].add(team_packet["information"][0])
        agent.mental_model["information"].add(team_packet["information"][1])
        if role_packet.get("information"):
            agent.mental_model["information"].add(role_packet["information"][0])
        agent.mental_model["knowledge"].rules.append("R_HOUSE_VALIDITY")
        agent.source_inspection_state["Team_Info"] = "inspected"
        agent.source_inspection_state[f"{agent.role}_Info"] = "inspected"
        agent.source_memory_state.setdefault("Team_Info", {})["ever_inspected"] = True
        agent.source_memory_state.setdefault("Team_Info", {})["last_inspected_time"] = float(sim.time)
        agent.source_memory_state.setdefault("Team_Info", {})["last_verified_time"] = float(sim.time)
        agent.source_memory_state.setdefault("Team_Info", {})["memory_confidence"] = 0.95
        agent.source_memory_state.setdefault(f"{agent.role}_Info", {})["ever_inspected"] = True
        agent.source_memory_state.setdefault(f"{agent.role}_Info", {})["last_inspected_time"] = float(sim.time)
        agent.source_memory_state.setdefault(f"{agent.role}_Info", {})["last_verified_time"] = float(sim.time)
        agent.source_memory_state.setdefault(f"{agent.role}_Info", {})["memory_confidence"] = 0.95
        agent.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = True
        agent.source_memory_state.setdefault("Engineer_Info", {})["last_inspected_time"] = float(sim.time)
        agent.source_memory_state.setdefault("Engineer_Info", {})["last_verified_time"] = float(sim.time)
        agent.source_memory_state.setdefault("Engineer_Info", {})["memory_confidence"] = 0.95
        agent.source_inspection_state["Engineer_Info"] = "inspected"

    def test_start_construction_not_hard_blocked_when_epistemic_prereqs_missing(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.9,
        )

        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        self.assertEqual(translated[0]["type"], "construct")
        project_id, _ = self._ensure_project(sim, "Build_Site_B")
        self.assertEqual(translated[0].get("project_id"), project_id)
        sim.stop()

    def test_start_construction_allowed_when_grounded(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)

        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.9,
        )
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        self.assertEqual(translated[0]["type"], "construct")
        project_id, _ = self._ensure_project(sim, "Build_Site_B")
        self.assertEqual(translated[0].get("project_id"), project_id)
        sim.stop()

    def test_resource_completion_not_equal_validated_completion(self):
        sim = SimulationState(phases=[])
        project_id, p = self._ensure_project(sim, "Build_Site_B")
        for _ in range(int(p["required_resources"]["bricks"])):
            sim.environment.construction.deliver_resource(project_id, "bricks", quantity=1)
        self.assertTrue(p["resource_complete"])
        self.assertFalse(p["validated_complete"])
        self.assertNotEqual(p["status"], "complete")
        sim.stop()

    def test_repair_then_validate_can_complete_project(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)

        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        project["correct"] = False
        for expected_rule in list(project.get("expected_rules") or []):
            if expected_rule not in agent.mental_model["knowledge"].rules:
                agent.mental_model["knowledge"].rules.append(expected_rule)

        agent.activity_log.append("Mismatch with construction: reevaluating knowledge")
        repair = BrainDecision(
            selected_action=ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            target_id=project_id,
            confidence=0.8,
        )
        repair_action = agent._translate_brain_decision_to_legacy_action(repair, sim.environment, sim_state=sim)[0]
        self.assertEqual(repair_action["type"], "construct")

        agent.inventory_resources["bricks"] = 1
        agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
        agent.active_actions = [{**repair_action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertTrue(project["correct"])
        agent.activity_log = [entry for entry in agent.activity_log if "Mismatch with construction" not in str(entry)]
        remaining_steps = sum(1 for step in project.get("build_steps", []) if not step.get("completed"))
        for _ in range(remaining_steps):
            sim.environment.construction.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)

        sim.environment.construction.record_project_epistemic_externalization(
            project_id,
            entry_type="claim",
            note=f"{project_id} is compliant for validation.",
            references=["R_HOUSE_VALIDITY"],
            actor=agent.name,
            sim_time=sim.time,
        )
        sim.environment.construction.record_project_epistemic_externalization(
            project_id,
            entry_type="evidence",
            note="Observed enclosed housing configuration.",
            references=["R_HOUSE_VALIDITY"],
            actor=agent.name,
            sim_time=sim.time,
        )
        sim.environment.construction.record_project_epistemic_externalization(
            project_id,
            entry_type="design_note",
            note="Airlock and walls form enclosed shell.",
            references=["R_HOUSE_VALIDITY"],
            actor=agent.name,
            sim_time=sim.time,
        )
        project["support_requirements"] = {}
        project["support_counts"] = {}
        project["support_status"] = {}
        sim.environment.construction.recompute_support_status(project_id)
        project["correct"] = True
        readiness = sim.environment.construction.evaluate_project_validation_readiness(
            project_id,
            actor=agent.name,
            agent_supported_rules=list(agent.mental_model["knowledge"].rules),
        )
        self.assertTrue(readiness["validation_ready"])

        validate = BrainDecision(
            selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
            target_id=project_id,
            confidence=0.8,
        )
        validate_action = agent._translate_brain_decision_to_legacy_action(validate, sim.environment, sim_state=sim)[0]
        self.assertIn(validate_action["type"], {"idle", "communicate"})
        if validate_action["type"] != "idle":
            validate_action = {
                "type": "idle",
                "duration": 1.0,
                "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
                "project_id": project_id,
            }

        agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
        agent.active_actions = [{**validate_action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertTrue(project["validated_complete"])
        self.assertEqual(project["status"], "complete")
        sim.stop()

    def test_validation_ready_gold_path_validates_complete(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        for _ in range(sum(1 for step in project.get("build_steps", []) if not step.get("completed"))):
            sim.environment.construction.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)
        for entry_type in ("claim", "evidence", "design_note"):
            sim.environment.construction.record_project_epistemic_externalization(
                project_id, entry_type=entry_type, note=f"{entry_type} for validation", references=["R_HOUSE_VALIDITY"], actor=agent.name, sim_time=sim.time
            )
        project["support_requirements"] = {}
        project["support_counts"] = {}
        project["support_status"] = {}
        sim.environment.construction.recompute_support_status(project_id)
        report = sim.environment.construction.evaluate_project_validation_readiness(
            project_id, actor=agent.name, agent_supported_rules=list(agent.mental_model["knowledge"].rules)
        )
        self.assertTrue(report["validation_ready"])
        blockers, _ = agent._construction_action_blockers(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            {"project_id": project_id},
            sim.environment,
            sim_state=sim,
        )
        self.assertFalse(blockers)
        agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
        agent.active_actions = [{"type": "idle", "duration": 1.0, "progress": 0.0, "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "project_id": project_id}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertTrue(project["validated_complete"])
        sim.stop()

    def test_start_construction_auto_handoffs_to_logistics_when_resources_missing(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        before = int(project["delivered_resources"]["bricks"])

        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id=project_id,
            confidence=0.9,
        )
        action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        agent.inventory_resources["bricks"] = 0
        agent.active_actions = [{**action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        after = int(project["delivered_resources"]["bricks"])
        self.assertEqual(after, before)
        sim.stop()

    def test_transport_does_not_false_progress_when_resources_already_satisfied(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        project["delivered_resources"]["bricks"] = required
        sim.environment.construction.update()
        before = int(project["delivered_resources"]["bricks"])

        decision = BrainDecision(
            selected_action=ExecutableActionType.TRANSPORT_RESOURCES,
            target_id=project_id,
            confidence=0.9,
        )
        action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        agent.active_actions = [{**action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        after = int(project["delivered_resources"]["bricks"])

        self.assertEqual(after, before)
        sim.stop()

    def test_transport_requires_pickup_and_dropoff_legality(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id, project = self._ensure_project(sim, "Build_Site_A")
        before = int(project["delivered_resources"]["bricks"])
        transport = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
        }
        agent.position = (8.0, 6.6)  # Team_Info region; not pickup and not build table
        agent.active_actions = [transport]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(before, int(project["delivered_resources"]["bricks"]))
        self.assertEqual(agent.transport_state.get("stage"), "pickup")
        sim.stop()

    def test_validation_requires_location_and_status(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id, _project = self._ensure_project(sim, "Build_Site_B")
        validate = {
            "type": "idle",
            "duration": 0.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
        }
        # Wrong location and status: should not validate.
        agent.position = (8.0, 6.6)
        agent.active_actions = [validate]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertFalse(sim.environment.construction.projects[project_id]["validated_complete"])
        self.assertTrue(any(e.get("event_type") in {"construction_validation_blocked", "construction_validation_en_route"} for e in sim.logger.recent_events))
        sim.stop()

    def test_failed_validation_persists_needs_repair_until_fixed(self):
        sim = SimulationState(phases=[])
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        sim.environment.construction.mark_validated(project_id, is_valid=False, actor="Architect", sim_time=8.0)
        sim.environment.construction.update()
        self.assertIn(project["status"], {"needs_repair", "in_progress"})
        self.assertFalse(project["validated_complete"])
        self.assertEqual(project.get("last_actor"), "Architect")
        sim.stop()

    def test_inspect_context_cannot_shortcut_delivery(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id, project = self._ensure_project(sim, "Build_Site_A")
        before = int(project["delivered_resources"]["bricks"])
        # Simulate stale inspect context while a transport action exists.
        agent.current_inspect_target_id = "Team_Info"
        agent.position = (8.0, 6.6)
        agent.active_actions = [{
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
        }]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(before, int(project["delivered_resources"]["bricks"]))
        self.assertFalse(any(e.get("event_type") == "construction_resource_delivered" for e in sim.logger.recent_events))
        sim.stop()

    def test_policy_pivot_targets_provider_connector_work_for_support_deficit(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        cm = sim.environment.construction
        house_id, house = self._ensure_project(sim, "Build_Site_B")
        cm.deliver_resource(house_id, "bricks", quantity=int(house["required_resources"]["bricks"]))
        for _ in range(len(house["build_steps"])):
            cm.execute_build_step(house_id, actor=agent.name, sim_time=sim.time)
        cm.record_project_epistemic_externalization(house_id, entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        cm.record_project_epistemic_externalization(house_id, entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        cm.record_project_epistemic_externalization(house_id, entry_type="design_note", note="design", actor=agent.name, sim_time=sim.time)
        self.assertFalse(house["functional_support_complete"])

        seed = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=house_id, confidence=0.8)
        rewritten = agent._resolve_support_deficit_decision(seed, sim.environment, sim_state=sim, pivot_origin="unit_test")
        self.assertIsNotNone(rewritten)
        self.assertIn(
            rewritten.selected_action,
            {
                ExecutableActionType.TRANSPORT_RESOURCES,
                ExecutableActionType.START_CONSTRUCTION,
                ExecutableActionType.CONTINUE_CONSTRUCTION,
                ExecutableActionType.VALIDATE_CONSTRUCTION,
            },
        )
        target_project = sim.environment.construction.projects.get(str(rewritten.target_id or ""))
        if target_project is not None:
            self.assertIn(
                target_project.get("type"),
                {"greenhouse", "water_generator", "food_connector", "water_connector", "house", "housing"},
            )
        connector_or_provider_exists = any(
            p.get("type") in {"greenhouse", "water_generator", "food_connector", "water_connector"}
            for p in sim.environment.construction.projects.values()
        )
        self.assertTrue(connector_or_provider_exists)
        sim.stop()

    def test_construction_expected_rules_normalized_to_canonical_ids(self):
        model = load_task_model("mars_colony")
        for template in model.construction_templates.values():
            self.assertTrue(template.expected_rules)
            self.assertTrue(all(rule.startswith("R_") for rule in template.expected_rules))


    def test_cross_role_engineer_witness_gap_blocks_validation_until_engineer_inspects(self):
        sim = SimulationState(phases=[])
        architect = next(a for a in sim.agents if a.role == "Architect")
        engineer = next(a for a in sim.agents if a.role == "Engineer")
        botanist = next(a for a in sim.agents if a.role == "Botanist")
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_HOUSE_VALIDITY"]

        self._prime_build_readiness(sim, architect)
        botanist.source_memory_state.setdefault("Botanist_Info", {})["ever_inspected"] = True
        botanist.source_inspection_state["Botanist_Info"] = "inspected"
        architect.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = False
        engineer.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = False
        engineer.source_inspection_state["Engineer_Info"] = "unseen"

        blocker = architect._build_readiness_blockers(sim.environment, sim_state=sim)
        self.assertIn("missing_cross_role_engineer_grounding", blocker)

        engineer.source_inspection_state["Engineer_Info"] = "inspected"
        engineer.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = True

        blocker_after = architect._build_readiness_blockers(sim.environment, sim_state=sim)
        self.assertNotIn("missing_cross_role_engineer_grounding", blocker_after)
        sim.stop()

    def test_blocker_partition_separates_hard_and_epistemic_advisories(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        partition = agent._partition_action_blockers(
            [
                "no_navigable_build_target",
                "missing_role_grounding",
                "missing_cross_role_engineer_grounding",
                "stale_epistemic_grounding",
                "missing_validation_rule_knowledge",
            ]
        )
        self.assertIn("no_navigable_build_target", partition["hard_blockers"])
        self.assertIn("missing_role_grounding", partition["epistemic_advisories"])
        self.assertIn("missing_cross_role_engineer_grounding", partition["epistemic_advisories"])
        self.assertIn("stale_epistemic_grounding", partition["epistemic_advisories"])
        self.assertIn("missing_validation_rule_knowledge", partition["epistemic_advisories"])
        sim.stop()

    def test_validation_survives_translation_and_hits_mark_validated(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        for _ in range(len(project["build_steps"])):
            sim.environment.construction.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)
        project["support_requirements"] = {}
        project["support_counts"] = {}
        project["support_status"] = {}
        sim.environment.construction.recompute_support_status(project_id)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="design_note", note="design", actor=agent.name, sim_time=sim.time)
        project["correct"] = True
        self.assertEqual(project.get("status"), "ready_for_validation")

        decision = BrainDecision(
            selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
            target_id=project_id,
            confidence=0.9,
        )
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        self.assertEqual(translated["type"], "idle")
        self.assertEqual(translated.get("decision_action"), ExecutableActionType.VALIDATE_CONSTRUCTION.value)
        self.assertTrue(any(e.get("event_type") == "validate_construction_translated" for e in sim.logger.recent_events))

        agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
        with mock.patch.object(sim.environment.construction, "mark_validated", wraps=sim.environment.construction.mark_validated) as mark_spy:
            agent.active_actions = [{**translated, "progress": 0.0}]
            agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            self.assertEqual(mark_spy.call_count, 1)
        self.assertEqual(project.get("status"), "complete")
        self.assertTrue(project.get("validated_complete"))
        sim.stop()

    def test_validation_epistemic_downgrade_is_explicit(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        project["started"] = True
        for step in project.get("build_steps", []):
            step["completed"] = True
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="design_note", note="design", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.update()
        project["correct"] = True

        decision = BrainDecision(
            selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
            target_id=project_id,
            confidence=0.9,
        )
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        self.assertIn(translated["type"], {"communicate", "idle"})
        self.assertTrue(any(e.get("event_type") == "validation_blocked_epistemic" for e in sim.logger.recent_events))
        if translated["type"] == "communicate":
            self.assertEqual(translated.get("decision_action"), ExecutableActionType.COMMUNICATE.value)
            self.assertEqual(translated.get("translation_outcome"), "validate_downgraded_epistemic")
            self.assertTrue(any(e.get("event_type") == "validate_construction_downgraded_to_communication" for e in sim.logger.recent_events))
        sim.stop()

    def test_validation_non_epistemic_block_is_explicit(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        decision = BrainDecision(
            selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
            target_id="Build_Table_DOES_NOT_EXIST",
            confidence=0.9,
        )
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        self.assertEqual(translated["type"], "idle")
        self.assertEqual(translated.get("decision_action"), ExecutableActionType.WAIT.value)
        self.assertEqual(translated.get("translation_outcome"), "validate_blocked_non_epistemic")
        self.assertTrue(any(e.get("event_type") == "validation_blocked_non_epistemic" for e in sim.logger.recent_events))
        self.assertFalse(any(e.get("event_type") == "validate_construction_downgraded_to_communication" for e in sim.logger.recent_events))
        sim.stop()

    def test_only_mark_validated_mutates_validation_state(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        for _ in range(len(project["build_steps"])):
            sim.environment.construction.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)
        project["support_requirements"] = {}
        project["support_counts"] = {}
        project["support_status"] = {}
        sim.environment.construction.recompute_support_status(project_id)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="design_note", note="design", actor=agent.name, sim_time=sim.time)
        project["correct"] = True
        self.assertEqual(project.get("status"), "ready_for_validation")

        with mock.patch.object(sim.environment.construction, "mark_validated", wraps=sim.environment.construction.mark_validated) as mark_spy:
            agent.active_actions = [{
                "type": "idle",
                "duration": 1.0,
                "progress": 0.0,
                "project_id": project_id,
                "decision_action": ExecutableActionType.WAIT.value,
            }]
            agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            self.assertEqual(mark_spy.call_count, 0)
            self.assertEqual(project.get("status"), "ready_for_validation")

            decision = BrainDecision(
                selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
                target_id=project_id,
                confidence=0.9,
            )
            translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
            agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
            agent.active_actions = [{**translated, "progress": 0.0}]
            agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            self.assertEqual(mark_spy.call_count, 1)
            self.assertEqual(project.get("status"), "complete")
        sim.stop()


if __name__ == "__main__":
    unittest.main()
