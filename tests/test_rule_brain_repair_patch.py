import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.agent import Agent
from modules.brain_context import BrainContextPacket
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState


class _CtxBuilder:
    def build(self, _sim, _agent):
        return BrainContextPacket(
            static_task_context={"role": "Engineer"},
            world_snapshot={"sim_time": 1.0, "phase_profile": {"name": "execution"}, "built_state": []},
            individual_cognitive_state={
                "traits": {"communication_propensity": 1.0, "help_tendency": 1.0},
                "known_gaps": ["water_gap"],
                "build_readiness": {"ready_for_build": False},
                "goal_stack": [{"goal_id": "prepare_water"}],
                "loop_counters": {"action_repeats": 0, "selected_action_repeats": 0},
                "seconds_since_dik_change": 10.0,
                "control_state": {"mode": "BOOTSTRAP", "mode_dwell_steps": 1},
                "inspect_state": {"source_exhaustion": {"Team_Info": {"exhausted": True, "no_new_dik_streak": 3}}},
            },
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
            history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
            action_affordances=[
                {"action_type": "observe_environment", "utility": 0.6},
                {"action_type": "wait", "utility": 0.2},
            ],
        )


class _Logger:
    def log_event(self, *_args, **_kwargs):
        return None


class TestRuleBrainRepairPatch(unittest.TestCase):
    def test_execution_lock_enters_for_actionable_build_project(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            decision = BrainDecision(selected_action=ExecutableActionType.CONTINUE_CONSTRUCTION, target_id="Build_Table_A", confidence=0.8)
            with patch.object(agent, "_execution_lock_viable_project", return_value=True), patch.object(agent, "_apply_metacognitive_regulation", side_effect=lambda d, *_args, **_kwargs: d):
                rewritten = agent._apply_policy_pivots(decision, environment=sim.environment, sim_state=sim, context=None, pivot_origin="unit")
            self.assertEqual(rewritten.selected_action, ExecutableActionType.CONTINUE_CONSTRUCTION)
            self.assertTrue(agent.execution_lock_state.get("execution_lock_active"))
            self.assertEqual(agent.execution_lock_state.get("locked_project_id"), "Build_Table_A")
            self.assertTrue(any(e.get("event_type") == "execution_lock_started" for e in sim.logger.get_recent_events(200)))
        finally:
            sim.stop()

    def test_execution_lock_biases_away_from_transport_and_consult(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            agent.execution_lock_state.update({"execution_lock_active": True, "locked_project_id": "Build_Table_A", "lock_last_progress_tick": int(agent.sim_step_count)})
            project = sim.environment.construction.projects["Build_Table_A"]
            project["resource_complete"] = True
            project["status"] = "in_progress"
            transport = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="Build_Table_A", confidence=0.6)
            consult = BrainDecision(selected_action=ExecutableActionType.CONSULT_TEAM_ARTIFACT, target_id="whiteboard", confidence=0.6)
            biased_transport = agent._apply_execution_lock_bias(transport, sim.environment, sim_state=sim)
            with patch.object(agent, "_execution_lock_viable_project", return_value=True):
                biased_consult = agent._apply_execution_lock_bias(consult, sim.environment, sim_state=sim)
            self.assertEqual(biased_transport.selected_action, ExecutableActionType.CONTINUE_CONSTRUCTION)
            self.assertEqual(biased_consult.selected_action, ExecutableActionType.CONTINUE_CONSTRUCTION)
            events = [e.get("event_type") for e in sim.logger.get_recent_events(240)]
            self.assertIn("execution_lock_transport_suppressed", events)
            self.assertIn("execution_lock_switch_suppressed", events)
        finally:
            sim.stop()

    def test_execution_lock_releases_on_timeout_and_blocker(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            agent.execution_lock_state.update(
                {
                    "execution_lock_active": True,
                    "locked_project_id": "Build_Table_A",
                    "lock_started_tick": 0,
                    "lock_last_progress_tick": 0,
                }
            )
            agent.sim_step_count = 12
            agent._maintain_execution_lock(sim.environment, sim_state=sim)
            self.assertFalse(agent.execution_lock_state.get("execution_lock_active"))

            agent.execution_lock_state.update(
                {
                    "execution_lock_active": True,
                    "locked_project_id": "Build_Table_A",
                    "lock_started_tick": 1,
                    "lock_last_progress_tick": 1,
                }
            )
            with patch.object(agent, "_execution_lock_viable_project", return_value=False):
                agent._maintain_execution_lock(sim.environment, sim_state=sim)
            self.assertFalse(agent.execution_lock_state.get("execution_lock_active"))
            self.assertTrue(any(e.get("event_type") == "execution_lock_released" for e in sim.logger.get_recent_events(300)))
        finally:
            sim.stop()

    def test_metacognitive_breaks_stale_support_loop(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            agent.support_goal_nonexec_counts = {"support_a:stale": 4}
            agent.communication_state["no_effect_streak"] = 3
            decision = BrainDecision(
                selected_action=ExecutableActionType.CONSULT_TEAM_ARTIFACT,
                target_id="whiteboard",
                confidence=0.7,
            )
            with patch.object(agent, "_choose_post_inspect_followup_decision", return_value=BrainDecision(selected_action=ExecutableActionType.REASSESS_PLAN, confidence=0.8)):
                rewritten = agent._apply_policy_pivots(decision, environment=sim.environment, sim_state=sim, context=None, pivot_origin="unit")
            self.assertEqual(rewritten.selected_action, ExecutableActionType.REASSESS_PLAN)
            self.assertEqual(agent.metacognitive_state.get("mode"), "RECOVER_REGROUND")
            self.assertTrue(any(e.get("event_type") == "metacognitive_support_loop_broken" for e in sim.logger.get_recent_events(120)))
        finally:
            sim.stop()

    def test_metacognitive_readiness_unlock_starts_execution_window_and_biases_execution(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            agent.last_build_blockers = ["insufficient_rule_knowledge"]
            decision = BrainDecision(selected_action=ExecutableActionType.CONSULT_TEAM_ARTIFACT, confidence=0.6)
            candidate = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="Build_Table_A", confidence=0.9)
            with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_metacognitive_execution_candidate", return_value=candidate):
                rewritten = agent._apply_policy_pivots(decision, environment=sim.environment, sim_state=sim, context=None, pivot_origin="unit")
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            self.assertGreater(float(agent.metacognitive_state.get("execution_window_until", 0.0)), float(sim.time))
            self.assertTrue(agent.execution_lock_state.get("execution_lock_active"))
            events = [e.get("event_type") for e in sim.logger.get_recent_events(120)]
            self.assertIn("metacognitive_execution_window_started", events)
            self.assertIn("metacognitive_switch_to_execution", events)
        finally:
            sim.stop()

    def test_metacognitive_defers_blocked_project_and_reactivates_on_new_support(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            blocked_project = "Build_Table_A"
            alt_candidate = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="Build_Table_B", confidence=0.8)
            decision = BrainDecision(selected_action=ExecutableActionType.START_CONSTRUCTION, target_id=blocked_project, confidence=0.7)
            with patch.object(agent, "_metacognitive_execution_candidate", return_value=alt_candidate), patch.object(agent, "_construction_action_blockers", return_value=(["missing_validation_rule_knowledge"], blocked_project)):
                rewritten = agent._apply_policy_pivots(decision, environment=sim.environment, sim_state=sim, context=None, pivot_origin="unit")
            self.assertEqual(rewritten.target_id, "Build_Table_B")
            self.assertIn(blocked_project, agent.metacognitive_state.get("deferred_projects", {}))
            self.assertTrue(any(e.get("event_type") == "metacognitive_project_deferred" for e in sim.logger.get_recent_events(120)))

            agent.metacognitive_state["deferred_projects"] = {blocked_project: {"deferred_at": sim.time, "reason": "project_blocked"}}
            with patch.object(agent, "_metacognitive_execution_candidate", side_effect=[BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=blocked_project, confidence=0.9), BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=blocked_project, confidence=0.9)]):
                rewritten_2 = agent._apply_policy_pivots(
                    BrainDecision(selected_action=ExecutableActionType.WAIT, confidence=0.4),
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="unit",
                )
            self.assertEqual(rewritten_2.target_id, blocked_project)
            self.assertNotIn(blocked_project, agent.metacognitive_state.get("deferred_projects", {}))
            self.assertTrue(any(e.get("event_type") == "metacognitive_project_reactivated" for e in sim.logger.get_recent_events(240)))
        finally:
            sim.stop()

    def test_metacognitive_suppresses_wait_when_viable_execution_exists(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            candidate = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="Build_Table_A", confidence=0.9)
            with patch.object(agent, "_metacognitive_execution_candidate", return_value=candidate):
                rewritten = agent._apply_policy_pivots(
                    BrainDecision(selected_action=ExecutableActionType.WAIT, confidence=0.2),
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="unit",
                )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            self.assertTrue(any(e.get("event_type") == "metacognitive_wait_suppressed_due_to_viable_action" for e in sim.logger.get_recent_events(120)))
        finally:
            sim.stop()

    def test_execution_lock_progress_updates_and_no_legality_bypass(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            agent.execution_lock_state.update(
                {
                    "execution_lock_active": True,
                    "locked_project_id": "Build_Table_A",
                    "lock_started_tick": 1,
                    "lock_last_progress_tick": 1,
                }
            )
            agent.sim_step_count = 5
            agent._register_progress(sim_state=sim, kind="construction_progress_updated", detail={"project_id": "Build_Table_A"})
            self.assertEqual(int(agent.execution_lock_state.get("lock_last_progress_tick", -1)), 5)
            blocked = BrainDecision(selected_action=ExecutableActionType.CONTINUE_CONSTRUCTION, target_id="Build_Table_A", confidence=0.7)
            with patch.object(agent, "_construction_action_blockers", return_value=(["missing_project_binding"], "Build_Table_A")):
                translated = agent._translate_brain_decision_to_legacy_action(blocked, sim.environment, sim_state=sim)
            self.assertTrue(any(a.get("type") == "idle" for a in translated))
        finally:
            sim.stop()

    def test_build_brain_request_includes_compact_control_state_fields(self):
        agent = Agent("Engineer", "Engineer")
        agent.control_state.update(
            {
                "mode": "LOGISTICS",
                "previous_mode": "COORDINATE",
                "mode_dwell_steps": 4,
                "last_transition_reason": "build_ready_incomplete_projects_bias_logistics",
                "last_policy_snapshot": {"top_features": {"build_opportunity": 1.0}},
            }
        )
        sim = SimpleNamespace(
            time=1.0,
            bootstrap_reuse_enabled=False,
            get_agent_brain_runtime=lambda _a: {},
            task_model=SimpleNamespace(task_id="mars_colony"),
        )
        context = _CtxBuilder().build(sim, agent)
        request = agent._build_brain_request(sim, context, request_explanation=False, trigger_reason="unit_test")
        self.assertEqual(request.control_mode, "LOGISTICS")
        self.assertEqual(request.previous_control_mode, "COORDINATE")
        self.assertEqual(request.mode_dwell_steps, 4)
        self.assertEqual(request.last_transition_reason, "build_ready_incomplete_projects_bias_logistics")
        self.assertEqual(request.control_state_snapshot.get("mode"), "LOGISTICS")

    def test_role_source_preferred_after_team_exhaustion(self):
        agent = Agent("Engineer", "Engineer")
        agent.goal_stack = [{"goal": "secure water connectivity", "status": "active"}]
        env = SimpleNamespace(
            knowledge_packets={"Team_Info": {}, "Engineer_Info": {}, "Botanist_Info": {}},
            get_interaction_target_position=lambda source_id, from_position=None: (1.0, 1.0),
            construction=SimpleNamespace(projects={}),
            interaction_targets={},
        )
        agent.source_inspection_state = {"Team_Info": "inspected", "Engineer_Info": "unseen", "Botanist_Info": "unseen"}
        agent.source_exhaustion_state = {
            "Team_Info": {"exhausted": True, "no_new_dik_streak": 3, "inspect_count": 2},
            "Engineer_Info": {"exhausted": False, "no_new_dik_streak": 0, "inspect_count": 0},
            "Botanist_Info": {"exhausted": False, "no_new_dik_streak": 0, "inspect_count": 0},
        }
        ranked = agent._candidate_information_sources(env)
        self.assertEqual(ranked[0][1], "Engineer_Info")

    def test_local_policy_refresh_for_rule_brain_when_planner_not_due(self):
        agent = Agent("Engineer", "Engineer")
        sim = SimpleNamespace(
            configured_brain_backend="rule_brain",
            brain_provider=RuleBrain(),
            brain_context_builder=_CtxBuilder(),
            get_agent_brain_runtime=lambda _a: {"provider": RuleBrain(), "configured_backend": "rule_brain"},
            logger=_Logger(),
            time=1.0,
        )
        env = SimpleNamespace(
            knowledge_packets={},
            construction=SimpleNamespace(projects={}),
            interaction_targets={},
            get_interaction_target_position=lambda *_args, **_kwargs: None,
        )
        refreshed = agent._attempt_local_rule_brain_refresh(sim, env, "split_mode_cadence_not_due")
        self.assertTrue(refreshed)
        self.assertTrue(agent.current_action)
        self.assertIsInstance(agent.control_state.get("method_state"), dict)
        self.assertIn("active_method_id", agent.control_state.get("method_state", {}))

    def test_policy_pivot_is_rerouted_through_rulebrain_controller(self):
        agent = Agent("Engineer", "Engineer")
        context = BrainContextPacket(
            static_task_context={"role": "Engineer"},
            world_snapshot={"sim_time": 1.0, "phase_profile": {"name": "execution"}, "built_state": []},
            individual_cognitive_state={
                "traits": {"communication_propensity": 1.0, "help_tendency": 1.0},
                "known_gaps": ["water_gap"],
                "build_readiness": {"ready_for_build": False},
                "goal_stack": [{"goal_id": "prepare_water"}],
                "loop_counters": {"action_repeats": 0, "selected_action_repeats": 0},
                "seconds_since_dik_change": 10.0,
                "control_state": {
                    "mode": "ACQUIRE_DIK",
                    "mode_dwell_steps": 2,
                    "method_state": {
                        "active_method_id": "AcquireRoleSpecificGrounding",
                        "active_method_step": "move_to_role_source",
                        "source_cooldowns": {"Team_Info": 30},
                    },
                },
                "inspect_state": {"source_exhaustion": {"Team_Info": {"exhausted": True, "no_new_dik_streak": 3}}},
            },
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
            history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
            action_affordances=[
                {"action_type": "inspect_information_source", "target_id": "Team_Info", "utility": 0.99},
                {"action_type": "inspect_information_source", "target_id": "Engineer_Info", "utility": 0.55},
                {"action_type": "communicate", "target_id": "nearby_agent", "utility": 0.25},
            ],
        )
        provider = RuleBrain()
        provider.policy_config = provider.policy_config.__class__(min_mode_dwell_steps=4)
        sim = SimpleNamespace(
            configured_brain_backend="rule_brain",
            brain_provider=provider,
            brain_context_builder=SimpleNamespace(build=lambda *_args, **_kwargs: context),
            get_agent_brain_runtime=lambda _a: {"provider": provider, "configured_backend": "rule_brain"},
            logger=_Logger(),
            time=1.0,
        )
        agent.post_inspect_handoff = {
            "pending": True,
            "source_id": "Team_Info",
            "dik_changed": False,
            "readiness_changed": False,
            "blockers": ["insufficient_rule_knowledge"],
            "blocker_category": "missing_rule",
            "outcome": "inspect_success_no_new_dik",
            "expires_at": 99.0,
        }
        fallback_decision = BrainDecision(
            selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
            target_id="Team_Info",
            reason_summary="legacy path requested Team_Info",
            confidence=0.5,
        )
        with patch.object(agent, "_choose_post_inspect_followup_decision", return_value=BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, reason_summary="compat fallback")):
            rewritten = agent._apply_policy_pivots(
                fallback_decision,
                environment=SimpleNamespace(
                    knowledge_packets={},
                    construction=SimpleNamespace(projects={}),
                    interaction_targets={},
                    get_interaction_target_position=lambda *_args, **_kwargs: None,
                ),
                sim_state=sim,
                context=context,
                pivot_origin="unit",
        )
        self.assertEqual(rewritten.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
        self.assertIn(rewritten.target_id, {"Engineer_Info", "Team_Info"})

    def test_runtime_snapshot_uses_control_state_method_state(self):
        agent = Agent("Engineer", "Engineer")
        agent.control_state["mode"] = "ACQUIRE_DIK"
        agent.control_state["method_state"] = {
            "active_method_id": "AcquireRoleSpecificGrounding",
            "active_method_step": "inspect_role_source",
            "step_retry_count": 1,
            "source_cooldowns": {"Team_Info": 42},
        }
        agent._sync_method_state_from_control()
        snapshot = agent.get_runtime_state_snapshot()
        self.assertEqual(snapshot.get("control_state", {}).get("mode"), "ACQUIRE_DIK")
        self.assertEqual(snapshot.get("method_state", {}).get("active_method_id"), "AcquireRoleSpecificGrounding")
        self.assertEqual(snapshot.get("method_state", {}).get("active_method_step"), "inspect_role_source")

    def test_default_agent_traits_are_ideal_baseline(self):
        sim = SimulationState(speed=1.0)
        try:
            self.assertTrue(sim.agents)
            for agent in sim.agents:
                self.assertEqual(agent.communication_propensity, 1.0)
                self.assertEqual(agent.goal_alignment, 1.0)
                self.assertEqual(agent.help_tendency, 1.0)
                self.assertEqual(agent.build_speed, 1.0)
                self.assertEqual(agent.rule_accuracy, 1.0)
        finally:
            if hasattr(sim, "planner_executor"):
                sim.planner_executor.shutdown(wait=False)

    def test_policy_pivot_keeps_closure_commitment_hard_override(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        sim.environment.construction.projects[project_id]["status"] = "ready_for_validation"
        agent.project_closure_state.update(
            {
                "active": True,
                "project_id": project_id,
                "commit_until": float(sim.time) + 20.0,
            }
        )
        with patch.object(agent, "_construction_action_blockers", return_value=([], project_id)):
            rewritten = agent._apply_policy_pivots(
                BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, reason_summary="test", confidence=0.4),
                environment=sim.environment,
                sim_state=sim,
                context=None,
                pivot_origin="unit",
            )
        self.assertEqual(rewritten.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        self.assertEqual(rewritten.target_id, project_id)
        sim.stop()

    def test_policy_pivot_demotes_stale_grounding_to_preference_without_local_evidence(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        sim.environment.construction.projects[project_id]["status"] = "ready_for_validation"
        context = BrainContextPacket(
            static_task_context={"role": agent.role},
            world_snapshot={"sim_time": sim.time, "built_state": []},
            individual_cognitive_state={"epistemic_sufficiency": {"refresh_pressure": 0.9, "role_missing": True}},
            team_state={},
            history_bands={},
            action_affordances=[],
        )
        fallback = BrainDecision(
            selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
            target_id=f"{agent.role}_Info",
            reason_summary="test inspect",
            confidence=0.9,
        )
        with patch.object(agent, "_choose_post_inspect_followup_decision", return_value=fallback):
            rewritten = agent._apply_policy_pivots(
                BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.8),
                environment=sim.environment,
                sim_state=sim,
                context=context,
                pivot_origin="unit",
            )
        self.assertEqual(rewritten.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        self.assertTrue(any(e.get("event_type") == "policy_preference_applied" for e in sim.logger.recent_events))
        sim.stop()

    def test_inspect_pursuit_abandon_before_start_clears_duplicate_latch_state(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[1]  # Engineer
            source_id = "Engineer_Info"
            target = sim.environment.get_interaction_target_position(source_id, from_position=agent.position)
            self.assertIsNotNone(target)
            now_ts = float(sim.time)
            agent.current_inspect_target_id = source_id
            agent.inspect_session = {
                "source_id": source_id,
                "target": target,
                "state": "target_selected",
                "started_at": now_ts,
                "last_updated_at": now_ts,
                "restarts": 1,
            }
            agent.inspect_pursuit = {
                "action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                "source_id": source_id,
                "slot_id": None,
                "target_position": target,
                "started_at": now_ts,
                "expires_at": now_ts + 4.0,
                "blocked_attempts": 0,
                "no_progress_ticks": 0,
                "last_distance_to_target": None,
            }
            agent.source_inspection_state[source_id] = "in_progress"
            agent.active_intent.update(
                {
                    "intent_id": f"{ExecutableActionType.INSPECT_INFORMATION_SOURCE.value}:{source_id}",
                    "target": source_id,
                    "min_commit_until": now_ts + 5.0,
                }
            )
            agent._clear_inspect_pursuit(reason="unit_stall", sim_state=sim, release_slot=True, environment=sim.environment)
            self.assertIsNone(agent.current_inspect_target_id)
            self.assertEqual(agent.inspect_session.get("state"), "idle")
            self.assertEqual(agent.inspect_session.get("source_id"), None)
            self.assertEqual(agent.source_inspection_state.get(source_id), "unseen")
            self.assertIsNone(agent.active_intent.get("intent_id"))
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id=source_id,
                reason_summary="fresh retry",
                confidence=0.8,
            )
            agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            events = [e["event_type"] for e in sim.logger.get_recent_events(120)]
            self.assertIn("inspect_target_selected", events)
            self.assertNotIn("inspect_restarted_duplicate", events[-20:])
        finally:
            sim.stop()

    def test_repeated_stalled_inspect_still_allows_later_inspect_started(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[1]  # Engineer
            source_id = "Engineer_Info"
            target = sim.environment.get_interaction_target_position(source_id, from_position=agent.position)
            self.assertIsNotNone(target)
            now_ts = float(sim.time)
            for _ in range(2):
                agent.inspect_session = {
                    "source_id": source_id,
                    "target": target,
                    "state": "target_selected",
                    "started_at": now_ts,
                    "last_updated_at": now_ts,
                    "restarts": 0,
                }
                agent.inspect_pursuit = {
                    "action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                    "source_id": source_id,
                    "slot_id": None,
                    "target_position": target,
                    "started_at": now_ts,
                    "expires_at": now_ts + 8.0,
                    "blocked_attempts": agent.inspect_pursuit_blocked_attempt_limit,
                    "no_progress_ticks": 0,
                    "last_distance_to_target": None,
                }
                agent.current_inspect_target_id = source_id
                agent._inspect_source(sim.environment, source_id, sim_state=sim)
            agent.position = target
            agent.current_inspect_target_id = source_id
            with patch("modules.agent.random.random", return_value=0.0):
                changed = agent._inspect_source(sim.environment, source_id, sim_state=sim)
            self.assertTrue(changed)
            events = [e["event_type"] for e in sim.logger.get_recent_events(240)]
            self.assertIn("inspect_started", events)
        finally:
            sim.stop()

    def test_active_closure_suppresses_unrelated_artifact_consult_without_side_effects(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            owner = sim.agents[0]
            teammate = sim.agents[1]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["status"] = "ready_for_validation"
            project["closure_owner"] = owner.name
            project["closure_status"] = "in_progress"
            before_info = len(teammate.mental_model["information"])
            decision = BrainDecision(
                selected_action=ExecutableActionType.CONSULT_TEAM_ARTIFACT,
                target_id="whiteboard",
                reason_summary="unit test",
                confidence=0.7,
            )
            with patch.object(teammate, "_can_attempt_verbal_plan_communication", return_value=True):
                rewritten = teammate._apply_policy_pivots(
                    decision,
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="unit",
                )
            self.assertIn(rewritten.selected_action, {ExecutableActionType.COMMUNICATE, ExecutableActionType.WAIT})
            self.assertEqual(project.get("closure_owner"), owner.name)
            self.assertEqual(project.get("status"), "ready_for_validation")
            self.assertEqual(len(teammate.mental_model["information"]), before_info)
            self.assertNotEqual(rewritten.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        finally:
            sim.stop()

    def test_active_closure_suppresses_generic_team_info_inspect_with_communication_first(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            owner = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["status"] = "ready_for_validation"
            project["expected_rules"] = ["R_BLOCKED"]
            sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
            owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
            owner.progress_tracker["forced_pivot"] = ""
            owner.progress_tracker["forced_pivot_until"] = 0.0
            owner.project_closure_state["repair_mode"] = True
            owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_BLOCKED|R_BLOCKED"
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                confidence=0.7,
            )
            with patch.object(owner, "_construction_action_blockers", return_value=(["missing_expected_rule:R_BLOCKED", "missing_validation_rule_knowledge"], project_id)), patch.object(owner, "_critical_unmet_source_targets", return_value={"Engineer_Info": 1}), patch.object(owner, "_can_attempt_verbal_plan_communication", return_value=True):
                rewritten = owner._apply_policy_pivots(
                    decision,
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="unit",
                )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.COMMUNICATE)
            self.assertNotEqual(rewritten.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
        finally:
            sim.stop()

    def test_active_closure_preserves_blocker_relevant_inspect_target(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            owner = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["status"] = "ready_for_validation"
            project["expected_rules"] = ["R_BLOCKED"]
            sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
            owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
            owner.progress_tracker["forced_pivot"] = ""
            owner.progress_tracker["forced_pivot_until"] = 0.0
            owner.project_closure_state["repair_mode"] = True
            owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_BLOCKED|R_BLOCKED"
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Architect_Info",
                confidence=0.7,
            )
            with patch.object(owner, "_construction_action_blockers", return_value=(["missing_expected_rule:R_BLOCKED", "missing_validation_rule_knowledge"], project_id)), patch.object(owner, "_critical_unmet_source_targets", return_value={"Architect_Info": 1}):
                rewritten = owner._apply_policy_pivots(
                    decision,
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="unit",
                )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
            self.assertEqual(rewritten.target_id, "Architect_Info")
        finally:
            sim.stop()

    def test_stalled_closure_suppresses_repeated_non_relevant_inspect_without_comm(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            owner = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["status"] = "ready_for_validation"
            project["expected_rules"] = ["R_STALLED"]
            sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
            owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
            owner.progress_tracker["forced_pivot"] = ""
            owner.progress_tracker["forced_pivot_until"] = 0.0
            owner.project_closure_state["repair_mode"] = True
            owner.project_closure_state["repair_unchanged_count"] = 3
            owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_STALLED|R_STALLED"
            owner.source_exhaustion_state.setdefault("Team_Info", {})["exhausted"] = True
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                confidence=0.7,
            )
            with patch.object(owner, "_construction_action_blockers", return_value=(["missing_expected_rule:R_STALLED", "missing_validation_rule_knowledge"], project_id)), patch.object(owner, "_critical_unmet_source_targets", return_value={"Team_Info": 1}), patch.object(owner, "_can_attempt_verbal_plan_communication", return_value=False):
                rewritten = owner._apply_policy_pivots(
                    decision,
                    environment=sim.environment,
                    sim_state=sim,
                    context=None,
                    pivot_origin="local_refresh",
                )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.REASSESS_PLAN)
        finally:
            sim.stop()

    def test_non_closure_inspect_behavior_remains_available(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                confidence=0.6,
            )
            rewritten = agent._apply_policy_pivots(
                decision,
                environment=sim.environment,
                sim_state=sim,
                context=None,
                pivot_origin="unit",
            )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
