"""A consult is useful only while it can teach something; a useful consult is never replaced before it runs."""

import tempfile
from unittest.mock import patch

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.knowledge import Information
from modules.simulation import SimulationState

CONSULT = ExecutableActionType.CONSULT_TEAM_ARTIFACT


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="consult", seed=3)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _artifact(sim, pid="site_b_house_001"):
    cm = sim.environment.construction
    cm.create_project("site_b", structure_type="house", project_id_override=pid)
    sim.team_knowledge_manager.artifacts.clear()
    return sim.team_knowledge_manager.upsert_construction_artifact(cm.projects[pid], 1.0)


def _stagnate(agent):
    agent.support_goal_nonexec_counts = {"SUPPORT_X": 3}  # the state in which the regrounding fallback fires


def _regulate(sim, agent):
    decision = BrainDecision(selected_action=CONSULT, target_id="whiteboard", confidence=0.8)
    return decision, agent._apply_metacognitive_regulation(decision, sim.environment, sim_state=sim)


def _consult_once(sim, agent, art):
    agent._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim, dik_gained=0, useful=True)


def test_1_first_useful_consult_stays_attractive_and_is_not_replaced(sim):
    agent = sim.agents[0]
    art = _artifact(sim)
    assessment = agent._consult_assessment(sim.environment, sim_state=sim)
    assert assessment["useful"] and "unseen" in assessment["reasons"]
    _stagnate(agent)
    decision, result = _regulate(sim, agent)
    assert result is decision and result.selected_action == CONSULT  # the Pilot's useful consult survives the stagnation fallback
    assert any(n == "metacognitive_regrounding_deferred_useful_consult" for n, _ in sim.captured_events)
    assert not any(n == "metacognitive_support_loop_broken" for n, _ in sim.captured_events)


def test_2_unchanged_already_known_artifact_becomes_redundant(sim):
    agent = sim.agents[0]
    art = _artifact(sim)
    _consult_once(sim, agent, art)
    assessment = agent._consult_assessment(sim.environment, sim_state=sim)
    assert assessment["useful"] is False and assessment["suppression_reason"] == "same_version_no_new_context"
    consult = [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a["action_type"] == CONSULT.value]
    assert consult and all(a.get("consult_redundant") for a in consult)
    _stagnate(agent)
    decision, result = _regulate(sim, agent)
    assert result.selected_action != CONSULT  # a consult that would teach nothing is still subject to the loop-breaking fallback


def test_3_artifact_revision_makes_consultation_useful_again(sim):
    agent = sim.agents[0]
    art = _artifact(sim)
    _consult_once(sim, agent, art)
    assert not agent._consult_assessment(sim.environment, sim_state=sim)["useful"]
    project = sim.environment.construction.projects[art.content["project_id"]]
    project["delivered_resources"]["bricks"] = 3  # the project advanced: the artifact summary changes
    sim.team_knowledge_manager.upsert_construction_artifact(project, 5.0)
    assessment = agent._consult_assessment(sim.environment, sim_state=sim)
    assert assessment["useful"] and "artifact_changed_since_consult" in assessment["reasons"]


def test_4_forgetting_relevant_dik_makes_reconsultation_useful_again(sim):
    agent = sim.agents[0]
    art = _artifact(sim)
    info = Information("I_TEST_ELEMENT", "test content")
    art.carried_information = [info]
    agent.mental_model["information"].add(info)
    _consult_once(sim, agent, art)
    assert not agent._consult_assessment(sim.environment, sim_state=sim)["useful"]  # retained: nothing to gain
    agent.mental_model["information"].discard(info)  # forgotten
    assessment = agent._consult_assessment(sim.environment, sim_state=sim)
    assert assessment["useful"] and "carries_unheld_dik" in assessment["reasons"] and assessment["carries_unheld_dik"] == ["I_TEST_ELEMENT"]


def test_5_new_blocker_or_goal_state_reopens_consultation(sim):
    agent = sim.agents[0]
    art = _artifact(sim)
    _consult_once(sim, agent, art)
    base = agent._artifact_coordination_context_signature(sim.environment, sim_state=sim)
    with patch.object(agent, "_artifact_coordination_context_signature", return_value=dict(base, blocker_category="missing_rule_knowledge")):
        assessment = agent._consult_assessment(sim.environment, sim_state=sim)
    assert assessment["useful"] and "blocker_context_changed" in assessment["reasons"]
    agent.goal = "A different goal"
    assert "goal_context_changed" in agent._consult_assessment(sim.environment, sim_state=sim)["reasons"]


def test_6_no_substitution_or_forced_task_action(sim):
    agent = sim.agents[0]
    _artifact(sim)
    before = {pid: dict(p) for pid, p in sim.environment.construction.projects.items()}
    _stagnate(agent)
    decision, result = _regulate(sim, agent)
    assert result is decision and decision.target_id == "whiteboard" and decision.selected_action == CONSULT
    assert {pid: dict(p) for pid, p in sim.environment.construction.projects.items()} == before  # nothing in the world was touched
    assert not any(n in {"recovery_pivot_applied", "planner_action_downgraded", "mecha_project_substitution_rejected"} for n, _ in sim.captured_events)


def test_7_seeded_run_executes_its_consults_and_episodes_are_counted():
    from modules.agent import Agent

    selections = []
    original = Agent._select_decision_through_pilot_adapter

    def watch(self, sim_state, context, trigger_reason, runtime=None):
        decision = original(self, sim_state, context, trigger_reason, runtime)
        if decision.selected_action == CONSULT:
            selections.append(self.name)
        return decision

    Agent._select_decision_through_pilot_adapter = watch
    try:
        with tempfile.TemporaryDirectory() as tmp:
            state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="consult_loop", seed=2,
                                    experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}})
            started, executed = [], []
            state.logger.register_event_listener(lambda e: started.append(e["payload_data"]["episode_id"]) if e["event_type"] == "pilot_intent_started" else (executed.append(1) if e["event_type"] == "artifact_consulted" else None))
            try:
                for _ in range(300):
                    state.update(1.0)
            finally:
                state.stop()
    finally:
        Agent._select_decision_through_pilot_adapter = original
    stats = [a.pilot_episode_stats for a in state.agents]
    assert sum(s["episodes"] for s in stats) == len(started) == len(set(started))  # every episode counted once
    assert sum(s["selections"] for s in stats) == sum(s["episodes"] + s["continued"] for s in stats)
    assert len(executed) >= 0.5 * len(selections), (len(executed), len(selections))  # pre-fix: ~6 executed of ~300 selected