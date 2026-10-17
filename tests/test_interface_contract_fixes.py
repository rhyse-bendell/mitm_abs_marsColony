"""Interface/measurement fixes found by the overnight batch analysis:
VALIDATE affordance contract, cross-role private-source inspection, session-adaptive token estimator, core-vs-connector outcome breakdown."""
import copy
import json
import tempfile

import pytest

from modules import ollama_pilot
from modules.action_gate import AgentActionGate
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction import ConstructionManager
from modules.controller_context import partition_affordances
from modules.ollama_prompt import build_prompt
from modules.simulation import SimulationState
from modules.support_accounting import validated_structure_breakdown
from modules.token_estimator import TokenEstimator
from tests.test_ollama_controller import FakeOllama, OLLAMA_A, _run, _sim, _stop  # noqa: F401


@pytest.fixture(autouse=True)
def _fake_service(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t))


def _ready_house(cm, project_id, supported):
    cm.create_project("site_b", structure_type="house", project_id_override=project_id)
    p = cm.projects[project_id]
    p["started"] = True
    p["delivered_resources"]["bricks"] = p["required_resources"]["bricks"]
    p["resource_complete"] = True
    p["structurally_complete"] = True
    p["staged_resources"] = {"bricks": p["required_resources"]["bricks"]}
    for step in p["build_steps"]:
        step["completed"] = True
    p["epistemic_workspace"]["entries"] = [{"entry_type": "claim"}, {"entry_type": "evidence"}, {"entry_type": "design_note"}]
    if supported:
        p["functional_support_complete"] = True
        p["support_requirements"] = {}
    else:
        p["support_requirements"] = {"water": 1, "food": 1}
        p["support_counts"] = {}
        p["functional_support_complete"] = False
    cm.update()
    return project_id


@pytest.fixture
def procedural_sim():
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="contract", seed=3)
        try:
            yield st
        finally:
            st.stop()


def _aff(st, agent, kind, pid):
    return [a for a in st.brain_context_builder.build(st, agent).action_affordances if a["action_type"] == kind and a.get("target_id") == pid]


# ------------------------------------------------------------------ 1. VALIDATE affordance contract
def test_unsupported_validation_is_blocked_and_supported_is_executable(procedural_sim):
    st = procedural_sim
    cm = st.environment.construction
    bad = _ready_house(cm, "site_b_house_001", supported=False)
    good = _ready_house(cm, "site_b_house_002", supported=True)
    agent = st.agents[0]
    a_bad = _aff(st, agent, "validate_construction", bad)
    a_good = _aff(st, agent, "validate_construction", good)
    assert a_bad and a_bad[0]["validation_ready"] is False and "missing_functional_support" in a_bad[0]["validation_blockers"]
    assert a_good and a_good[0]["validation_ready"] is True and a_good[0]["validation_blockers"] == []
    packet = st.brain_context_builder.build(st, agent)
    ex, bl = partition_affordances(packet.action_affordances)
    assert not any(a["action_type"] == "validate_construction" and a["target_id"] == bad for a in ex)
    assert any(a["action_type"] == "validate_construction" and a["target_id"] == good for a in ex)
    blocked = [b for b in bl if b["action_type"] == "validate_construction"]
    assert [b["target_id"] for b in blocked] == [bad] and "missing_functional_support" in blocked[0]["blockers"]
    # the Procedural Baseline still receives the unchanged affordance list (both validations are present for it)
    assert a_bad and a_good


def test_blocked_validation_has_no_affordance_id_in_the_prompt(procedural_sim):
    st = procedural_sim
    cm = st.environment.construction
    bad = _ready_house(cm, "site_b_house_001", supported=False)
    good = _ready_house(cm, "site_b_house_002", supported=True)
    packet = st.brain_context_builder.build(st, st.agents[0])
    ex, bl = partition_affordances(packet.action_affordances)
    canon = {"identity": {"agent_id": "a", "role": "Engineer"}, "executable_actions": ex, "blocked_actions": bl, "last_rejection": None, "observations": {}, "recent_outcomes": {}}
    built = build_prompt(canon, budget_tokens=8000)
    offered = {(a["action_type"], a.get("target_id")) for a in built["affordance_map"].values()}
    assert ("validate_construction", good) in offered and ("validate_construction", bad) not in offered
    body = json.loads(built["user"].split("CONTEXT:\n", 1)[1].split("\n\nChoose", 1)[0])
    row = next(b for b in body["blocked_actions"] if b["target_id"] == bad)
    assert row["action_type"] == "validate_construction" and "affordance_id" not in row and "missing_functional_support" in row["blockers"]


def test_affordance_readiness_agrees_with_the_commit_gate_and_is_read_only(procedural_sim):
    st = procedural_sim
    cm = st.environment.construction
    bad = _ready_house(cm, "site_b_house_001", supported=False)
    good = _ready_house(cm, "site_b_house_002", supported=True)
    gate = AgentActionGate()
    for pid in (bad, good):
        verdict = gate.evaluate(agent=st.agents[0], decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=pid), sim_state=st)
        aff = _aff(st, st.agents[0], "validate_construction", pid)[0]
        assert verdict.legal == aff["validation_ready"]
        if not verdict.legal:
            assert set(aff["validation_blockers"]) == set(verdict.blockers)
    st.brain_context_builder.build(st, st.agents[0])  # settle the derived support cache, then prove the query itself changes nothing
    before = copy.deepcopy(cm.projects)
    st.brain_context_builder.build(st, st.agents[0])
    assert cm.projects == before


def test_execution_time_rejection_remains_legitimate(procedural_sim):
    st = procedural_sim
    cm = st.environment.construction
    pid = _ready_house(cm, "site_b_house_001", supported=True)
    assert _aff(st, st.agents[0], "validate_construction", pid)[0]["validation_ready"] is True
    cm.projects[pid]["support_requirements"] = {"water": 1}  # state changes after the offer
    cm.projects[pid]["support_counts"] = {}
    verdict = AgentActionGate().evaluate(agent=st.agents[0], decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=pid), sim_state=st)
    assert verdict.legal is False and "missing_functional_support" in verdict.blockers


# ------------------------------------------------------------------ 2. cross-role private-source inspection
@pytest.fixture
def mixed_sim():
    st = _sim({"Engineer": OLLAMA_A})
    yield st
    _stop(st)


def _agent(st, role):
    return next(a for a in st.agents if a.role == role)


def test_private_source_of_another_role_is_blocked_for_the_generative_pilot(mixed_sim):
    st = mixed_sim
    eng, arch = _agent(st, "Engineer"), _agent(st, "Architect")
    assert eng.is_generative_pilot() and not arch.is_generative_pilot()
    env = st.environment
    assert eng.source_access_blocker("Architect_Info", env) == "private_source_inaccessible"
    assert eng.source_access_blocker("Botanist_Info", env) == "private_source_inaccessible"
    assert eng.source_access_blocker("Engineer_Info", env) is None and eng.source_access_blocker("Team_Info", env) is None
    packet = st.brain_context_builder.build(st, eng)
    ex, bl = partition_affordances(packet.action_affordances)
    ex_inspect = {a["target_id"] for a in ex if a["action_type"] == "inspect_information_source"}
    bl_inspect = {a["target_id"]: a["blockers"] for a in bl if a["action_type"] == "inspect_information_source"}
    assert {"Engineer_Info", "Team_Info"} & ex_inspect and not ({"Architect_Info", "Botanist_Info"} & ex_inspect)
    assert bl_inspect and all(b == ["private_source_inaccessible"] for b in bl_inspect.values())


def test_generative_inspect_of_private_source_is_refused_without_substitution(mixed_sim):
    st = mixed_sim
    eng = _agent(st, "Engineer")
    events = []
    st.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    decision = BrainDecision(ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="Architect_Info", confidence=0.9)
    verdict = st.action_gate.evaluate(agent=eng, decision=decision, sim_state=st)
    assert verdict.legal is False and verdict.blockers == ["private_source_inaccessible"]
    actions = eng._translate_brain_decision_to_legacy_action(decision, st.environment, sim_state=st)
    assert [a["type"] for a in actions] == ["idle"] and actions[0]["blockers"] == ["private_source_inaccessible"]
    names = [e[0] for e in events]
    assert "source_retargeted_due_to_role_private_mismatch" not in names and "mecha_action_blocked" in names
    assert not any(a.get("decision_action") in {"communicate", "request_assistance"} for a in actions)  # no automatic alternative either


def test_shared_and_own_sources_stay_inspectable_for_the_generative_pilot(mixed_sim):
    st = mixed_sim
    eng = _agent(st, "Engineer")
    for src in ("Team_Info", "Engineer_Info"):
        verdict = st.action_gate.evaluate(agent=eng, decision=BrainDecision(ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id=src, confidence=0.9), sim_state=st)
        assert verdict.legal is True, src


def test_procedural_pilot_keeps_its_documented_retargeting(mixed_sim):
    st = mixed_sim
    arch = _agent(st, "Architect")
    events = []
    st.logger.register_event_listener(lambda e: events.append(e["event_type"]))
    decision = BrainDecision(ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="Botanist_Info", confidence=0.9)
    assert st.action_gate.evaluate(agent=arch, decision=decision, sim_state=st).legal is True  # gate unchanged for the Procedural Baseline
    arch._translate_brain_decision_to_legacy_action(decision, st.environment, sim_state=st)
    assert "mecha_action_blocked" not in events
    aff = next(a for a in st.brain_context_builder.build(st, arch).action_affordances if a["action_type"] == "inspect_information_source" and a["target_id"] == "Botanist_Info")
    assert aff["source_accessible"] is False  # the fact is stated, the Procedural Pilot's list is otherwise unchanged


# ------------------------------------------------------------------ 3. token estimator
def test_estimator_starts_conservative_and_adapts_with_a_margin():
    e = TokenEstimator("m")
    assert e.chars_per_token() == 2.8 and e.estimate("x" * 280) == 101
    for chars, tokens in ((4000, 1000), (4200, 1000), (3900, 1000)):
        e.observe(chars, tokens)
    cpt = e.chars_per_token()
    assert cpt == pytest.approx(3.9 * 0.93, rel=1e-6) and cpt > 2.8  # densest observed ratio, minus the safety margin
    assert e.estimate("x" * 3900) >= 3900 / 3.9  # never below the densest observation
    e.observe(2000, 1000)  # a denser prompt appears -> the estimate gets more conservative immediately
    assert e.chars_per_token() == pytest.approx(2.0 * 0.93 if 2.0 * 0.93 >= 2.0 else 2.0)
    s = e.state()
    assert s["observations"] == 4 and s["adapted"] and s["min_observed_chars_per_token"] == 2.0
    assert e.observe(0, 5) is None and e.observe(5, 0) is None  # unusable counts are ignored


def test_estimator_is_clamped_and_never_below_the_floor():
    e = TokenEstimator("m")
    for _ in range(5):
        e.observe(100, 1000)  # absurdly dense -> clamped at the floor
    assert e.chars_per_token() == 2.0
    for _ in range(64):
        e.observe(100000, 1000)
    assert 2.0 <= e.chars_per_token() <= 4.0


def _canon_big(n=60):
    ex = [{"action_type": "wait", "target_id": None}]
    return {"identity": {"agent_id": "a", "role": "Engineer"}, "executable_actions": ex, "blocked_actions": [], "last_rejection": None, "observations": {},
            "held_dik": {"knowledge": [f"rule number {i}: some knowledge text that is moderately long" for i in range(n)], "information": [], "data": []}, "recent_outcomes": {}}


def test_adaptive_estimate_drops_less_but_never_exceeds_the_budget():
    canon = _canon_big()
    fixed = build_prompt(canon, budget_tokens=900)
    e = TokenEstimator("m")
    for _ in range(4):
        e.observe(4000, 1000)  # 4.0 chars/token observed
    adaptive = build_prompt(canon, budget_tokens=900, estimator=e)
    kept = lambda b: b["metadata"]["sections"]["held_dik.knowledge"]["included_items"]
    assert kept(adaptive) > kept(fixed)
    assert adaptive["metadata"]["estimated_prompt_tokens"] <= 900  # the configured budget is never exceeded by the estimate
    assert adaptive["metadata"]["estimator_state"]["adapted"] and "session-adaptive" in adaptive["metadata"]["estimator"]
    # deterministic: same state -> same prompt; trimming priority unchanged (a prefix of the same ordered items is kept)
    assert build_prompt(canon, budget_tokens=900, estimator=e)["prompt_hash"] == adaptive["prompt_hash"]
    assert fixed["prompt_hash"] != adaptive["prompt_hash"]


def test_estimator_learns_from_real_prompt_eval_counts_and_is_recorded(monkeypatch):
    class Counted(FakeOllama):
        def chat(self, **kw):
            r = super().chat(**kw)
            chars = sum(len(m["content"]) for m in kw["messages"])
            r["token_metadata"] = {"prompt_eval_count": int(chars / 4.0), "eval_count": 7}
            return r

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: Counted(b, t))
    st = _sim({"Engineer": OLLAMA_A})
    try:
        _run(st, 70)
        est = st.token_estimators["model-a"]
        assert est.observations >= 3 and est.state()["adapted"] and est.chars_per_token() > 2.8
        rows = [json.loads(l) for l in (__import__("pathlib").Path(st.logger.output_session.session_folder) / "logs" / "controller_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        dec = [r for r in rows if r["kind"] == "ollama_decision"]
        assert all("token_estimation" in r and r["token_estimation"]["actual_prompt_tokens"] > 0 for r in dec)
        assert all(r["budget"]["estimated_prompt_tokens"] <= r["budget"]["budget_tokens"] or r["budget"]["tier1_over_budget"] for r in dec)
        assert "estimator_state" in dec[-1]["budget"]
    finally:
        _stop(st)
    manifest = json.loads((__import__("pathlib").Path(st.logger.output_session.session_folder) / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["token_estimators"]["model-a"]["observations"] >= 3


# ------------------------------------------------------------------ 4. outcome measures
def _validated(cm, pid, ptype, site="site_b"):
    cm.create_project(site, structure_type=ptype, project_id_override=pid)
    p = cm.projects[pid]
    p["status"], p["validated_complete"], p["correct"] = "complete", True, True
    return pid


def test_outcome_breakdown_keeps_connectors_separate_from_core_structures():
    cm = ConstructionManager()
    for i in range(3):
        _validated(cm, f"site_b_food_connector_{i:03d}", "food_connector")
    _validated(cm, "site_b_water_connector_001", "water_connector")
    only_connectors = validated_structure_breakdown(cm)
    assert only_connectors["total_validated_structures"] == 4 and only_connectors["validated_core_structures"] == 0
    assert only_connectors["validated_connectors"] == 4 and only_connectors["validated_food_connectors"] == 3 and only_connectors["validated_water_connectors"] == 1
    assert only_connectors["supported_civilian_capacity"] == {"min": 0, "max": 0} and only_connectors["connector_share_of_projects_started"] == 1.0
    _validated(cm, "site_b_house_001", "house")
    _validated(cm, "site_b_greenhouse_001", "greenhouse")
    _validated(cm, "site_b_water_generator_001", "water_generator")
    b = validated_structure_breakdown(cm)
    assert (b["validated_core_structures"], b["validated_houses"], b["validated_greenhouses"], b["validated_generators"]) == (3, 1, 1, 1)
    assert b["total_validated_structures"] == 7 and b["validated_connectors"] == 4
    assert {"structurally_supported_housing", "structurally_supported_greenhouses", "supported_civilian_capacity", "site_slots_used", "site_slots_exhausted"} <= set(b)


def test_final_measures_carry_the_breakdown_and_keep_the_legacy_total(procedural_sim):
    st = procedural_sim
    _validated(st.environment.construction, "site_b_food_connector_001", "food_connector")
    cons = st._final_measures()["construction"]
    assert cons["structures_validated"] == 1 and cons["outcome_breakdown"]["validated_core_structures"] == 0 and cons["outcome_breakdown"]["validated_connectors"] == 1
