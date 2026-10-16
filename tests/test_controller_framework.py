"""Per-agent controller framework: independent controller specs, canonical controller-neutral context, manifest/telemetry, seeds, GUI, no remote code.

Only the Procedural Baseline is enabled. `local_ollama` is a declared placeholder that can be described, serialised and put in a manifest
but never instantiated; nothing here may invoke a remote provider.
"""

import contextlib
import io
import json
import random
import tempfile
from pathlib import Path

import pytest

from modules import controller_spec as cs
from modules.controller_context import FORBIDDEN_KEYS, assert_canonical_context_clean, build_canonical_context, find_forbidden_keys
from modules.simulation import SimulationState

ROLES = ["Architect", "Botanist", "Engineer"]


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="ctrl", seed=3)
        try:
            yield state
        finally:
            with contextlib.redirect_stdout(io.StringIO()):
                state.stop()


def _agent(sim, role):
    return next(a for a in sim.agents if a.role == role)


def _ctx(sim, agent):
    packet = sim.brain_context_builder.build(sim, agent)
    return build_canonical_context(packet, agent_id=agent.agent_id, role=agent.role, controller_type=agent.controller_type_id())


# ------------------------------------------------------------------ specification


def test_each_role_resolves_its_own_controller_independently():
    specs = cs.resolve_controller_assignments(ROLES)
    assert set(specs) == set(ROLES) and all(s.controller_type == cs.PROCEDURAL_BASELINE and s.enabled for s in specs.values())
    mixed = cs.resolve_controller_assignments(ROLES, {"Botanist": {"controller_type": "local_ollama", "settings": {"model": "model-a", "temperature": 0.2}}, "Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-b"}}})
    assert mixed["Architect"] == specs["Architect"]  # changing other agents never alters this one
    assert mixed["Botanist"].settings["model"] == "model-a" and mixed["Engineer"].settings["model"] == "model-b"
    only_botanist = cs.resolve_controller_assignments(ROLES, {"Botanist": "local_ollama"})
    assert only_botanist["Architect"] == specs["Architect"] and only_botanist["Engineer"] == specs["Engineer"] and only_botanist["Botanist"] != specs["Botanist"]


def test_compositions_all_one_two_all_and_same_or_different_models_are_expressible_and_distinct():
    def comp(a):
        return cs.condition_identity(cs.resolve_controller_assignments(ROLES, a))

    ids = {
        comp({}),
        comp({"Botanist": "local_ollama"}),
        comp({"Botanist": "local_ollama", "Engineer": "local_ollama"}),
        comp({r: "local_ollama" for r in ROLES}),
        comp({"Botanist": {"controller_type": "local_ollama", "settings": {"model": "a"}}, "Engineer": {"controller_type": "local_ollama", "settings": {"model": "a"}}}),
        comp({"Botanist": {"controller_type": "local_ollama", "settings": {"model": "a"}}, "Engineer": {"controller_type": "local_ollama", "settings": {"model": "b"}}}),
    }
    assert len(ids) == 6  # composition (and per-agent model) is part of the condition identity


def test_invalid_specs_are_rejected_and_placeholders_never_instantiate(sim):
    with pytest.raises(cs.ControllerConfigError):
        cs.ControllerSpec.from_value("gpt_somebody")
    with pytest.raises(cs.ControllerConfigError):
        cs.ControllerSpec.from_value({"controller_type": "procedural_baseline", "settings": {"model": "x"}})  # procedural carries no generative settings
    with pytest.raises(cs.ControllerConfigError):
        cs.resolve_controller_assignments(ROLES, {"Gardener": "procedural_baseline"})
    with pytest.raises(cs.ControllerNotEnabledError):
        cs.ControllerSpec.from_value("local_ollama").backend()
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(cs.ControllerNotEnabledError):
        SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="x", seed=1, controller_assignments={"Botanist": "local_ollama"})
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(cs.ControllerConfigError):
        SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="x", seed=1, controller_assignments={"Nobody": "procedural_baseline"})


def test_generative_settings_have_a_defined_place_and_procedural_has_none():
    ollama = cs.ControllerSpec.from_value("local_ollama").to_dict()
    assert set(ollama["settings"]) == set(cs.GENERATIVE_SETTING_FIELDS) and ollama["enabled"] is False
    assert {"model", "temperature", "controller_generation_seed", "max_context_tokens", "max_output_tokens", "timeout_s", "prompt_version", "inference_parameters"} <= set(ollama["settings"])
    assert cs.ControllerSpec.from_value("procedural_baseline").to_dict()["settings"] == {}


# ------------------------------------------------------------------ runtime plumbing


def test_three_explicit_procedural_controllers_reproduce_existing_behaviour():
    def run(assignments):
        with tempfile.TemporaryDirectory() as tmp:
            random.seed(5)
            st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="repro", seed=5, controller_assignments=assignments)
            seq = []
            st.logger.register_event_listener(lambda e: seq.append((e["time"], e["payload_data"].get("agent"), e["payload_data"].get("selected_action"), e["payload_data"].get("target_id"))) if e["event_type"] == "pilot_selection_via_adapter" else None)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(150):
                    st.update(1.0)
                st.stop()
            return seq

    baseline = run(None)
    assert baseline and run({r: "procedural_baseline" for r in ROLES}) == baseline


def test_each_agent_gets_its_own_spec_and_telemetry_names_the_controller():
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="tele", seed=2, controller_assignments={r: "procedural_baseline" for r in ROLES})
        seen = {}
        st.logger.register_event_listener(lambda e: seen.setdefault(e["payload_data"].get("agent"), e["payload_data"]) if e["event_type"] == "pilot_selection_via_adapter" else None)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(60):
                st.update(1.0)
            st.stop()
        assert {a.role for a in st.agents} == set(ROLES) and len({id(a.controller_spec) for a in st.agents}) >= 1
        assert all(a.controller_spec.controller_type == "procedural_baseline" for a in st.agents)
        assert set(seen) == {a.name for a in st.agents}
        assert all(p["controller_type"] == "procedural_baseline" and p["controller_agent_id"] for p in seen.values())
        assert len({p["controller_agent_id"] for p in seen.values()}) == 3


def test_manifest_reconstructs_all_three_assignments_and_keeps_seeds_distinct(sim):
    manifest = sim.controller_manifest()
    assert set(manifest["per_agent"]) == {a.name for a in sim.agents}
    assert manifest["composition_id"] == "|".join(f"{n}=procedural_baseline" for n in sorted(a.name for a in sim.agents))
    for name, row in manifest["per_agent"].items():
        assert row["controller"]["controller_type"] == "procedural_baseline" and row["agent_id"] and row["role"] in ROLES
        assert set(row["experimental_profile"]["resolved_constituents"]) >= {"coordination_propensity", "memory_reliability"}
        assert row["seeds"] == {"world_seed": 3, "controller_generation_seed": None}  # separate fields; the world seed never doubles as a generation seed
    # the same data is written to the session manifest on disk
    on_disk = json.loads(Path(sim.logger.output_session.session_folder, "session_manifest.json").read_text(encoding="utf-8"))
    flat = json.dumps(on_disk)
    assert "controller_assignments" in flat and "controller_generation_seed" in flat and "composition_id" in flat


def test_controller_generation_seed_is_independent_of_the_world_seed():
    spec = cs.ControllerSpec.from_value({"controller_type": "local_ollama", "settings": {"controller_generation_seed": 99}})
    assert spec.to_dict()["settings"]["controller_generation_seed"] == 99
    state = random.getstate()
    cs.resolve_controller_assignments(ROLES, {"Botanist": spec})
    assert random.getstate() == state  # resolving controller settings never touches the world RNG


# ------------------------------------------------------------------ canonical, controller-neutral context


def test_canonical_context_is_per_agent_clean_and_does_not_leak_private_dik(sim):
    a, b = _agent(sim, "Architect"), _agent(sim, "Botanist")
    element = sim.task_model.dik_elements["I_BOT_GREENHOUSE_DEPENDENCIES"]
    for kind in ("data", "information"):
        for ag in (a, b):
            for item in list(ag.mental_model[kind]):
                ag.mental_model[kind].discard(item)
    b.mental_model["information"].add(b._create_dik_object_from_element(element, source_id="private"))
    ca, cb = _ctx(sim, a), _ctx(sim, b)
    assert ca["identity"]["role"] == "Architect" and cb["identity"]["role"] == "Botanist" and ca["identity"]["agent_id"] != cb["identity"]["agent_id"]
    assert "I_BOT_GREENHOUSE_DEPENDENCIES" in json.dumps(cb["held_dik"]) and "I_BOT_GREENHOUSE_DEPENDENCIES" not in json.dumps(ca["held_dik"])
    assert not find_forbidden_keys(ca) and not find_forbidden_keys(cb)
    assert set(ca) == {"schema", "identity", "task", "goals", "observations", "held_dik", "known_gaps", "communications_and_artifacts", "team_plan_context", "recent_outcomes", "legal_actions"}


def test_forbidden_keys_are_rejected_at_any_depth():
    assert {"correct", "environment", "mental_model"} <= FORBIDDEN_KEYS
    with pytest.raises(ValueError):
        assert_canonical_context_clean({"observations": {"built_state": [{"id": "p", "correct": False}]}})
    assert_canonical_context_clean({"observations": {"built_state": [{"id": "p", "validated_correctness": True}]}})


def test_request_carries_the_canonical_context_for_the_baseline_too(sim):
    a = _agent(sim, "Architect")
    context = sim.brain_context_builder.build(sim, a)
    request = a._build_sync_pilot_request(sim, context, "test")
    canon = request.task_context["canonical_context"]
    assert canon["identity"]["controller_type"] == "procedural_baseline" and canon["legal_actions"] == list(context.action_affordances)
    assert not find_forbidden_keys(request.task_context["canonical_context"])


def test_plan_context_in_the_canonical_view_follows_legitimate_acquisition(sim):
    from modules.team_knowledge import TeamKnowledgeManager  # noqa: F401

    bot = _agent(sim, "Botanist")
    m = sim.team_knowledge_manager
    m.propose_team_plan(plan_id="px", proposed_by="Architect", sim_time=sim.time, goal_ids=["g"], goal_summary="g", project_targets=["SECRET_PROJECT"],
                        assignments_by_role={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "SECRET_PROJECT"}},
                        supporters=["Architect"], expires_at=sim.time + 100.0, review_at=sim.time + 20.0)
    before = _ctx(sim, bot)
    assert "SECRET_PROJECT" not in json.dumps(before, default=str) and before["team_plan_context"]["known_plan"] is None
    summary = m._team_plan_summary(dict(m.artifacts["team_plan:px"].content))
    bot._note_known_plan(sim, summary, via="message")
    after = _ctx(sim, bot)
    assert after["team_plan_context"]["known_plan"]["plan_id"] == "px" and after["team_plan_context"]["known_plan"]["read_via"] == "message"


# ------------------------------------------------------------------ headless config and no remote code


def test_headless_condition_accepts_per_agent_controllers_and_reports_the_composition():
    from modules.experiment_runner import run_condition

    with contextlib.redirect_stdout(io.StringIO()):
        summary = run_condition(seed=1, duration_s=30, controller_assignments={"Architect": "procedural_baseline", "Botanist": "procedural_baseline", "Engineer": "procedural_baseline"})
    assert summary["controller_composition_id"].count("procedural_baseline") == 3
    assert set(summary["controller_assignments"]["per_agent"]) == set(ROLES)
    assert cs.parse_cli_assignments(["Botanist=local_ollama", "Engineer=procedural_baseline"]) == {"Botanist": "local_ollama", "Engineer": "procedural_baseline"}


def test_batch_conditions_carry_controller_composition():
    import importlib.util

    spec = importlib.util.spec_from_file_location("charbatch", Path("scripts/characterization_batch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cond = mod._cond(controllers={"Botanist": "procedural_baseline"})
    mod.CONDITIONS["x"] = cond
    assert mod._settings("x")["controller_assignments"] == {"Botanist": "procedural_baseline"}


def test_no_remote_provider_code_is_involved(monkeypatch):
    import urllib.request

    src = (Path("modules") / "controller_spec.py").read_text(encoding="utf-8") + (Path("modules") / "controller_context.py").read_text(encoding="utf-8")
    assert not any(tok in src for tok in ("urllib", "requests", "http.client", "openai", "anthropic", "brain_provider"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network call attempted")))
    from modules import brain_provider

    monkeypatch.setattr(brain_provider.OllamaLocalBrainProvider, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Ollama provider constructed")))
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="nonet", seed=1, controller_assignments={r: "procedural_baseline" for r in ROLES})
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(20):
                st.update(1.0)
            st.stop()


# ------------------------------------------------------------------ GUI


@pytest.fixture
def app():
    pytest.importorskip("tkinter")
    import interface

    try:
        a = interface.MarsColonyInterface()
    except Exception as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass


def test_gui_has_three_independent_per_agent_controller_selectors(app):
    roles = app.agent_card_order[:3]
    assert len(roles) == 3 and all(r in app.agent_controller_vars for r in roles)
    assert {app.agent_controller_vars[r].get() for r in roles} == {"Procedural Baseline"}
    settings = app._collect_experiment_settings()
    assert {r: v["controller_type"] for r, v in settings["controller_assignments"].items()} == {r: "procedural_baseline" for r in roles}
    assert all(cfg["controller"]["controller_type"] == "procedural_baseline" for cfg in settings["agent_configs"])
    assert "Controllers:" in app.test_drive_summary_var.get()
    # only the enabled controller is selectable now, and the selectors are distinct variables (no team-wide switch)
    assert len({id(app.agent_controller_vars[r]) for r in roles}) == 3
    app.agent_controller_vars[roles[1]].set("Procedural Baseline")
    app.apply_controller_to_all()
    app.reset_all_controllers()
    assert {app.agent_controller_vars[r].get() for r in roles} == {"Procedural Baseline"}
    assert not hasattr(app, "global_controller_var")
