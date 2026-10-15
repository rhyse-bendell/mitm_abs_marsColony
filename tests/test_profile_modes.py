"""Profile modes: broad_legacy (current behavior, default) vs compact_panel (residual mechanisms frozen at the reference)."""

import copy
import json
import tempfile

import pytest

from modules.experimental_profile import available_profile_modes, load_experimental_profile_config, resolve_experimental_profile, resolve_profile_mode
from modules.experiment_runner import main as headless_main, run_condition
from modules.simulation import SimulationState

CFG = load_experimental_profile_config()
RESIDUALS = list(CFG["profile_modes"]["modes"]["compact_panel"]["residual_mechanisms"])
HIGH_TEAM_LOW_TASK = {"teamwork_potential": 0.9, "taskwork_potential": 0.1}


def _sim(settings=None, agent_configs=None, seed=1):
    tmp = tempfile.TemporaryDirectory()
    state = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="modes", seed=seed, experimental_settings=settings or {}, **({"agent_configs": agent_configs} if agent_configs else {}))
    state._tmp = tmp
    return state


@pytest.fixture
def make_sim():
    made = []

    def factory(*a, **k):
        s = _sim(*a, **k)
        made.append(s)
        return s

    yield factory
    for s in made:
        s.stop()
        s._tmp.cleanup()


def _profiles(state):
    return {a.name: dict(a.mechanism_profile) for a in state.agents}


def test_15_broad_legacy_is_the_default_and_reproduces_current_residuals(make_sim):
    default = make_sim({"constructs": HIGH_TEAM_LOW_TASK})
    broad = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "broad_legacy"})
    assert _profiles(default) == _profiles(broad)
    agent = default.agents[0]
    assert agent.experimental_profile["profile_mode"] == "broad_legacy"
    assert agent.experimental_profile["residual_mechanism_references"] == {}
    # the gestalts still drive the residual mechanisms in broad mode (current behavior)
    assert any(abs(agent.mechanism_profile[m] - 0.5) > 1e-6 for m in RESIDUALS)


def test_16_compact_panel_freezes_residuals_exactly(make_sim):
    state = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "compact_panel"})
    for agent in state.agents:
        assert {m: agent.mechanism_profile[m] for m in RESIDUALS} == {m: 0.5 for m in RESIDUALS}


def test_17_constituent_overrides_win_in_compact_panel(make_sim):
    state = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "compact_panel", "constituent_overrides": {"dik_push_propensity": 0.95, "task_execution_efficiency": 0.2}})
    for agent in state.agents:
        assert agent.mechanism_profile["dik_push_propensity"] == pytest.approx(0.95)
        assert agent.mechanism_profile["build_speed"] == pytest.approx(0.2)
        assert all(agent.mechanism_profile[m] == 0.5 for m in RESIDUALS)


def test_18_legacy_low_level_overrides_apply_last_and_are_logged(make_sim):
    configs = [{"name": "Architect", "role": "Architect", "constructs": HIGH_TEAM_LOW_TASK, "mechanism_overrides": {"rule_accuracy": 0.9}}]
    state = make_sim({"profile_mode": "compact_panel"}, agent_configs=configs)
    agent = state.agents[0]
    assert agent.mechanism_profile["rule_accuracy"] == pytest.approx(0.9)
    assert agent.experimental_profile["legacy_overrides_of_residual_mechanisms"]["rule_accuracy"] == {"frozen_value": 0.5, "legacy_override": 0.9}
    manifest = state._experimental_manifest()["per_agent"]["Architect"]
    assert manifest["legacy_overrides_of_residual_mechanisms"]["rule_accuracy"]["legacy_override"] == 0.9


def test_19_mode_residual_policy_and_final_profile_are_in_the_manifest(make_sim):
    state = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "compact_panel"})
    manifest = state._experimental_manifest()
    assert manifest["profile_mode"] == "compact_panel" and manifest["residual_policy"]
    assert manifest["residual_mechanism_references"] == {m: 0.5 for m in RESIDUALS}
    assert "compact-panel reference" in manifest["residual_reference_label"] and "human" in manifest["residual_reference_label"]
    row = manifest["per_agent"][state.agents[0].name]
    for key in ("gestalt_constructs", "generated_constituents", "explicit_constituent_overrides", "resolved_constituents", "residual_mechanism_values", "resolved_mechanisms"):
        assert key in row
    assert row["residual_mechanism_values"] == {m: 0.5 for m in RESIDUALS}
    with open(state.logger.output_session.session_folder / "session_manifest.json", encoding="utf-8") as fh:
        assert json.load(fh)["experimental_panel"]["profile_mode"] == "compact_panel"


@pytest.mark.parametrize("mode", ["broad_legacy", "compact_panel"])
def test_20_same_seed_and_mode_are_identical(make_sim, mode):
    runs = []
    for _ in range(2):
        s = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": mode}, seed=4)
        for _t in range(40):
            s.update(1.0)
        runs.append((_profiles(s), [(a.name, tuple(round(x, 6) for x in a.position)) for a in s.agents]))
    assert runs[0] == runs[1]


def test_21_mode_changes_only_residual_mechanism_resolution(make_sim):
    broad = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "broad_legacy"})
    compact = make_sim({"constructs": HIGH_TEAM_LOW_TASK, "profile_mode": "compact_panel"})
    for a, b in zip(broad.agents, compact.agents):
        assert a.experimental_profile["resolved_constituents"] == b.experimental_profile["resolved_constituents"]
        assert a.experimental_profile["constituent_mechanisms"] == b.experimental_profile["constituent_mechanisms"]
        differing = {k for k in a.mechanism_profile if a.mechanism_profile[k] != b.mechanism_profile[k]}
        assert differing <= set(RESIDUALS)


def test_22_unknown_mode_is_an_error_and_modes_are_listed():
    with pytest.raises(ValueError):
        resolve_profile_mode("nonsense")
    with pytest.raises(ValueError):
        resolve_experimental_profile({"teamwork_potential": 0.5}, profile_mode="nonsense")
    assert set(available_profile_modes()) == {"broad_legacy", "compact_panel"}
    assert resolve_profile_mode(None)["profile_mode"] == "broad_legacy"


def test_23_headless_profile_mode_reaches_the_run(tmp_path, capsys):
    out = tmp_path / "summary.json"
    assert headless_main(["--seed", "0", "--duration", "3", "--profile-mode", "compact_panel", "--out", str(out), "--project-root", str(tmp_path / "root")]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["manifest_condition"]["profile_mode"] == "compact_panel"
    assert summary["experimental_settings"]["profile_mode"] == "compact_panel"