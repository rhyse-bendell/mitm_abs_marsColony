"""Experiment tab boots into a plain Procedural Baseline test drive (no LLM, compact panel, neutral gestalts, seed 0)."""

import os

import pytest

pytest.importorskip("tkinter")

import interface  # noqa: E402


@pytest.fixture
def app():
    try:
        a = interface.MarsColonyInterface()
    except Exception as exc:  # no display / Tk unavailable
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass


def _check_defaults(app):
    st = app._collect_experiment_settings()
    assert st["profile_mode"] == "compact_panel"
    assert st["brain_backend"] == "rule_brain" and st["seed"] == 0
    assert st["planner_config"]["enable_startup_llm_sanity"] is False
    assert st["timesteps_per_run"] == 600 and int(app.stop_at_sim_time_var.get()) == 600
    assert st["speed"] * app.base_dt == pytest.approx(1.0)  # one sim-second per tick, like the headless step
    for cfg in st["agent_configs"]:
        assert cfg["constructs"] == {"teamwork_potential": 0.5, "taskwork_potential": 0.5}
        assert cfg["mechanism_overrides"] == {} and cfg["constituent_overrides"] == {}
        assert cfg["initial_goal_seeds"] and cfg["agent_id"].endswith("_1")  # task-package defaults reach the sim, as headless
    text = app.test_drive_summary_var.get()
    assert "Procedural Baseline Pilot" in text and "Seed: 0" in text and "inherited from profile" in text and "Output:" in text


def test_gui_boots_into_test_drive_defaults(app):
    _check_defaults(app)


def test_reset_restores_test_drive_defaults(app):
    role = app.agent_card_order[0]
    app.agent_profiles[role]["team_value"].set(0.9)
    app.agent_constituents[role]["memory_reliability"].set("0.2")
    app.legacy_override_enabled[role].set(True)
    app.seed_var.set(7)
    app.profile_mode_var.set(next(l for l, k in app._profile_mode_labels.items() if k == "broad_legacy"))
    app.brain_backend_var.set("ollama")
    assert "overridden: memory_reliability" in app.test_drive_summary_var.get()
    assert "NOT the test-drive default" in app.test_drive_summary_var.get()
    app.reset_to_test_drive_defaults()
    _check_defaults(app)


def test_interactive_auto_stop_uses_sim_time(app):
    class _Sim:
        time = 599.0

    app.sim = _Sim()
    assert not app._interactive_duration_reached()
    app.sim.time = 600.0
    assert app._interactive_duration_reached()
    app.stop_at_sim_time_var.set(0)
    assert not app._interactive_duration_reached()
