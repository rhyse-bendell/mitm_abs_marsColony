import importlib
import unittest
from unittest.mock import patch

from modules.headless_runner import run_batch_experiment


class _FakeSim:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.update_calls = 0
        self.stopped = False
        _FakeSim.instances.append(self)

    def update(self, _dt):
        self.update_calls += 1

    def stop(self):
        self.stopped = True


class _StubVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class TestBatchRunner(unittest.TestCase):
    def test_single_run_exact_timesteps(self):
        _FakeSim.instances = []
        progress = []
        settings = {
            "agent_configs": [{"name": "Architect", "role": "Architect", "packet_access": ["Team_Packet"]}],
            "num_runs": 1,
            "timesteps_per_run": 7,
            "base_dt": 1.0,
            "experiment_name": "demo",
        }
        with patch("modules.headless_runner.SimulationState", _FakeSim):
            run_batch_experiment(settings, progress_callback=progress.append)

        self.assertEqual(len(_FakeSim.instances), 1)
        self.assertEqual(_FakeSim.instances[0].update_calls, 7)
        self.assertTrue(_FakeSim.instances[0].stopped)
        self.assertEqual(_FakeSim.instances[0].kwargs["experiment_name"], "demo")

    def test_multiple_runs_are_distinct_and_suffixed(self):
        _FakeSim.instances = []
        settings = {
            "agent_configs": [{"name": "Architect", "role": "Architect", "packet_access": ["Team_Packet"]}],
            "num_runs": 3,
            "timesteps_per_run": 2,
            "base_dt": 1.0,
            "experiment_name": "batch_test",
        }
        with patch("modules.headless_runner.SimulationState", _FakeSim):
            run_batch_experiment(settings)

        self.assertEqual(len(_FakeSim.instances), 3)
        self.assertEqual([sim.update_calls for sim in _FakeSim.instances], [2, 2, 2])
        run_names = [sim.kwargs["experiment_name"] for sim in _FakeSim.instances]
        self.assertEqual(run_names, ["batch_test_run001", "batch_test_run002", "batch_test_run003"])
        self.assertEqual(len(set(run_names)), 3)


class TestInterfaceExperimentHelpers(unittest.TestCase):
    def setUp(self):
        try:
            interface_mod = importlib.import_module("interface")
            self.MarsColonyInterface = interface_mod.MarsColonyInterface
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")

    def test_collect_experiment_settings_includes_batch_controls(self):
        app = self.MarsColonyInterface.__new__(self.MarsColonyInterface)
        app.num_runs = _StubVar(4)
        app.timesteps_per_run_var = _StubVar(123)
        app.speed_multiplier = _StubVar(1.5)
        app.experiment_name_var = _StubVar("exp")
        app.flash_mode = _StubVar(False)
        app.brain_backend_var = _StubVar("rule_brain")
        app.local_model_var = _StubVar("qwen")
        app.local_base_url_var = _StubVar("http://localhost")
        app.local_timeout_var = _StubVar(90.0)
        app.fallback_backend_var = _StubVar("rule_brain")
        app.enable_startup_llm_sanity_var = _StubVar(True)
        app.startup_llm_sanity_timeout_var = _StubVar(60.0)
        app.startup_llm_sanity_max_sources_var = _StubVar(1)
        app.startup_llm_sanity_max_items_var = _StubVar(2)
        app.startup_llm_sanity_raw_max_chars_var = _StubVar(4000)
        app.bootstrap_reuse_enabled_var = _StubVar(True)
        app.bootstrap_summary_max_chars_var = _StubVar(280)
        app.high_latency_local_llm_mode_var = _StubVar(True)
        app.unrestricted_local_qwen_mode_var = _StubVar(True)
        app.planner_timeout_seconds_var = _StubVar(90.0)
        app.planner_request_policy_var = _StubVar("cadence_with_dik_integration")
        app.planning_interval_steps_var = _StubVar(60)
        app.dik_integration_cooldown_steps_var = _StubVar(12)
        app.dik_integration_batch_threshold_var = _StubVar(2)
        app.warmup_timeout_var = _StubVar(45.0)
        app.startup_llm_sanity_completion_tokens_var = _StubVar(512)
        app.planner_completion_tokens_var = _StubVar(1024)
        app.high_latency_stale_result_grace_var = _StubVar(1800.0)
        app.BACKEND_DEFAULTS = self.MarsColonyInterface.BACKEND_DEFAULTS
        app.build_agent_configs = lambda: [{"name": "Architect", "role": "Architect", "packet_access": ["Team_Packet", "Architect_Packet"], "traits": {}}]
        app._collect_construction_parameters = lambda: {"pile_a_quantity": 20}

        settings = app._collect_experiment_settings()

        self.assertEqual(settings["num_runs"], 4)
        self.assertEqual(settings["timesteps_per_run"], 123)
        self.assertEqual(settings["agent_configs"][0]["packet_access"], ["Team_Packet", "Architect_Packet"])

    def test_packet_default_resolver_honors_role_specific_and_fallback(self):
        resolved = self.MarsColonyInterface._resolve_default_packet_access(["Team_Packet", "Architect_Packet"])
        self.assertEqual(resolved, {"Team_Packet", "Architect_Packet"})
        fallback = self.MarsColonyInterface._resolve_default_packet_access([])
        self.assertEqual(fallback, {"Team_Packet"})


if __name__ == "__main__":
    unittest.main()
