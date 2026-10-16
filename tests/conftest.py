from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "legacy_async_planner_vehicle: exercises the legacy async-planner infrastructure through a test-only Ollama provider vehicle")


@pytest.fixture(autouse=True)
def _legacy_async_planner_vehicle(request):
    """Test-only vehicle for the legacy async-planner infrastructure tests.

    Production no longer lets a brain_backend string instantiate Ollama (ControllerSpec `local_ollama` is the only path). These tests
    exercise planner timeouts/barriers/fallback bookkeeping and used the retired alias as a convenient provider; for them only, the
    alias resolution is restored inside this fixture. It is never active for any other test and never touches ControllerSpec routing."""
    if request.node.get_closest_marker("legacy_async_planner_vehicle") is None:
        yield
        return
    from unittest.mock import patch

    import modules.brain_provider as bp
    import modules.controller_spec as cs
    import modules.simulation as sim_mod

    legacy = tuple(cs.LEGACY_OLLAMA_BACKENDS)
    real_create = bp.create_brain_provider

    def vehicle(config=None):
        config = config or bp.BrainBackendConfig()
        if config.backend.lower() in legacy:
            return bp.OllamaLocalBrainProvider(config=config, fallback=bp.ProceduralBaselinePilot())
        return real_create(config)

    with patch.object(cs, "LEGACY_OLLAMA_BACKENDS", ()), patch.object(sim_mod, "create_brain_provider", vehicle):
        yield
