from modules.brain_provider import BrainBackendConfig, create_brain_provider, resolve_pilot_display_name


def test_rule_brain_backend_resolves():
    provider = create_brain_provider(BrainBackendConfig(backend="rule_brain"))
    assert provider.__class__.__name__ == "RuleBrain"


def test_procedural_baseline_backend_resolves():
    provider = create_brain_provider(BrainBackendConfig(backend="procedural_baseline"))
    assert provider.__class__.__name__ == "RuleBrain"


def test_display_name_helper_prefers_procedural_baseline():
    assert resolve_pilot_display_name("rule_brain") == "Procedural Baseline Pilot"
