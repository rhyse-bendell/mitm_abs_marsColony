from pathlib import Path


def test_agent_does_not_import_procedural_baseline_fallback_selector():
    text = Path("modules/agent.py").read_text()
    assert "from modules.brain_provider import select_productive_fallback_action" not in text
    assert "select_productive_fallback_action(" not in text
