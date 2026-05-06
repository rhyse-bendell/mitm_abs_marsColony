from modules.procedural_baseline_pilot import ProceduralBaselinePilotAdapter


class _StubProvider:
    def __init__(self):
        self.called = False

    def generate_plan(self, request):
        self.called = True
        return "ok"


def test_adapter_has_expected_identity():
    adapter = ProceduralBaselinePilotAdapter(provider=_StubProvider())
    assert adapter.pilot_id == "procedural_baseline"


def test_choose_action_delegates_to_provider():
    provider = _StubProvider()
    adapter = ProceduralBaselinePilotAdapter(provider=provider)
    out = adapter.choose_action(request=None)
    assert provider.called is True
    assert out == "ok"


def test_handle_blocked_action_noop_returns_none():
    adapter = ProceduralBaselinePilotAdapter(provider=_StubProvider())
    assert adapter.handle_blocked_action(agent=None, original_decision=None, gate_result=None) is None
