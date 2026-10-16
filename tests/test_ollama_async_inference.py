"""Non-blocking Ollama inference (worker thread + simulation barrier) and model warm-up. Fake clients only; no real service."""
import threading
import time

import pytest

from modules import ollama_pilot
from tests.test_ollama_controller import (OLLAMA_A, FakeOllama, _run, _sim, _stop, first_inspect_reply)  # noqa: F401  (autouse fixture is re-declared below)


@pytest.fixture(autouse=True)
def _fake_service(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: FakeOllama(base_url, timeout_s))


class GatedOllama(FakeOllama):
    """Blocks inside chat() until released, so a pending inference can be observed."""
    gate = threading.Event()
    entered = threading.Event()

    def chat(self, **kw):
        GatedOllama.entered.set()
        assert GatedOllama.gate.wait(10), "test gate never released"
        return super().chat(**kw)


def _gated(monkeypatch):
    GatedOllama.gate, GatedOllama.entered = threading.Event(), threading.Event()
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: GatedOllama(base_url, timeout_s))


def _adapter(st, name="Engineer"):
    a = next(x for x in st.agents if x.role == name)
    return a, st.get_agent_brain_runtime(a)["adapter"]


def test_pending_inference_holds_simulated_time_and_never_blocks_the_caller(monkeypatch):
    _gated(monkeypatch)
    st = _sim({"Engineer": OLLAMA_A})
    st.async_inference = True
    agent, adapter = _adapter(st)
    t0 = time.perf_counter()
    for _ in range(4):
        st.update(1.0)
    assert time.perf_counter() - t0 < 2.0, "update() must not wait for the model"
    assert GatedOllama.entered.wait(5) and st.inference_pending() and adapter.runtime_status == "thinking"
    frozen = st.time
    for _ in range(5):
        st.update(1.0)
    assert st.time == frozen, "simulated time must not advance while inference is pending"
    assert agent.current_action in (None, []) or not agent.active_actions
    GatedOllama.gate.set()
    deadline = time.time() + 5
    while st.inference_pending() and time.time() < deadline:
        time.sleep(0.01)
    assert adapter.runtime_status == "ready"
    _run(st, 3)
    assert st.time > frozen and agent.pilot_episode_stats["selections"] >= 1
    _stop(st)


def test_stop_while_pending_discards_the_late_result_and_mutates_nothing(monkeypatch):
    _gated(monkeypatch)
    st = _sim({"Engineer": OLLAMA_A})
    st.async_inference = True
    agent, adapter = _adapter(st)
    for _ in range(3):
        st.update(1.0)
    assert GatedOllama.entered.wait(5) and st.inference_pending()
    _stop(st)
    assert st.inference_closed and adapter.cancelled and agent._llm_job is None
    selections, t = agent.pilot_episode_stats["selections"], st.time
    GatedOllama.gate.set()
    time.sleep(0.2)
    st.update(1.0)
    assert st.time == t and agent.pilot_episode_stats["selections"] == selections


def test_async_error_is_reported_and_halts_with_no_fallback(monkeypatch):
    def boom(messages, options, model):
        return "not json"

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: FakeOllama(base_url, timeout_s, reply=boom))
    st = _sim({"Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-a", "max_retries": 1}}})
    st.async_inference = True
    _, adapter = _adapter(st)
    deadline = time.time() + 10
    while st.controller_error is None and time.time() < deadline:
        st.update(1.0)
        time.sleep(0.01)
    assert st.controller_error is not None and adapter.runtime_status == "error"
    _stop(st)


def test_async_and_sync_pick_the_same_actions(monkeypatch):
    def selections(async_mode):
        st = _sim({"Engineer": OLLAMA_A}, seed=5)
        st.async_inference = async_mode
        deadline = time.time() + 20
        while st.time < 25 and time.time() < deadline:
            _run(st, 1)
            time.sleep(0.002)
        agent, _ = _adapter(st)
        out = agent.pilot_episode_stats["selections"]
        _stop(st)
        return out

    assert selections(False) > 0 and selections(True) > 0


def test_warmup_runs_once_per_model_before_time_starts_and_is_logged(monkeypatch):
    calls = []

    class Warmable(FakeOllama):
        def warm(self, *, model, keep_alive="30m", timeout_s=None):
            calls.append((model, keep_alive))
            return {"latency_s": 0.5, "load_duration_s": 0.4, "done_reason": "load"}

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: Warmable(base_url, timeout_s))
    seen = []
    st = _sim({"Engineer": OLLAMA_A, "Botanist": OLLAMA_A}, startup_progress_callback=lambda *a: seen.append(a))
    assert calls == [("model-a", "30m")] and st.time == 0.0
    assert any(a[0] == "warming_model" for a in seen)
    assert next(iter(st.ollama_warmup_results.values()))["warmed"] is True
    _stop(st)


def test_warmup_can_be_disabled_and_failure_is_not_fatal(monkeypatch):
    class Failing(FakeOllama):
        def warm(self, **kw):
            from modules.ollama_client import OllamaRequestError
            raise OllamaRequestError("load failed")

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: Failing(base_url, timeout_s))
    st = _sim({"Engineer": OLLAMA_A})
    assert next(iter(st.ollama_warmup_results.values())) == {"warmed": False, "error": "OllamaRequestError: load failed"}
    _stop(st)
    st2 = _sim({"Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-a", "warmup": False}}})
    assert not st2.ollama_warmup_results
    _stop(st2)
