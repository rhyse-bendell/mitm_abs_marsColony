"""Resource guard shared by the GUI governor and the headless synchronous Ollama path. Fake telemetry only: no real GPU/RAM probe, no real service,
no process is touched; sleeps and clocks are injected."""
import json
import tempfile
from pathlib import Path

import pytest

from modules import ollama_pilot
from modules.ollama_client import OllamaRequestError
from modules.inference_governor import InferenceGovernor
from modules.resource_guard import MB, GuardCancelled, ResourceGuard
from tests.test_ollama_controller import FakeOllama, OLLAMA_A, _run, _sim, _stop  # noqa: F401


@pytest.fixture(autouse=True)
def _fake_service(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t))


def gpu(used, total=8188.0, util=5.0):
    return {"name": "FakeGPU", "total_mb": total, "used_mb": used, "free_mb": total - used, "util_percent": util}


def models(vram=5252.0, size=5252.0):
    return [{"name": "m", "size_bytes": int(size * 1048576), "size_vram_bytes": int(vram * 1048576)}]


class ScriptedMonitor:
    """refresh() pops the next scripted snapshot (the last one repeats)."""

    def __init__(self, *snaps):
        self.snaps = list(snaps)
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1
        return self.snaps.pop(0) if len(self.snaps) > 1 else self.snaps[0]

    def snapshot(self):
        return self.snaps[0]


def snap(used=700.0 + 5252.0, ram=16000.0, util=5.0, detail=None, with_gpu=True):
    return {"ram_available_mb": ram, "ram_total_mb": 32000.0, "gpu": gpu(used, util=util) if with_gpu else None,
            "ollama_loaded_detail": models() if detail is None else detail}


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def make_guard(monitor, clock=None, events=None, **kw):
    clock = clock or Clock()
    return ResourceGuard(monitor, 2048.0, on_event=(lambda n, p: events.append((n, p))) if events is not None else None, clock=clock.now, sleep=clock.sleep, poll_s=1.0, **kw), clock


# ------------------------------------------------------------------ evaluation
def test_idle_clean_state_is_safe_and_learns_the_session_baseline():
    g, _ = make_guard(ScriptedMonitor(snap()))
    reason, detail = g.check()
    assert reason is None and g.baseline_other_mb == pytest.approx(700.0) and g.best_residency == pytest.approx(1.0) and g.gpu_available is True


def test_low_ram_is_unsafe_without_any_gpu():
    g, _ = make_guard(ScriptedMonitor(snap(ram=1000.0, with_gpu=False)))
    assert g.check()[0] == "low_memory"
    g2, _ = make_guard(ScriptedMonitor(snap(ram=16000.0, with_gpu=False)))
    assert g2.check() == (None, {"ram_available_mb": 16000.0}) and g2.gpu_available is False  # nvidia-smi absent: RAM protection only, run continues
    assert g2.manifest_state()["gpu_telemetry"] == "unavailable"


def test_non_model_vram_growth_alone_is_reported_not_a_reason_but_lost_headroom_is():
    # growth of non-model VRAM (another app OR Ollama's own runtime buffers - indistinguishable) is information only
    g, _ = make_guard(ScriptedMonitor(snap(used=700.0 + 5252.0), snap(used=700.0 + 5252.0 + 1500.0)))
    assert g.check()[0] is None
    reason, detail = g.check()
    assert reason is None and detail["other_vram_mb"] == pytest.approx(2200.0)
    # what is unsafe is a direct condition: a neighbour that takes the headroom
    g2, _ = make_guard(ScriptedMonitor(snap(used=700.0 + 5252.0), snap(used=8100.0)))
    assert g2.check()[0] is None and g2.check()[0] == "gpu_vram_headroom_lost"
    g3, _ = make_guard(ScriptedMonitor(snap(used=700.0), snap(used=700.0 + 5252.0)))  # the model loading is not another application
    assert g3.check()[0] is None


def test_loss_of_vram_headroom_and_partial_offload_are_detected():
    g, _ = make_guard(ScriptedMonitor(snap(used=8100.0)))
    assert g.check()[0] == "gpu_vram_headroom_lost"
    g2, _ = make_guard(ScriptedMonitor(snap(), snap(used=700.0 + 3000.0, detail=models(vram=3000.0, size=5252.0))))
    assert g2.check()[0] is None
    assert g2.check()[0] == "model_vram_residency_dropped"


def test_high_gpu_utilisation_while_idle_is_only_judged_when_probed():
    g, _ = make_guard(ScriptedMonitor(snap(util=99.0)))
    assert g.check(util_check=False)[0] is None
    assert g.check(util_check=True)[0] == "gpu_busy_other_process"


# ------------------------------------------------------------------ headless wait: throttles, resumes, stays cancellable
def test_before_call_waits_until_clear_for_consecutive_polls_and_records_it():
    events = []
    mon = ScriptedMonitor(snap(), snap(used=7900.0, util=98.0), snap(used=7900.0, util=98.0), snap(util=6.0), snap(util=6.0), snap(util=6.0))
    g, clock = make_guard(mon, events=events, idle_probe_s=0.0)
    assert g.before_call() == 0.0          # first call: clean
    waited = g.before_call()               # game appears: throttled, then clear for 2 polls
    assert waited >= 3.0 and g.throttle_events == 1 and g.reasons == {"gpu_vram_headroom_lost": 1}
    names = [n for n, _ in events]
    assert names == ["inference_resource_wait_started", "inference_resource_wait_ended"] and events[1][1]["waited_s"] == pytest.approx(waited)
    st = g.manifest_state()
    assert st["active"] and st["throttle_events"] == 1 and st["total_resource_wait_s"] == pytest.approx(round(waited, 3))


def test_wait_is_cancellable_and_never_inspects_other_processes():
    mon = ScriptedMonitor(snap(), snap(used=8100.0))
    g, _ = make_guard(mon, idle_probe_s=0.0)
    g.before_call()
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 3

    with pytest.raises(GuardCancelled):
        g.before_call(cancel=cancel)
    import re
    src = Path(__import__("modules.resource_guard", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    assert not re.search(r"\b(kill|terminate|taskkill|Popen|psutil|os\.system|subprocess)\b", src)  # observes only; manipulates nothing


def test_slow_call_arms_a_gpu_busy_probe_before_the_next_call():
    mon = ScriptedMonitor(snap(), snap(), snap(), snap(), snap(util=99.0), snap(util=99.0), snap(util=5.0), snap(util=5.0), snap(util=5.0))
    g, _ = make_guard(mon, idle_probe_s=0.0)
    for _ in range(4):
        g.before_call()
        g.note_call_finished(3.0)
    g.note_call_finished(40.0)  # one call far slower than the recent ones
    waited = g.before_call()
    assert waited > 0 and g.slow_call_probes >= 1 and g.reasons.get("gpu_busy_other_process") == 1


# ------------------------------------------------------------------ headless adapter integration
class TimeoutOnce(FakeOllama):
    calls = 0

    def chat(self, **kw):
        TimeoutOnce.calls += 1
        if TimeoutOnce.calls == 1:
            raise OllamaRequestError("timeout after 120.0s calling local Ollama /api/chat")
        r = super().chat(**kw)
        r["token_metadata"] = {"prompt_eval_count": 10, "eval_count": 5, "total_duration": int(2e9)}
        return r


def _adapter_with_guard(guard):
    from modules import controller_spec as cs

    s = cs.ControllerSpec.from_value({"controller_type": "local_ollama", "settings": {"model": "model-a", "max_retries": 1}}).full_settings()
    return ollama_pilot.OllamaPilotAdapter(s, agent_id="engineer_1", role="Engineer", guard_provider=lambda: guard)


@pytest.fixture
def request_obj():
    st = _sim({"Engineer": OLLAMA_A})
    ag = next(a for a in st.agents if a.role == "Engineer")
    ctx = st.brain_context_builder.build(st, ag)
    req = ag._build_brain_request(st, ctx, False, "t", request_id="r1")
    yield st, req
    _stop(st)


def test_timeout_explained_by_contention_is_waited_out_and_not_a_controller_failure(request_obj):
    st, req = request_obj
    TimeoutOnce.calls = 0
    mon = ScriptedMonitor(snap(), snap(util=99.0), snap(util=99.0), snap(util=5.0), snap(util=5.0), snap(util=5.0))
    g, _ = make_guard(mon, idle_probe_s=0.0)
    ad = _adapter_with_guard(g)
    ad.client = TimeoutOnce()
    resp = ad.choose_action(req)
    assert resp is not None and ad.last_trace["validation_outcome"] == "valid"
    assert ad.last_trace["retry_count"] == 0 and ad.last_trace.get("resource_timeout_retries") == 1  # did not consume the model-retry budget
    first = ad.last_trace["attempts"][0]
    assert first["error_class"] == "transport_timeout_resource_contention" and first["resource_reason"] == "gpu_busy_other_process"
    assert ad.last_trace["resource_wait_s"] > 0
    last = ad.last_trace["attempts"][-1]
    assert last["model_inference_s"] == 2.0 and last["transport_overhead_s"] == 0.0  # model inference and transport/infrastructure time recorded separately


def test_timeout_without_resource_explanation_stays_a_transport_failure_class(request_obj):
    st, req = request_obj

    class AlwaysTimeout(FakeOllama):
        def chat(self, **kw):
            raise OllamaRequestError("timeout after 120.0s calling local Ollama /api/chat")

    mon = ScriptedMonitor(snap())
    g, _ = make_guard(mon, idle_probe_s=0.0)
    ad = _adapter_with_guard(g)
    ad.client = AlwaysTimeout()
    with pytest.raises(ollama_pilot.ControllerFailure) as exc:
        ad.choose_action(req)
    assert exc.value.trace["failure_class"] == "transport_timeout"


@pytest.mark.parametrize("raw,expected", [("not json at all", "malformed_output"), ('{"affordance_id": "a999"}', "illegal_action"), ('{"affordance_id": "a0", "arguments": {"structure_type": "castle"}}', "illegal_action")])
def test_model_output_failures_are_classified_apart_from_transport(request_obj, raw, expected):
    st, req = request_obj
    g, _ = make_guard(ScriptedMonitor(snap()), idle_probe_s=0.0)
    ad = _adapter_with_guard(g)
    ad.client = FakeOllama(reply=lambda m, o, mod: raw)
    with pytest.raises(ollama_pilot.ControllerFailure) as exc:
        ad.choose_action(req)
    assert exc.value.trace["failure_class"] == expected


# ------------------------------------------------------------------ whole headless run: protection active, time does not advance while throttled
def test_headless_run_uses_the_guard_waits_without_advancing_time_and_records_it_in_the_manifest(monkeypatch):
    import modules.resource_guard as rg

    state = {"calls": 0}
    real_check = rg.ResourceGuard.check

    def scripted_check(self, *, fresh=True, util_check=False):
        state["calls"] += 1
        # the machine is busy for the first few checks of the run, then free
        busy = state["calls"] in range(2, 6)
        return real_check.__get__(self)(fresh=fresh, util_check=util_check) if not busy else ("gpu_vram_headroom_lost", {"gpu_free_mb": 100.0, "gpu_total_mb": 8188.0})

    monkeypatch.setattr(rg.ResourceGuard, "check", scripted_check)
    monkeypatch.setattr(rg.time, "sleep", lambda s: None)
    st = _sim({"Engineer": OLLAMA_A})
    events = []
    st.logger.register_event_listener(lambda e: events.append((e["event_type"], e["time"], e["payload_data"])))
    try:
        assert st.async_inference is False and st.resource_settings["resource_mode"] == "conservative" and st.resource_settings["max_concurrent_inference"] == 1
        _run(st, 25)
        waits = [e for e in events if e[0] == "inference_resource_wait_started"]
        ends = [e for e in events if e[0] == "inference_resource_wait_ended"]
        assert waits and ends and waits[0][1] == ends[0][1]  # same simulated instant: waiting did not advance simulated time
        assert st.time > 5 and st.controller_error is None
    finally:
        _stop(st)
    manifest = json.loads((Path(st.logger.output_session.session_folder) / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resource_guard"]["active"] and manifest["resource_guard"]["throttle_events"] >= 1
    assert manifest["resource_protection"] == {"inference_path": "synchronous_headless", "guard_active": True, "max_concurrent_inference": 1, "resource_mode": "conservative"}
    rows = [json.loads(l) for l in (Path(st.logger.output_session.session_folder) / "logs" / "controller_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r.get("resource_wait_s") is not None for r in rows if r["kind"] == "ollama_decision")


def test_procedural_only_runs_never_create_a_guard():
    with tempfile.TemporaryDirectory() as tmp:
        from modules.simulation import SimulationState

        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="proc", seed=3)
        for _ in range(5):
            st.update(1.0)
        assert st._resource_guard is None and st.resource_monitor is None
        st.stop()
        manifest = json.loads((Path(st.logger.output_session.session_folder) / "session_manifest.json").read_text(encoding="utf-8"))
        assert manifest["resource_guard"] is None and manifest["resource_protection"]["guard_active"] is False


# ------------------------------------------------------------------ GUI governor shares the same protective core
def test_governor_uses_the_same_guard_for_gpu_not_only_ram():
    g, _ = make_guard(ScriptedMonitor(snap(used=8100.0)))
    gov = InferenceGovernor(1, 2048.0, monitor=ScriptedMonitor(snap(used=8100.0)), guard=g)
    text = gov._resource_block()
    assert text and "gpu_vram_headroom_lost" in text
    gov2 = InferenceGovernor(1, 2048.0, monitor=ScriptedMonitor(snap()), guard=g)
    assert gov2._resource_block() is None
    gov.close()
    gov2.close()


# ------------------------------------------------------------------ real-hardware regression: the model's own runtime buffers are not "contention"
REAL_MODEL = [{"name": "qwen3.5:9b", "size_bytes": 6302707216, "size_vram_bytes": 5523206306}]  # real /api/ps numbers for qwen3.5:9b at 8k context


def test_the_models_own_runtime_buffers_are_never_mistaken_for_contention():
    """Real headless run, qwen3.5:9b at 8k context: idle GPU 14 MB; model loaded 6,440 MB used vs /api/ps size_vram 5,267 MB; the ~1.1 GB difference (and its
    later growth) is Ollama's own KV cache / compute buffers. Growth heuristics throttled the run forever; only direct unsafe conditions may throttle."""
    g, _ = make_guard(None)
    for used in (14.0, 5300.0, 6440.0, 6440.0 + 900.0):  # idle, warm-up weights only, after first inference, after larger prompts
        reason, d = g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(used), "ollama_loaded_detail": REAL_MODEL if used > 5000 else []})
        assert reason is None, (used, reason, d)
    # a neighbour taking the headroom IS unsafe, and a busy GPU while we are idle IS unsafe
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(8100.0), "ollama_loaded_detail": REAL_MODEL})[0] == "gpu_vram_headroom_lost"
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(6440.0, util=97.0), "ollama_loaded_detail": REAL_MODEL}, util_check=True)[0] == "gpu_busy_other_process"


def test_a_reloaded_instance_at_a_larger_context_is_not_a_residency_drop_but_a_real_drop_still_is():
    """Warm-up loaded the model fully on the GPU (default context); the first real call reloaded it at 8k context: 87.6% on GPU, steady. Not a drop."""
    g, _ = make_guard(None)
    warm = [{"name": "m", "size_bytes": int(5251.7 * MB), "size_vram_bytes": int(5251.7 * MB)}]
    big = [{"name": "m", "size_bytes": 6302707216, "size_vram_bytes": 5523206306}]
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(6000.0), "ollama_loaded_detail": warm})[0] is None
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(6431.0), "ollama_loaded_detail": big})[0] is None  # new instance size: reference restarts
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(6431.0), "ollama_loaded_detail": big})[0] is None
    shrunk = [{"name": "m", "size_bytes": 6302707216, "size_vram_bytes": int(4300 * MB)}]  # the SAME instance loses GPU share (pushed to CPU by a neighbour)
    assert g.evaluate({"ram_available_mb": 16000.0, "gpu": gpu(5200.0), "ollama_loaded_detail": shrunk})[0] == "model_vram_residency_dropped"


def test_warm_up_loads_the_instance_with_the_runs_context_window():
    seen = {}

    class W:
        def __init__(self, *a, **k):
            pass

        def warm(self, *, model, keep_alive="30m", timeout_s=None, num_ctx=None):
            seen.update(model=model, num_ctx=num_ctx)
            return {"latency_s": 0.1, "load_duration_s": 0.1, "done_reason": "load"}

    import modules.ollama_pilot as op
    old = op.client_factory
    op.client_factory = lambda b, t: W()
    try:
        out = op.warm_up({"model": "m", "base_url": "http://127.0.0.1:11434", "max_context_tokens": 8192, "timeout_s": 120.0})
    finally:
        op.client_factory = old
    assert out["warmed"] and seen == {"model": "m", "num_ctx": 8192}
