"""Headless Local Ollama + resource guard, END TO END with scripted host telemetry (no real GPU/RAM/service).

The telemetry goes in through the host MONITOR, so the guard's own evaluation decides (nothing about the guard is patched):
    clean -> contention (free VRAM lost) -> still contended -> recovered -> the SAME run resumes and completes its decisions.
Simulated time must not advance while throttled; Stop must interrupt the wait cleanly; the manifest must prove the guard was active; queue/resource
wait is recorded apart from model inference latency; no other controller is ever consulted."""
import json
import threading
import time
from pathlib import Path

import pytest

from modules import ollama_pilot
from tests.test_ollama_controller import FakeOllama, OLLAMA_A, _run, _sim, _stop, _trace_rows
from tests.test_resource_guard import ScriptedMonitor, snap


@pytest.fixture(autouse=True)
def _fake_service(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _guarded_sim(monitor):
    st = _sim({"Engineer": OLLAMA_A})
    st.resource_monitor = monitor  # injected telemetry source; the guard (created lazily) evaluates it with its real logic
    guard = st.resource_guard()
    clock = FakeClock()
    guard._sleep, guard._clock, guard.poll_s, guard.idle_probe_s = clock.sleep, clock.now, 1.0, 0.0
    return st, guard, clock


BUSY = snap(used=8100.0)  # other process holds almost all VRAM: free VRAM far below the headroom floor
CLEAN = snap()


def test_throttle_then_recover_then_the_same_run_resumes_and_decides():
    mon = ScriptedMonitor(BUSY, BUSY, BUSY, CLEAN)  # 3 contended samples, then clean for the rest of the run
    st, guard, clock = _guarded_sim(mon)
    events = []
    st.logger.register_event_listener(lambda e: events.append((e["event_type"], e["time"], e["payload_data"])))
    try:
        assert st.async_inference is False and st.resource_settings["resource_mode"] == "conservative"  # synchronous headless, conservative default
        _run(st, 30)
        starts = [e for e in events if e[0] == "inference_resource_wait_started"]
        ends = [e for e in events if e[0] == "inference_resource_wait_ended"]
        assert starts and ends, "the unsafe state must queue the call BEFORE inference"
        assert starts[0][2]["reason"] == "gpu_vram_headroom_lost"  # decided by the guard's own evaluation of the scripted telemetry
        assert starts[0][1] == ends[0][1]  # same simulated instant: throttling did not advance simulated time
        assert ends[0][2]["waited_s"] > 0 and ends[0][2]["polls"] >= 3  # needed recovery: 2 consecutive clean polls
        t_wait = starts[0][1]
        assert st.time > t_wait + 5 and st.controller_error is None  # the run resumed after recovery and kept going
        rows = [r for r in _trace_rows(st) if r.get("kind") == "ollama_decision"]
        assert rows and all(r["validation_outcome"] == "valid" for r in rows)  # decisions completed, no RuleBrain fallback
        first = rows[0]
        assert first["resource_wait_s"] > 0 and first["wall_clock_inference_s"] >= 0  # resource wait is recorded APART from model inference time
        assert all(r.get("resource_wait_s", 0.0) == 0.0 for r in rows[1:])  # after recovery no further waiting
        assert not [e for e in events if e[0] in {"fallback_bootstrap_action_forced", "agent_backend_hard_demoted"}]
    finally:
        _stop(st)
    manifest = json.loads((Path(st.logger.output_session.session_folder) / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resource_guard"]["active"] and manifest["resource_guard"]["throttle_events"] == 1
    assert manifest["resource_guard"]["throttle_reasons"] == {"gpu_vram_headroom_lost": 1} and manifest["resource_guard"]["total_resource_wait_s"] > 0
    assert manifest["resource_protection"]["inference_path"] == "synchronous_headless" and manifest["resource_protection"]["guard_active"] is True


def test_ram_pressure_throttles_the_same_way_without_any_gpu():
    low_ram = snap(ram=500.0, with_gpu=False)
    clean = snap(with_gpu=False)
    st, guard, clock = _guarded_sim(ScriptedMonitor(low_ram, low_ram, clean))
    events = []
    st.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    try:
        _run(st, 12)
        assert [p["reason"] for n, p in events if n == "inference_resource_wait_started"][:1] == ["low_memory"]
        assert any(n == "inference_resource_wait_ended" for n, _ in events) and st.controller_error is None and st.time >= 12
    finally:
        _stop(st)


def test_stop_during_a_resource_wait_interrupts_cleanly_without_a_controller_error():
    mon = ScriptedMonitor(BUSY)  # never recovers
    st, guard, clock = _guarded_sim(mon)
    guard._sleep = lambda s: time.sleep(0.02)  # real (short) waiting so another thread can act; the telemetry stays contended
    done = threading.Event()
    box = {}

    def worker():
        try:
            _run(st, 5)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + 20
    while guard.waiting_reason is None and time.time() < deadline:
        time.sleep(0.02)
    assert guard.waiting_reason, "the run should be waiting for resources"
    sim_time_while_waiting = st.time
    time.sleep(0.2)
    assert st.time == sim_time_while_waiting  # simulated time does not move while throttled
    _stop(st)  # the experimenter presses Stop
    assert done.wait(15), "Stop must interrupt the wait"
    assert "error" not in box and guard.waiting_reason is None
    assert st.controller_error is None and st.terminal_state != "controller_error"  # a Stop is not a controller failure
    rows = [r for r in _trace_rows(st) if r.get("kind") == "ollama_decision"]
    assert rows and rows[-1].get("cancelled") is True and rows[-1]["validation_outcome"] == "cancelled"
    assert not [r for r in rows if r.get("validation_outcome") == "valid"]  # nothing decided in its place


def test_the_guard_only_observes_and_no_fallback_controller_exists_for_a_throttled_agent():
    mon = ScriptedMonitor(BUSY, CLEAN)
    st, guard, clock = _guarded_sim(mon)
    try:
        _run(st, 8)
        engineer = next(a for a in st.agents if a.role == "Engineer")
        assert engineer.controller_type_id() == "local_ollama"
        runtime = st.get_agent_brain_runtime(engineer)
        assert runtime["config"].fallback_backend == "none" and not runtime.get("hard_demoted")
        assert st.resource_settings["max_concurrent_inference"] == 1
    finally:
        _stop(st)
