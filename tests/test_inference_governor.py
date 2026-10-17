"""Host-resource protection for local Ollama runs: inference governor (queueing), resource modes, telemetry, soft memory guard,
multi-model warning, configurable Outputs root. Fake clients/monitors only: no real service, no network, no process is touched."""
import re
import threading
import time
from pathlib import Path

import pytest

from modules import disk_usage, host_resources, inference_governor as ig, ollama_pilot
from modules.inference_governor import InferenceGovernor, distinct_local_models, multi_model_warning, resolve_resource_settings
from modules.logging_tools import OUTPUTS_ROOT_ENV, OutputSessionManager, resolve_outputs_root
from tests.test_ollama_controller import FakeOllama, OLLAMA_A, OLLAMA_B, _run, _sim, _stop, _trace_rows, _set, app  # noqa: F401

@pytest.fixture(autouse=True)
def _fake_service(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t))


THREE = {"Architect": OLLAMA_A, "Botanist": OLLAMA_A, "Engineer": OLLAMA_A}


class CountingOllama(FakeOllama):
    """Blocks inside chat() until released and counts how many calls are inside chat() at once."""
    lock = threading.Lock()
    inside = 0
    peak = 0
    gate = threading.Event()
    entered = threading.Semaphore(0)
    order = []

    def chat(self, **kw):
        cls = CountingOllama
        with cls.lock:
            cls.inside += 1
            cls.peak = max(cls.peak, cls.inside)
            cls.order.append(self.base_url)
        cls.entered.release()
        assert cls.gate.wait(15), "test gate never released"
        try:
            return super().chat(**kw)
        finally:
            with cls.lock:
                cls.inside -= 1


@pytest.fixture
def counting(monkeypatch):
    CountingOllama.inside = CountingOllama.peak = 0
    CountingOllama.gate = threading.Event()
    CountingOllama.entered = threading.Semaphore(0)
    CountingOllama.order = []
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: CountingOllama(b, t))
    yield CountingOllama
    CountingOllama.gate.set()


class FakeMonitor:
    def __init__(self, ram_available_mb=16000.0):
        self.ram = ram_available_mb

    def snapshot(self):
        return {"ram_available_mb": self.ram, "ram_total_mb": 32000.0}


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _drive(st, sim_seconds, wall_s=30.0):
    """Tick like the GUI does (update + short sleep) until the simulation has really advanced `sim_seconds`, or wall time runs out."""
    import contextlib
    import io

    end = time.time() + wall_s
    with contextlib.redirect_stdout(io.StringIO()):
        while st.time < sim_seconds and time.time() < end:
            st.update(1.0)
            time.sleep(0.005)


def _async_sim(assign, **kw):
    st = _sim(assign, **kw)
    st.async_inference = True
    return st


# ------------------------------------------------------------------ settings / modes
def test_default_concurrency_is_one_and_modes_resolve_without_hardware_inference():
    assert resolve_resource_settings()["max_concurrent_inference"] == 1
    assert resolve_resource_settings("conservative")["max_concurrent_inference"] == 1
    assert resolve_resource_settings("balanced")["max_concurrent_inference"] == 2
    custom = resolve_resource_settings("custom", 3, 4096)
    assert custom["max_concurrent_inference"] == 3 and custom["memory_floor_mb"] == 4096
    assert resolve_resource_settings("balanced", 9, 1)["max_concurrent_inference"] == 2  # explicit numbers only honoured in Custom
    for bad in ("turbo", ""):
        if bad:
            with pytest.raises(ValueError):
                resolve_resource_settings(bad)
    with pytest.raises(ValueError):
        resolve_resource_settings("custom", 0)
    st = _sim({})
    assert st.resource_settings["resource_mode"] == "conservative" and st.resource_settings["max_concurrent_inference"] == 1


# ------------------------------------------------------------------ queueing
def test_three_ollama_agents_queue_instead_of_three_simultaneous_calls(counting):
    st = _async_sim(THREE)
    for _ in range(3):
        st.update(1.0)
    gov = st.inference_governor()
    assert gov.max_concurrent == 1
    assert counting.entered.acquire(timeout=5)
    assert gov.active_count == 1 and gov.queue_depth == 2, "exactly one call runs; the other two agents wait in the queue"
    statuses = sorted(st.get_agent_brain_runtime(a)["adapter"].runtime_status for a in st.agents)
    assert statuses == ["queued", "queued", "thinking"]
    frozen = st.time
    for _ in range(5):
        st.update(1.0)
    assert st.time == frozen, "queue wait must not advance simulated time"
    counting.gate.set()
    _drive(st, frozen + 3)
    assert st.time >= frozen + 3
    assert counting.peak == 1, "never more than one concurrent Ollama call in Conservative mode"
    assert gov.peak_active == 1
    _stop(st)


def test_queued_decisions_keep_agent_identity_fifo_and_log_queue_time_separately(counting):
    counting.gate.set()
    st = _async_sim(THREE)
    _drive(st, 25)
    rows = [r for r in _trace_rows(st) if r.get("kind") == "ollama_decision" and not r.get("controller_failure")]
    assert rows, "decisions were recorded"
    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent_id"], []).append(r)
        assert r["request_id"].startswith(r["agent_id"]), "a queued decision stays bound to the agent that asked for it"
    assert len(by_agent) == 3, "every agent eventually decided"
    assert all("queue_wait_s" in r and "queue_depth_at_enqueue" in r for r in rows), "queue wait / depth logged on each decision"
    stats = st.inference_governor_stats()
    assert stats["peak_active"] == 1 and stats["completed"] >= 3 and stats["total_inference_s"] >= 0 and stats["max_queue_wait_s"] >= 0
    # first requester ran first (FIFO): the earliest-enqueued job has depth 0
    assert min(r["queue_depth_at_enqueue"] for r in rows) == 0
    _stop(st)


def test_stop_discards_queued_and_pending_work_cleanly(counting):
    st = _async_sim(THREE)
    for _ in range(3):
        st.update(1.0)
    gov = st.inference_governor()
    assert counting.entered.acquire(timeout=5)
    queued = [a._llm_job["future"] for a in st.agents if a._llm_job["future"].state == "queued"]
    assert len(queued) == 2
    _stop(st)
    assert gov.closed and gov.queue_depth == 0
    assert all(f.future.cancelled() for f in queued), "never-started jobs are cancelled, not run"
    assert all(a._llm_job is None for a in st.agents)
    t = st.time
    st.update(1.0)
    assert st.time == t, "a stopped run never advances"
    counting.gate.set()
    time.sleep(0.2)
    assert counting.peak == 1 and len(counting.order) == 1, "no queued call was launched after Stop"


def test_pause_does_not_launch_queued_work(counting):
    st = _async_sim(THREE)
    st.set_inference_paused(True)
    for _ in range(6):
        st.update(1.0)
    gov = st.inference_governor()
    assert gov.active_count == 0 and gov.queue_depth == 3 and counting.order == []
    st.set_inference_paused(False)
    st.update(1.0)
    assert counting.entered.acquire(timeout=5) and gov.active_count == 1
    counting.gate.set()
    _stop(st)


# ------------------------------------------------------------------ telemetry / guards
def test_telemetry_failure_never_crashes_and_gpu_absence_is_reported():
    def boom():
        raise RuntimeError("probe failed")

    mon = host_resources.HostResourceMonitor(gpu_reader=boom, memory_reader=boom, process_reader=boom, ollama_ps=boom)
    snap = mon.refresh()
    assert snap["ram_available_mb"] is None and snap["gpu"] is None and snap["ollama_loaded_models"] is None
    text = host_resources.format_snapshot(snap)
    assert text["gpu"] == "GPU telemetry unavailable" and text["ram"].startswith("unknown")
    gov = InferenceGovernor(1, 2048, monitor=mon)  # unknown memory must not block work
    job = gov.submit(lambda: 7)
    gov.pump()
    assert job.result(5) == 7
    gov.close()


def test_real_probes_return_plausible_values_or_none():
    total, avail = host_resources.read_memory()
    assert (total is None and avail is None) or (total > 0 and 0 <= avail <= total)
    assert host_resources.read_process_mb() is None or host_resources.read_process_mb() > 0
    gpu = host_resources.read_gpu()
    assert gpu is None or {"name", "total_mb", "used_mb", "free_mb"} <= set(gpu)


def test_low_memory_guard_throttles_rather_than_launching_inference():
    events = []
    mon = FakeMonitor(ram_available_mb=500.0)
    gov = InferenceGovernor(1, 2048, monitor=mon, on_event=lambda n, p: events.append(n))
    started = []
    job = gov.submit(lambda: started.append(1) or "done")
    for _ in range(5):
        gov.pump()
    assert started == [] and gov.active_count == 0 and gov.queue_depth == 1
    assert "low memory" in gov.throttled_reason and events == ["inference_throttled"], "one visible event, not one per recheck"
    mon.ram = 8000.0  # periodic recheck: resources recovered
    gov.pump()
    assert job.result(5) == "done" and gov.throttled_reason is None and "inference_throttle_cleared" in events
    gov.close()


def test_throttled_run_can_still_be_stopped_and_shows_status():
    st = _async_sim({"Engineer": OLLAMA_A})
    st.resource_monitor = FakeMonitor(ram_available_mb=100.0)
    for _ in range(4):
        st.update(1.0)
    gov = st.inference_governor()
    assert gov.active_count == 0 and gov.queue_depth == 1
    assert "low memory" in st.resource_status()["throttle"]
    _stop(st)
    assert gov.closed and gov.queue_depth == 0


# ------------------------------------------------------------------ model warning
def test_multiple_distinct_models_produce_a_warning_but_are_not_forbidden():
    a = {"Architect": OLLAMA_A, "Engineer": OLLAMA_B}
    assert distinct_local_models(a) == [("http://127.0.0.1:11434", "model-a"), ("http://127.0.0.1:11434", "model-b")]
    warning = multi_model_warning(a)
    assert warning and "model-a" in warning and "model-b" in warning and "swapping" in warning
    assert multi_model_warning({"Architect": OLLAMA_A, "Engineer": OLLAMA_A}) is None
    assert multi_model_warning({"Architect": {"controller_type": "procedural_baseline", "settings": {}}}) is None


# ------------------------------------------------------------------ isolation guarantees
def test_procedural_only_runs_never_instantiate_governor_monitor_or_client(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Ollama client must not be created for a Procedural-only run")

    monkeypatch.setattr(ollama_pilot, "client_factory", forbidden)
    st = _async_sim({})
    _run(st, 15)
    assert st._inference_governor is None and st.resource_monitor is None and st.inference_governor_stats() is None
    _stop(st)


def test_new_modules_start_no_processes_and_make_no_network_or_kill_calls():
    for mod in (ig, host_resources, disk_usage):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for token in ("taskkill", "os.kill", "terminate(", ".kill(", "psutil", "import socket", "import requests", "urllib", "http.client", "openai", "anthropic"):
            assert token not in src, f"{mod.__name__} must not contain {token!r}"
    calls = re.findall(r"subprocess\.\w+\(\[([^\]]*)\]", Path(host_resources.__file__).read_text(encoding="utf-8"))
    assert len(calls) == 1 and "exe" in calls[0] and "--query-gpu" in calls[0], "only the read-only nvidia-smi query may be spawned"


def test_ollama_client_only_adds_the_read_only_ps_endpoint():
    from modules.ollama_client import ALLOWED_PATHS

    assert set(ALLOWED_PATHS) == {"/api/version", "/api/tags", "/api/chat", "/api/ps"}


# ------------------------------------------------------------------ outputs root / disk usage
def test_outputs_root_default_env_and_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv(OUTPUTS_ROOT_ENV, raising=False)
    root, src = resolve_outputs_root(tmp_path)
    assert root == tmp_path / "Outputs" and src == "default"
    monkeypatch.setenv(OUTPUTS_ROOT_ENV, str(tmp_path / "elsewhere"))
    root, src = resolve_outputs_root(tmp_path)
    assert root == tmp_path / "elsewhere" and src.startswith("env:")
    root, src = resolve_outputs_root(tmp_path, tmp_path / "explicit")
    assert root == tmp_path / "explicit" and src == "explicit"


def test_session_and_manifest_use_the_configured_outputs_root(tmp_path, monkeypatch):
    monkeypatch.delenv(OUTPUTS_ROOT_ENV, raising=False)
    other = tmp_path / "bigdrive"
    st = _sim({}, outputs_root=str(other))
    assert Path(st.logger.output_session.session_folder).parent == other
    manifest = Path(st.logger.output_session.session_folder) / "session_manifest.json"
    _stop(st)
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["outputs_root"] == str(other) and data["outputs_root_source"] == "explicit"
    assert data["resource_settings"]["max_concurrent_inference"] == 1 and "inference_governor" in data
    default = OutputSessionManager("x", project_root=tmp_path)
    assert default.outputs_root == tmp_path / "Outputs" and default.outputs_root_source == "default"


def test_disk_usage_reports_sizes_without_touching_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.bin").write_bytes(b"x" * 2048)
    (tmp_path / "g.txt").write_bytes(b"y" * 1024)
    assert disk_usage.folder_size_bytes(tmp_path) == 3072
    assert disk_usage.folder_size_bytes(tmp_path / "missing") == 0
    rep = disk_usage.outputs_report(tmp_path, tmp_path / "a")
    assert rep["total_bytes"] == 3072 and rep["session_bytes"] == 2048 and rep["session_count"] == 1
    assert disk_usage.human_bytes(1536) == "1.5 KB"
    assert (tmp_path / "a" / "f.bin").exists()


# ------------------------------------------------------------------ GUI (one Tk instance: repeated Tk creation is flaky in this venv)
def test_gui_resource_mode_model_warning_status_text_and_panel(app, monkeypatch):
    # resource mode: default Conservative; Balanced = 2; Custom validated
    assert app.resource_mode_var.get() == "Conservative"
    assert app._resource_settings_from_gui()["max_concurrent_inference"] == 1
    app.resource_mode_var.set("Balanced")
    assert app._resource_settings_from_gui()["max_concurrent_inference"] == 2
    app.resource_mode_var.set("Custom")
    app.resource_max_conc_var.set("3")
    assert app._resource_settings_from_gui()["max_concurrent_inference"] == 3
    app.resource_max_conc_var.set("0")
    with pytest.raises(ValueError):
        app._resource_settings_from_gui()
    app.resource_mode_var.set("Conservative")
    # panel works with no Ollama agent selected
    app._update_resource_panel()
    assert "no agent" in app.resource_vars["ollama"].get() and app.resource_throttle_var.get() == ""
    # connection / multi-model messaging
    roles = app.agent_card_order[:3]
    _set(app, roles[0], "Local Ollama", model="model-a")
    assert app.resource_warning_var.get() == ""
    status = app.agent_ollama_vars[roles[0]]["status"].get()
    assert "Local Ollama connected" in status and "0.0-fake" in status and "model-a" in status
    _set(app, roles[1], "Local Ollama", model="model-b")
    assert "model-a" in app.resource_warning_var.get() and "model-b" in app.resource_warning_var.get()
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, available=False))
    app._ollama_status_check(roles[0])
    msg = app.agent_ollama_vars[roles[0]]["status"].get()
    assert "Local Ollama is not running." in msg and "Open the Ollama app" in msg and 'click "Check Ollama / refresh models"' in msg
    app._update_resource_panel()  # Ollama agent selected, service down: still never raises