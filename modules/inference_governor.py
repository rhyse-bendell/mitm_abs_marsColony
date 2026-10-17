"""Application-level inference governor for local model calls.

At most max_concurrent model calls are in flight at once; further requests wait in a FIFO queue. The governor never calls a model
itself: it only decides WHEN a queued job may start. Queue wait is wall-clock and never advances simulated time (the simulation barrier
in SimulationState.update holds time while any job is queued or running). A job that never started is discarded on close(): no fallback
controller ever takes its place.

Starting a job is soft-guarded: if available system RAM is below memory_floor_mb, or the pause flag is set, the job stays queued and
the governor reports why. Nothing is killed, unloaded or manipulated."""
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

MODE_CONCURRENCY = {"conservative": 1, "balanced": 2}
DEFAULT_MEMORY_FLOOR_MB = 2048.0


def resolve_resource_settings(mode="conservative", max_concurrent=None, memory_floor_mb=None):
    """Mode -> concrete limits. Only 'custom' honours explicit values; no hardware capacity is inferred."""
    mode = str(mode or "conservative").strip().lower()
    if mode not in {"conservative", "balanced", "custom"}:
        raise ValueError(f"Unknown resource mode: {mode!r}")
    if mode == "custom":
        mc = int(max_concurrent if max_concurrent is not None else 1)
        floor = float(memory_floor_mb if memory_floor_mb is not None else DEFAULT_MEMORY_FLOOR_MB)
    else:
        mc, floor = MODE_CONCURRENCY[mode], DEFAULT_MEMORY_FLOOR_MB
    if mc < 1:
        raise ValueError("max_concurrent_inference must be >= 1")
    return {"resource_mode": mode, "max_concurrent_inference": mc, "memory_floor_mb": max(0.0, floor)}


def distinct_local_models(assignments):
    """Sorted distinct (base_url, model) pairs across agents set to local_ollama. assignments: {role: {controller_type, settings}}."""
    seen = set()
    for spec in (assignments or {}).values():
        if (spec or {}).get("controller_type") != "local_ollama":
            continue
        s = spec.get("settings") or {}
        model = str(s.get("model") or "").strip()
        if model:
            seen.add((str(s.get("base_url") or "http://127.0.0.1:11434"), model))
    return sorted(seen)


def multi_model_warning(assignments):
    """Warning text when more than one distinct local model is selected (informational; never forbids), else None."""
    pairs = distinct_local_models(assignments)
    names = sorted({m for _b, m in pairs})
    if len(names) < 2:
        return None
    return ("Multiple distinct local models are selected: " + ", ".join(names) + ".\n"
            "Several models resident at once can increase RAM/VRAM pressure and cause model-swapping latency. "
            "The run is allowed; nothing is unloaded or deleted automatically.")


OLLAMA_NOT_RUNNING_MESSAGE = ("Local Ollama is not running.\n"
                              "Open the Ollama app, wait for it to start, then click \"Check Ollama / refresh models\".")


class GovernedJob:
    """Future-like handle (done()/result()) with wall-clock timing; created queued, started only by the governor."""

    def __init__(self, fn, label):
        self.fn, self.label = fn, label
        self.future = Future()
        self.enqueued_at = time.perf_counter()
        self.started_at = None
        self.finished_at = None
        self.queue_depth_at_enqueue = 0

    @property
    def state(self):
        if self.future.done():
            return "done"
        return "running" if self.started_at is not None else "queued"

    def done(self):
        return self.future.done()

    def result(self, timeout=None):
        return self.future.result(timeout)

    @property
    def queue_wait_s(self):
        return None if self.started_at is None else self.started_at - self.enqueued_at

    @property
    def inference_s(self):
        return None if self.started_at is None or self.finished_at is None else self.finished_at - self.started_at


class InferenceGovernor:
    def __init__(self, max_concurrent=1, memory_floor_mb=DEFAULT_MEMORY_FLOOR_MB, monitor=None, on_event=None, guard=None):
        self.guard = guard  # shared ResourceGuard (RAM + GPU); None = the original RAM-floor check
        self.max_concurrent = max(1, int(max_concurrent))
        self.memory_floor_mb = float(memory_floor_mb)
        self.monitor = monitor
        self._on_event = on_event or (lambda *_: None)
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent, thread_name_prefix="ollama-infer")
        self._lock = threading.Lock()
        self._queue = deque()
        self._active = 0
        self.peak_active = 0
        self.paused = False
        self.closed = False
        self.throttled_reason = None
        self.completed = []  # per finished job: label, queue_wait_s, inference_s, queue_depth_at_enqueue

    @property
    def active_count(self):
        return self._active

    @property
    def queue_depth(self):
        return len(self._queue)

    def has_work(self):
        return self._active > 0 or bool(self._queue)

    def stats(self):
        done = list(self.completed)
        waits = [c["queue_wait_s"] for c in done]
        return {"max_concurrent": self.max_concurrent, "peak_active": self.peak_active, "completed": len(done), "active": self._active, "queued": len(self._queue),
                "total_queue_wait_s": round(sum(waits), 3), "max_queue_wait_s": round(max(waits), 3) if waits else 0.0,
                "total_inference_s": round(sum(c["inference_s"] for c in done), 3), "throttled_reason": self.throttled_reason}

    def submit(self, fn, label=""):
        job = GovernedJob(fn, label)
        with self._lock:
            job.queue_depth_at_enqueue = len(self._queue) + self._active
            self._queue.append(job)
        return job

    def _resource_block(self):
        snap = self.monitor.snapshot() if self.monitor is not None else {}
        if self.guard is not None:
            reason, detail = self.guard.evaluate(snap, util_check=False)  # cached sample: never probes on the GUI thread
            if reason:
                self.guard.last_reason = reason
                return self.guard.describe(reason, detail)
            return None
        avail = snap.get("ram_available_mb")
        if avail is not None and avail < self.memory_floor_mb:
            return f"low memory: {avail / 1024:.1f} GB available < floor {self.memory_floor_mb / 1024:.1f} GB - inference waiting (Stop to abandon)"
        return None

    def pump(self):
        """Start queued jobs, FIFO, while capacity, the pause flag and the memory floor allow. Called from the simulation tick; never blocks."""
        if self.closed:
            return
        with self._lock:
            if self.paused or not self._queue:
                return
            if self._active < self.max_concurrent:
                reason = self._resource_block()
                if reason:
                    if reason != self.throttled_reason:
                        self._on_event("inference_throttled", {"reason": reason, "queued": len(self._queue), "active": self._active})
                    self.throttled_reason = reason
                    return
                if self.throttled_reason is not None:
                    self._on_event("inference_throttle_cleared", {"queued": len(self._queue)})
                self.throttled_reason = None
            while self._queue and self._active < self.max_concurrent:
                job = self._queue.popleft()
                self._active += 1
                self.peak_active = max(self.peak_active, self._active)
                job.started_at = time.perf_counter()
                self._executor.submit(self._run, job)

    def _run(self, job):
        try:
            job.future.set_result(job.fn())
        except BaseException as exc:  # noqa: BLE001 - delivered to the consumer through the future
            job.future.set_exception(exc)
        finally:
            job.finished_at = time.perf_counter()
            with self._lock:
                self._active -= 1
                self.completed.append({"label": job.label, "queue_wait_s": job.queue_wait_s, "inference_s": job.inference_s, "queue_depth_at_enqueue": job.queue_depth_at_enqueue})

    def close(self):
        """Run is stopping: queued jobs are discarded un-started; running jobs finish in the background and their results are never consumed."""
        self.closed = True
        with self._lock:
            discarded = len(self._queue)
            for job in self._queue:
                job.future.cancel()
            self._queue.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        return discarded
