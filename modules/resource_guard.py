"""Resource guard shared by the GUI governor and the headless (synchronous) Ollama path.

It only ever decides WHETHER a local-model call may start now. It reads host telemetry (RAM, optional nvidia-smi, Ollama /api/ps) and, when
resources are unsafe, makes the caller WAIT (headless: block before the call; GUI: the governor keeps the job queued). Nothing is killed, unloaded or
manipulated, no other process is touched, no fallback controller is consulted, and waiting never advances simulated time.

What counts as unsafe (no hardware numbers are assumed):
  * low_memory                  available system RAM below the configured floor;
  * (non-model VRAM is reported as `other_vram_mb` for diagnosis only; its growth is NOT a reason - see evaluate())
  * gpu_vram_headroom_lost      free VRAM fell below max(512 MB, 4% of total);
  * model_vram_residency_dropped the loaded model's VRAM share fell clearly below the best share seen after a clean load (partly pushed to CPU);
  * gpu_busy_other_process      GPU utilisation is high while WE are idle (probed at start, after a slow call, and while waiting).
GPU checks are skipped (and reported as unavailable) when nvidia-smi is absent; RAM protection still works."""
import threading
import time
from collections import deque

MB = 1024 * 1024
GPU_DELTA_FLOOR_MB = 1024.0
GPU_DELTA_FRACTION = 0.12
HEADROOM_FLOOR_MB = 512.0
HEADROOM_FRACTION = 0.04
RESIDENCY_DROP = 0.05
UTIL_BUSY_PERCENT = 85.0
SLOW_CALL_MIN_S = 8.0
SLOW_CALL_FACTOR = 4.0
MAX_RESOURCE_TIMEOUT_RETRIES = 3


class GuardCancelled(RuntimeError):
    """The run was stopped while the guard was waiting for resources."""


class ResourceGuard:
    def __init__(self, monitor, memory_floor_mb=2048.0, *, on_event=None, clock=time.monotonic, sleep=time.sleep, poll_s=5.0, clear_polls=2, idle_probe_s=1.5, mode="conservative"):
        self.monitor = monitor
        self.memory_floor_mb = float(memory_floor_mb)
        self.mode = mode
        self._on_event = on_event or (lambda *_: None)
        self._clock, self._sleep = clock, sleep
        self.poll_s, self.clear_polls, self.idle_probe_s = float(poll_s), int(clear_polls), float(idle_probe_s)
        self._lock = threading.Lock()
        self.baseline_other_mb = None
        self.best_residency = None
        self._residency_size_mb = None
        self.expected_model_mb = None
        self.gpu_available = None
        self._started = False
        self._slow_suspect = False
        self._durations = deque(maxlen=20)
        self.checks = 0
        self.throttle_events = 0
        self.total_wait_s = 0.0
        self.max_wait_s = 0.0
        self.reasons = {}
        self.slow_call_probes = 0
        self.timeout_resource_retries = 0
        self.last_reason = None
        self.waiting_reason = None  # non-None while a wait is in progress (visible to status displays)

    # ------------------------------------------------------------------ evaluation (pure apart from learning the session baseline)
    def evaluate(self, snap, *, util_check=False):
        """-> (reason or None, detail). `snap` is a HostResourceMonitor snapshot."""
        snap = snap or {}
        detail = {"ram_available_mb": snap.get("ram_available_mb")}
        avail = snap.get("ram_available_mb")
        if avail is not None and avail < self.memory_floor_mb:
            detail["memory_floor_mb"] = self.memory_floor_mb
            return "low_memory", detail
        gpu = snap.get("gpu")
        self.gpu_available = bool(gpu)
        if not gpu:
            return None, detail
        total, used, free = float(gpu["total_mb"]), float(gpu["used_mb"]), float(gpu["free_mb"])
        models = snap.get("ollama_loaded_detail")
        detail.update(gpu_total_mb=total, gpu_used_mb=used, gpu_free_mb=free, gpu_util_percent=gpu.get("util_percent"))
        model_vram = None
        residency = None
        if models is not None:
            model_vram = sum((m.get("size_vram_bytes") or 0) for m in models) / MB
            size = sum((m.get("size_bytes") or 0) for m in models) / MB
            detail.update(model_vram_mb=round(model_vram, 1), model_size_mb=round(size, 1))
            if size > 0:
                residency = model_vram / size
                detail["model_residency"] = round(residency, 3)
                self.expected_model_mb = max(self.expected_model_mb or 0.0, size)
                if self._residency_size_mb is None or abs(size - self._residency_size_mb) > 0.02 * self._residency_size_mb:
                    # A different loaded instance (e.g. the first real call reloaded the model at the run's larger context window): its share on the GPU is
                    # that instance's steady state on this hardware, not a drop. The reference restarts for each instance size.
                    self._residency_size_mb = size
                    self.best_residency = None
        reason = None
        # 1. loss of headroom
        headroom = max(HEADROOM_FLOOR_MB, HEADROOM_FRACTION * total)
        if free < headroom:
            reason = "gpu_vram_headroom_lost"
            detail["headroom_required_mb"] = round(headroom, 1)
        # 2. non-model VRAM is REPORTED, never a trigger on its own. Growth over a baseline cannot tell another application from Ollama's own runtime buffers
        #    (KV cache / compute graph: ~1 GB, allocated lazily after warm-up and growing with prompt size), and per-process VRAM attribution is not available
        #    under the Windows driver model (nvidia-smi reports N/A). Real evidence of contention is the direct criteria: lost headroom (1), partial offload (3),
        #    a busy GPU while we are idle (5) and low RAM. (The first real headless run was throttled forever by the earlier growth heuristic.)
        if model_vram is not None and model_vram > 0:
            other = max(0.0, used - model_vram)
            detail["other_vram_mb"] = round(other, 1)
            if self.baseline_other_mb is None:
                self.baseline_other_mb = other
            else:
                self.baseline_other_mb = min(self.baseline_other_mb, other)  # informational reference for the manifest
        # 3. the loaded model no longer fits where it did
        if reason is None and residency is not None:
            if self.best_residency is not None and residency < self.best_residency - RESIDENCY_DROP:
                reason = "model_vram_residency_dropped"
                detail["best_residency"] = round(self.best_residency, 3)
            elif reason is None:
                self.best_residency = max(self.best_residency or 0.0, residency)
        # 4. not loaded: would the model we have seen fit?
        if reason is None and models is not None and not models and self.expected_model_mb and free < self.expected_model_mb:
            reason = "insufficient_vram_for_model"
            detail["expected_model_mb"] = round(self.expected_model_mb, 1)
        # 5. another process keeps the GPU busy while we are idle
        util = gpu.get("util_percent")
        if reason is None and util_check and util is not None and util >= UTIL_BUSY_PERCENT:
            reason = "gpu_busy_other_process"
        return reason, detail

    def describe(self, reason, detail):
        d = detail or {}
        if reason == "low_memory":
            return f"low memory: {d.get('ram_available_mb', 0) / 1024:.1f} GB available < floor {self.memory_floor_mb / 1024:.1f} GB - inference waiting (Stop to abandon)"
        gpu = ""
        if d.get("gpu_free_mb") is not None:
            gpu = f" (GPU {d['gpu_free_mb'] / 1024:.1f} GB free of {d['gpu_total_mb'] / 1024:.1f} GB" + (f", {d['gpu_util_percent']:.0f}% busy" if d.get("gpu_util_percent") is not None else "") + ")"
        return f"GPU resources not safe: {reason}{gpu} - inference waiting (Stop to abandon)"

    def snapshot(self, fresh=True):
        if self.monitor is None:
            return {}
        return self.monitor.refresh() if fresh else self.monitor.snapshot()

    def check(self, *, fresh=True, util_check=False):
        self.checks += 1
        return self.evaluate(self.snapshot(fresh), util_check=util_check)

    # ------------------------------------------------------------------ call lifecycle (headless / adapter side)
    def note_call_finished(self, latency_s):
        """Record a finished model call; one that is far slower than recent ones arms a GPU-busy probe before the next call."""
        lat = float(latency_s or 0.0)
        with self._lock:
            prior = list(self._durations)
            if len(prior) >= 3:
                med = sorted(prior)[len(prior) // 2]
                if lat >= max(SLOW_CALL_MIN_S, SLOW_CALL_FACTOR * med):
                    self._slow_suspect = True
            elif lat >= 4 * SLOW_CALL_MIN_S:
                self._slow_suspect = True
            self._durations.append(lat)

    def before_call(self, cancel=None):
        """Block (headless) until a model call may start; returns seconds waited. Raises GuardCancelled if the run is stopped while waiting."""
        util_check = False
        if not self._started:
            self._started = True
            self._sleep(self.idle_probe_s)  # let our own warm-up tail leave the GPU before the first idle sample
            util_check = True
        if self._slow_suspect:
            util_check = True
            self.slow_call_probes += 1
        reason, detail = self.check(fresh=True, util_check=util_check)
        if not reason:
            if util_check:
                self._slow_suspect = False
            return 0.0
        return self._wait(reason, detail, cancel)

    def contention_suspected(self):
        """After a transport timeout: does the current resource state explain it? (fresh sample incl. GPU utilisation)"""
        reason, detail = self.check(fresh=True, util_check=True)
        return reason, detail

    def note_timeout_retry(self):
        self.timeout_resource_retries += 1
        self._slow_suspect = True

    def _wait(self, reason, detail, cancel):
        start = self._clock()
        self.throttle_events += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.last_reason = reason
        self.waiting_reason = self.describe(reason, detail)
        self._on_event("inference_resource_wait_started", {"reason": reason, **{k: v for k, v in detail.items() if v is not None}})
        clear = 0
        polls = 0
        try:
            while True:
                slept = 0.0
                while slept < self.poll_s:
                    if cancel is not None and cancel():
                        raise GuardCancelled("run stopped while waiting for resources")
                    step = min(0.25, self.poll_s - slept)
                    self._sleep(step)
                    slept += step
                polls += 1
                reason2, detail2 = self.check(fresh=True, util_check=True)  # we are idle while waiting, so utilisation is the other processes'
                if reason2:
                    clear = 0
                    self.waiting_reason = self.describe(reason2, detail2)
                else:
                    clear += 1
                    if clear >= self.clear_polls:
                        break
        finally:
            waited = max(0.0, self._clock() - start)
            self.total_wait_s += waited
            self.max_wait_s = max(self.max_wait_s, waited)
            self.waiting_reason = None
        self._slow_suspect = False
        self._on_event("inference_resource_wait_ended", {"reason": reason, "waited_s": round(waited, 3), "polls": polls})
        return waited

    # ------------------------------------------------------------------ evidence for the manifest / GUI
    def manifest_state(self):
        return {"active": True, "mode": self.mode, "memory_floor_mb": self.memory_floor_mb, "gpu_telemetry": ("available" if self.gpu_available else ("unavailable" if self.gpu_available is False else "not_sampled")),
                "baseline_non_model_vram_mb": None if self.baseline_other_mb is None else round(self.baseline_other_mb, 1),
                "best_model_vram_residency": None if self.best_residency is None else round(self.best_residency, 3), "checks": self.checks,
                "throttle_events": self.throttle_events, "total_resource_wait_s": round(self.total_wait_s, 3), "max_resource_wait_s": round(self.max_wait_s, 3),
                "throttle_reasons": dict(self.reasons), "slow_call_probes": self.slow_call_probes, "timeout_resource_retries": self.timeout_resource_retries,
                "poll_s": self.poll_s, "clear_polls": self.clear_polls}
