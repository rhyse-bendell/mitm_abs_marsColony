"""Lightweight, read-only host telemetry for local-model runs. Diagnostic only: every probe is best-effort and failure-tolerant.

The only process ever started is nvidia-smi (optional, short timeout, read-only query). Nothing here terminates, unloads or
manipulates any other process or application."""
import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time

MB = 1024 * 1024


def _windows_memory():
    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMSTAT()
    st.dwLength = ctypes.sizeof(MEMSTAT)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return st.ullTotalPhys / MB, st.ullAvailPhys / MB


def _windows_process_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / MB


class _WindowsCpu:
    """System CPU percent between two consecutive samples of GetSystemTimes."""

    def __init__(self):
        self._prev = None

    @staticmethod
    def _times():
        idle, kern, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        return idle.value, kern.value + user.value

    def sample(self):
        idle, total = self._times()
        prev, self._prev = self._prev, (idle, total)
        if prev is None or total == prev[1]:
            return None
        return round(100.0 * (1.0 - (idle - prev[0]) / (total - prev[1])), 1)


def read_memory():
    """(total_mb, available_mb), or (None, None) when unavailable."""
    try:
        if sys.platform == "win32":
            return _windows_memory()
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = float(v.split()[0]) / 1024.0
        return info["MemTotal"], info.get("MemAvailable", info.get("MemFree"))
    except Exception:  # noqa: BLE001 - telemetry must never fail a run
        return None, None


def read_process_mb():
    try:
        if sys.platform == "win32":
            return _windows_process_mb()
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MB
    except Exception:  # noqa: BLE001
        return None


def read_gpu(timeout_s=3.0):
    """NVIDIA VRAM via nvidia-smi if present (dict with name/total_mb/used_mb/free_mb), else None. Never required."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=timeout_s, check=True).stdout.strip().splitlines()[0]
        name, total, used, free, util = [p.strip() for p in out.split(",")]
        try:
            util_percent = float(util)
        except ValueError:  # "[N/A]" on some drivers
            util_percent = None
        return {"name": name, "total_mb": float(total), "used_mb": float(used), "free_mb": float(free), "util_percent": util_percent}
    except Exception:  # noqa: BLE001
        return None


class HostResourceMonitor:
    """Low-frequency sampler. snapshot() returns the cached sample (never probes); refresh() probes once. start() refreshes on a daemon
    thread every interval_s seconds, so neither the Tk thread nor the simulation tick ever waits on a probe."""

    def __init__(self, interval_s=3.0, ollama_ps=None, gpu_reader=None, memory_reader=None, process_reader=None, ollama_detail=None):
        self.interval_s = max(1.0, float(interval_s))
        self._ollama_ps = ollama_ps  # callable -> list of loaded model names, or None when unavailable
        self._ollama_detail = ollama_detail  # callable -> list of {name, size_bytes, size_vram_bytes}, or None (used by the resource guard)
        # readers are looked up at call time so tests can substitute them without touching real hardware
        self._gpu_reader = gpu_reader or (lambda: read_gpu())
        self._memory_reader = memory_reader or (lambda: read_memory())
        self._process_reader = process_reader or (lambda: read_process_mb())
        self._cpu = _WindowsCpu() if sys.platform == "win32" else None
        self._lock = threading.Lock()
        self._snapshot = {}
        self._stop = threading.Event()
        self._thread = None

    def refresh(self):
        snap = {"sampled_at": time.time()}
        try:
            snap["ram_total_mb"], snap["ram_available_mb"] = self._memory_reader()
        except Exception:  # noqa: BLE001 - one failing probe never breaks another or the run
            snap["ram_total_mb"] = snap["ram_available_mb"] = None
        for key, fn in (("process_mb", self._process_reader), ("gpu", self._gpu_reader), ("ollama_loaded_models", self._ollama_ps), ("ollama_loaded_detail", self._ollama_detail)):
            try:
                snap[key] = fn() if fn else None
            except Exception:  # noqa: BLE001
                snap[key] = None
        try:
            if self._cpu is not None:
                snap["cpu_percent"] = self._cpu.sample()
            else:
                snap["cpu_percent"] = round(os.getloadavg()[0] / (os.cpu_count() or 1) * 100.0, 1) if hasattr(os, "getloadavg") else None
        except Exception:  # noqa: BLE001
            snap["cpu_percent"] = None
        with self._lock:
            self._snapshot = snap
        return snap

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def start(self):
        if self._thread is not None:
            return
        self.refresh()

        def loop():
            while not self._stop.wait(self.interval_s):
                self.refresh()

        self._thread = threading.Thread(target=loop, name="host-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


def format_snapshot(snap, *, queued=0, active=0, mode="conservative", throttled_reason=None, ollama_connected=None):
    """Field -> display text for the read-only GUI resource panel."""
    def gb(v):
        return "unknown" if v is None else f"{v / 1024:.1f} GB"

    gpu = snap.get("gpu")
    models = snap.get("ollama_loaded_models")
    if ollama_connected is None:
        ollama_connected = models is not None
    return {
        "mode": mode,
        "ollama": "connected" if ollama_connected else "unavailable",
        "loaded_models": "unknown" if models is None else (", ".join(models) if models else "none loaded"),
        "jobs": f"{active} active / {queued} queued",
        "ram": f"{gb(snap.get('ram_available_mb'))} available of {gb(snap.get('ram_total_mb'))}",
        "process": "unknown" if snap.get("process_mb") is None else f"{snap['process_mb'] / 1024:.2f} GB",
        "cpu": "unknown" if snap.get("cpu_percent") is None else f"{snap['cpu_percent']:.0f}%",
        "gpu": "GPU telemetry unavailable" if not gpu else f"{gpu['free_mb'] / 1024:.1f} GB free of {gpu['total_mb'] / 1024:.1f} GB ({gpu['name']})",
        "throttle": throttled_reason or "",
    }
