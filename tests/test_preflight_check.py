import importlib.util
from pathlib import Path
import sys
import types


def _load_preflight_module():
    module_path = Path("scripts/preflight_check.py")
    spec = importlib.util.spec_from_file_location("preflight_check", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_reports_missing_required_module(monkeypatch):
    preflight = _load_preflight_module()

    def fake_import(name):
        if name == "matplotlib":
            return None, RuntimeError("missing matplotlib")
        if name == "numpy":
            return types.SimpleNamespace(__version__="1.26.4"), None
        if name == "tkinter":
            return types.SimpleNamespace(), None
        if name == "pathfinding":
            return types.SimpleNamespace(), None
        return None, ImportError(name)

    monkeypatch.setattr(preflight, "_safe_import", fake_import)
    monkeypatch.setattr(preflight, "REQUIREMENTS_PATH", Path("requirements.txt"))

    report = preflight.check_environment()

    assert report.has_errors
    assert any("matplotlib" in msg.message and msg.repairable for msg in report.messages)


def test_preflight_reports_healthy_environment(monkeypatch, tmp_path):
    preflight = _load_preflight_module()
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy<2\n", encoding="utf-8")

    matplotlib_mod = types.SimpleNamespace(get_backend=lambda: "TkAgg")

    def fake_import(name):
        mapping = {
            "tkinter": types.SimpleNamespace(),
            "numpy": types.SimpleNamespace(__version__="1.26.4"),
            "matplotlib": matplotlib_mod,
            "pathfinding": types.SimpleNamespace(),
        }
        if name in mapping:
            return mapping[name], None
        return None, ImportError(name)

    monkeypatch.setattr(preflight, "_safe_import", fake_import)
    monkeypatch.setattr(preflight, "REQUIREMENTS_PATH", req_file)

    report = preflight.check_environment()

    assert not report.has_errors
    assert not any(msg.level == "warning" for msg in report.messages)


def test_run_repair_invokes_expected_pip_command(monkeypatch, tmp_path):
    preflight = _load_preflight_module()
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy<2\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "REQUIREMENTS_PATH", req_file)

    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    rc = preflight.run_repair(python_executable="python3")

    assert rc == 0
    assert captured["check"] is False
    assert captured["command"] == preflight.build_repair_command(
        python_executable="python3", requirements_path=req_file
    )
