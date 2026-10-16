"""Data tab: read-only session path/file discovery and preview logic, plus GUI tab wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import data_browser
from modules.logging_tools import OutputSessionManager


def _make_session(tmp_path):
    s = OutputSessionManager("dt", timestamp="20260101_000000", project_root=tmp_path)
    s.setup_session_dirs()
    return s


def test_session_paths_reuse_output_session_manager(tmp_path):
    s = _make_session(tmp_path)
    p = data_browser.session_paths(s)
    assert p["session"] == s.session_folder and p["logs"] == s.logs_dir and p["measures"] == s.measures_dir
    assert p["outputs_root"] == s.outputs_root
    assert data_browser.session_paths(None) is None
    assert data_browser.session_paths(SimpleNamespace()) is None


def test_discovery_lists_key_files_first_and_only_existing(tmp_path):
    s = _make_session(tmp_path)
    assert data_browser.discover_session_files(s.session_folder) == []
    (s.logs_dir / "events.csv").write_text("a,b\n1,2\n")
    (s.logs_dir / "agent_x.csv").write_text("a\n1\n")
    (s.session_folder / "session_manifest.json").write_text("{}")
    (s.measures_dir / "run_summary.json").write_text("{}")
    got = data_browser.discover_session_files(s.session_folder)
    labels = [g[0] for g in got]
    assert "logs/interaction_trace.jsonl" not in labels  # not created yet -> not listed
    keyed = [g[0] for g in got if g[2]]
    assert keyed == ["session_manifest.json", "measures/run_summary.json", "logs/events.csv"]
    assert labels[:3] == keyed and "logs/agent_x.csv" in labels[3:]
    # refresh picks up newly created files
    (s.logs_dir / "interaction_trace.jsonl").write_text("{}\n")
    assert "logs/interaction_trace.jsonl" in [g[0] for g in data_browser.discover_session_files(s.session_folder)]
    assert data_browser.discover_session_files(tmp_path / "nope") == []


def test_csv_preview_is_capped_and_flags_truncation(tmp_path):
    f = tmp_path / "big.csv"
    f.write_text("h1,h2\n" + "".join(f"{i},x\n" for i in range(data_browser.CSV_PREVIEW_MAX_ROWS + 50)))
    pv = data_browser.preview_file(f)
    assert pv["kind"] == "csv" and pv["columns"] == ["h1", "h2"]
    assert len(pv["rows"]) == data_browser.CSV_PREVIEW_MAX_ROWS and pv["truncated"] is True
    small = tmp_path / "small.csv"
    small.write_text("h\n1\n2\n")
    pv = data_browser.preview_file(small)
    assert len(pv["rows"]) == 2 and pv["truncated"] is False


def test_json_jsonl_text_unknown_missing_previews(tmp_path):
    j = tmp_path / "a.json"
    j.write_text('{"a":1,"b":[1,2]}')
    assert '\n  "a": 1' in data_browser.preview_file(j)["text"]  # pretty-printed
    bad = tmp_path / "bad.json"
    bad.write_text('{"half":')  # mid-write: shown raw, no crash
    assert data_browser.preview_file(bad)["text"] == '{"half":'
    jl = tmp_path / "t.jsonl"
    jl.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(data_browser.JSONL_PREVIEW_MAX_LINES + 5)))
    pv = data_browser.preview_file(jl)
    assert pv["truncated"] and len(pv["text"].splitlines()) == data_browser.JSONL_PREVIEW_MAX_LINES
    t = tmp_path / "x.log"
    t.write_text("z" * (data_browser.TEXT_PREVIEW_MAX_BYTES + 10))
    pv = data_browser.preview_file(t)
    assert pv["truncated"] and len(pv["text"]) == data_browser.TEXT_PREVIEW_MAX_BYTES
    u = tmp_path / "x.bin"
    u.write_bytes(b"\x00\x01")
    assert data_browser.preview_file(u)["kind"] == "unknown" and "Path:" in data_browser.preview_file(u)["text"]
    assert data_browser.preview_file(tmp_path / "gone.csv")["kind"] == "missing"


def test_preview_never_modifies_files(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("a\n1\n")
    before = (f.read_bytes(), f.stat().st_mtime_ns)
    data_browser.preview_file(f)
    assert (f.read_bytes(), f.stat().st_mtime_ns) == before


# ---- GUI wiring
tk = pytest.importorskip("tkinter")
import interface  # noqa: E402


@pytest.fixture
def app():
    try:
        a = interface.MarsColonyInterface()
    except Exception as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass


def test_data_tab_exists_and_handles_no_session(app):
    tabs = [app.notebook.tab(t, "text") for t in app.notebook.tabs()]
    assert "Data" in tabs and tabs.index("Data") == tabs.index("Interactions") + 1
    app.refresh_data_tab()
    assert app.data_summary_vars["folder"].get() == "-"
    for key in ("session", "logs", "measures", "file"):
        assert "disabled" in app.data_buttons[key].state()
    app._data_open("session")  # must not raise
    assert "not available" in app.data_status_var.get()


def test_data_tab_follows_session_and_survives_sim_teardown(app, tmp_path):
    s = _make_session(tmp_path)
    (s.logs_dir / "events.csv").write_text("a,b\n1,2\n")
    app.sim = SimpleNamespace(logger=SimpleNamespace(output_session=s), time=12.5)
    app.refresh_data_tab()
    assert app.data_summary_vars["folder"].get() == str(s.session_folder)
    assert app.data_summary_vars["time"].get() == "12.5 s"
    assert "logs/events.csv" in app._data_files
    assert "disabled" not in app.data_buttons["session"].state()
    app.data_file_tree.selection_set("logs/events.csv")
    app.root.update_idletasks()
    app._data_show_preview()
    assert "events.csv" in app.data_preview_note.get()
    app.sim = None  # final files stay reachable after stop
    app.refresh_data_tab()
    assert "logs/events.csv" in app._data_files
