# File: modules/data_browser.py
"""Read-only discovery and preview helpers for the GUI Data tab.

Pure functions (no Tk, no simulation access) so they can be unit tested. Paths come from the
OutputSessionManager the logger already owns; nothing here creates, edits or deletes files.
"""

import csv
import json
import time
from pathlib import Path

CSV_PREVIEW_MAX_ROWS = 500
TEXT_PREVIEW_MAX_BYTES = 200_000
JSONL_PREVIEW_MAX_LINES = 500
SNAPSHOT_FILE_LIMIT = 50  # snapshots can be numerous; list only a bounded number

KEY_FILES = (
    "session_manifest.json",
    "measures/run_summary.json",
    "measures/team_summary.json",
    "measures/final_measures.json",
    "logs/events.csv",
    "logs/interaction_trace.jsonl",
)
PREVIEWABLE = {".txt", ".log", ".md"}


def session_paths(output_session):
    """Standard path set for an OutputSessionManager-like object (None if there is no session)."""
    folder = getattr(output_session, "session_folder", None)
    if not folder:
        return None
    folder = Path(folder)
    return {
        "session": folder,
        "outputs_root": Path(getattr(output_session, "outputs_root", folder.parent)),
        "logs": Path(getattr(output_session, "logs_dir", folder / "logs")),
        "measures": Path(getattr(output_session, "measures_dir", folder / "measures")),
    }


def discover_session_files(session_folder):
    """Recognised files as (relative label, absolute Path, is_key); key files first."""
    if not session_folder:
        return []
    root = Path(session_folder)
    if not root.is_dir():
        return []
    found = {}
    for rel in KEY_FILES:
        path = root / rel
        if path.is_file():
            found[rel] = (rel, path, True)
    snapshots = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in found:
            continue
        if rel.startswith("snapshots/"):
            snapshots += 1
            if snapshots > SNAPSHOT_FILE_LIMIT:
                continue
        found[rel] = (rel, path, False)
    return [v for v in found.values() if v[2]] + [v for v in found.values() if not v[2]]


def file_metadata(path):
    path = Path(path)
    try:
        st = path.stat()
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        return f"Path: {path}\nSize: {st.st_size:,} bytes\nModified: {mtime}"
    except OSError as exc:
        return f"Path: {path}\nNot available: {exc}"


def _read_head(path, max_bytes):
    with open(path, "rb") as f:
        raw = f.read(max_bytes + 1)
    return raw[:max_bytes].decode("utf-8", errors="replace"), len(raw) > max_bytes


def preview_csv(path, max_rows=CSV_PREVIEW_MAX_ROWS):
    """Read only the header plus max_rows rows (and one more to detect truncation)."""
    columns, rows, truncated = [], [], False
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                columns = row
                continue
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append(row)
    return {"kind": "csv", "columns": columns, "rows": rows, "truncated": truncated, "max_rows": max_rows}


def preview_file(path):
    """Preview dict for any file. Never raises. kinds: csv, text, unknown, missing, error."""
    path = Path(path)
    meta = file_metadata(path)
    if not path.is_file():
        return {"kind": "missing", "text": meta + "\n\nNot available yet.", "meta": meta}
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            result = preview_csv(path)
            result["meta"] = meta
            return result
        if suffix == ".json":
            text, truncated = _read_head(path, TEXT_PREVIEW_MAX_BYTES)
            if not truncated:
                try:
                    text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
                except ValueError:
                    pass  # mid-write or invalid: show raw
            return {"kind": "text", "text": text, "truncated": truncated, "meta": meta}
        if suffix == ".jsonl":
            lines, truncated = [], False
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if len(lines) >= JSONL_PREVIEW_MAX_LINES:
                        truncated = True
                        break
                    lines.append(line.rstrip("\n"))
            text = "\n".join(lines)
            if len(text) > TEXT_PREVIEW_MAX_BYTES:
                text, truncated = text[:TEXT_PREVIEW_MAX_BYTES], True
            return {"kind": "text", "text": text, "truncated": truncated, "meta": meta}
        if suffix in PREVIEWABLE:
            text, truncated = _read_head(path, TEXT_PREVIEW_MAX_BYTES)
            return {"kind": "text", "text": text, "truncated": truncated, "meta": meta}
    except OSError as exc:
        return {"kind": "error", "text": f"{meta}\n\nCould not read file: {exc}", "meta": meta}
    return {"kind": "unknown", "text": meta + "\n\nNo preview for this file type. Use 'Open Selected File'.", "meta": meta}
