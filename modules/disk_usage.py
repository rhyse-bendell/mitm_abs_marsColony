"""Read-only disk-usage helpers for output folders. Reports sizes only: nothing is moved, migrated or deleted."""
import os


def folder_size_bytes(path):
    """Total size of regular files under path (0 if missing). Symlinks are not followed; unreadable entries are skipped."""
    total = 0
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def outputs_report(outputs_root, session_folder=None):
    """{'outputs_root','total_bytes','session_bytes','session_count'} for display on the Data tab (on demand only)."""
    root = str(outputs_root)
    sessions = 0
    try:
        sessions = sum(1 for e in os.scandir(root) if e.is_dir())
    except OSError:
        pass
    return {"outputs_root": root, "total_bytes": folder_size_bytes(root), "session_bytes": folder_size_bytes(session_folder) if session_folder else 0, "session_count": sessions}
