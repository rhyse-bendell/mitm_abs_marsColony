from pathlib import Path


def test_analysis_gui_launcher_points_to_analysis_gui():
    launcher = Path("launch_analysis_gui.bat")
    assert launcher.exists()
    content = launcher.read_text(encoding="utf-8")
    assert "analysis_gui.py" in content
