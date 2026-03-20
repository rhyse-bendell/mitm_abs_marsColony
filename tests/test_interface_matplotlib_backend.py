from pathlib import Path


def test_interface_avoids_pyplot_in_tk_path():
    content = Path("interface.py").read_text(encoding="utf-8")

    assert "import matplotlib.pyplot" not in content
    assert "plt." not in content
    assert "from matplotlib.figure import Figure" in content
    assert "from matplotlib.patches import Circle, Rectangle" in content
