import sys
from types import ModuleType, SimpleNamespace

backend = ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = object
pyplot = ModuleType("matplotlib.pyplot")
pyplot.subplots = lambda *args, **kwargs: (None, None)
sys.modules.setdefault("matplotlib", ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.backends", ModuleType("matplotlib.backends"))
sys.modules["matplotlib.backends.backend_tkagg"] = backend
sys.modules["matplotlib.pyplot"] = pyplot

from analysis_gui import AnalysisGUI


class DummySlider:
    def __init__(self, value=0):
        self.value = value
        self.to = None

    def configure(self, **kwargs):
        self.to = kwargs.get("to", self.to)

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def winfo_exists(self):
        return 1


class BrokenSlider(DummySlider):
    def winfo_exists(self):
        raise RuntimeError("widget destroyed")


def _build_gui(frame_count=3, replay_index=0):
    gui = AnalysisGUI.__new__(AnalysisGUI)
    gui.replay_engine = SimpleNamespace(frames=[object() for _ in range(frame_count)])
    gui.replay_index = replay_index
    gui._interaction_slider_updating = False
    gui.interaction_slider = DummySlider()
    return gui


def test_clamp_replay_index_bounds():
    gui = _build_gui(frame_count=4)
    assert gui._clamp_replay_index(-2) == 0
    assert gui._clamp_replay_index(10) == 3
    assert gui._clamp_replay_index(2) == 2


def test_sync_interaction_slider_updates_only_when_needed():
    gui = _build_gui(frame_count=5, replay_index=3)

    gui._sync_interaction_slider()

    assert gui.interaction_slider.to == 4
    assert gui.interaction_slider.get() == 3

    gui._interaction_slider_updating = False
    gui._sync_interaction_slider()
    assert gui._interaction_slider_updating is False


def test_interaction_slider_callback_ignores_reentrant_call():
    gui = _build_gui(frame_count=5, replay_index=1)
    called = {"refresh": 0, "frame": 0}
    gui.update_interaction_frame_view = lambda: called.__setitem__("refresh", called["refresh"] + 1)
    gui.update_frame_view = lambda: called.__setitem__("frame", called["frame"] + 1)

    gui._interaction_slider_updating = True
    gui.on_interaction_slider("4")

    assert gui.replay_index == 1
    assert called == {"refresh": 0, "frame": 0}


def test_interaction_slider_callback_clamps_and_refreshes_views():
    gui = _build_gui(frame_count=2, replay_index=0)
    called = {"refresh": 0, "frame": 0}
    gui.update_interaction_frame_view = lambda: called.__setitem__("refresh", called["refresh"] + 1)
    gui.update_frame_view = lambda: called.__setitem__("frame", called["frame"] + 1)

    gui.on_interaction_slider("9")

    assert gui.replay_index == 1
    assert called == {"refresh": 1, "frame": 1}


def test_interaction_slider_availability_handles_destroyed_widget():
    gui = _build_gui(frame_count=1)
    gui.interaction_slider = BrokenSlider()

    assert gui._interaction_slider_available() is False
