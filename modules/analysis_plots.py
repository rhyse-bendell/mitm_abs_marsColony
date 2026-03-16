from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from modules.analysis_models import AnalysisSession


PLOT_OPTIONS = {
    "event_counts_over_time": "Event Count Over Time",
    "movement_events_over_time": "Movement Event Count Over Time",
    "planner_outcomes_over_time": "Planner/Brain Outcomes Over Time",
    "fallback_and_witness_over_time": "Fallback + Witness Signals Over Time",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series_from_events(events: List[Dict[str, Any]], filter_prefixes: Tuple[str, ...] = tuple(), include: Tuple[str, ...] = tuple()) -> Tuple[List[float], List[int]]:
    rows = sorted(events, key=lambda e: _to_float(e.get("time")))
    x: List[float] = []
    y: List[int] = []
    count = 0
    for e in rows:
        et = str(e.get("event_type", ""))
        if include and et not in include:
            continue
        if filter_prefixes and not any(et.startswith(p) for p in filter_prefixes):
            continue
        count += 1
        x.append(_to_float(e.get("time")))
        y.append(count)
    return x, y


def build_plot(session: AnalysisSession, plot_key: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    events = session.artifacts.events

    if plot_key == "movement_events_over_time":
        x, y = _series_from_events(events, filter_prefixes=("movement_",))
        ax.plot(x, y, label="movement_*", color="#1f77b4")
    elif plot_key == "planner_outcomes_over_time":
        for key, color in [
            ("planner_", "#1f77b4"),
            ("brain_", "#ff7f0e"),
        ]:
            x, y = _series_from_events(events, filter_prefixes=(key,))
            ax.plot(x, y, label=key.rstrip("_"), color=color)
    elif plot_key == "fallback_and_witness_over_time":
        include = (
            "brain_provider_fallback",
            "witness_expectation_failed",
            "witness_expectation_recovered",
        )
        grouped = Counter(str(e.get("event_type")) for e in events if str(e.get("event_type")) in include)
        ax.bar(list(grouped.keys()), list(grouped.values()), color=["#d62728", "#9467bd", "#2ca02c"])
        ax.set_ylabel("count")
    else:
        x, y = _series_from_events(events)
        ax.plot(x, y, label="all events", color="#1f77b4")

    phase_rows = session.artifacts.phase_summary or []
    for phase in phase_rows:
        t = phase.get("start_time")
        if t is None:
            continue
        ax.axvline(_to_float(t), linestyle="--", alpha=0.35, color="gray")

    ax.set_title(PLOT_OPTIONS.get(plot_key, plot_key))
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative count")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig
