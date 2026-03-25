# File: interface.py

import tkinter as tk
import tkinter.font as tkfont
import queue
import threading
import traceback
import json
import math
import time
import hashlib
import colorsys
from pathlib import Path
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle
from modules.simulation import SimulationState
from tkinter import StringVar, BooleanVar, DoubleVar, IntVar
from modules.construction import ConstructionManager
from modules.phase_definitions import MISSION_PHASES
from modules.task_model import load_task_model, normalize_rule_token
from modules.interaction_graph import CANONICAL_NODES
from modules.headless_runner import run_batch_experiment


class MarsColonyInterface:
    _PILE_DRAW_SIZE = 0.24
    _PILE_OUTSIDE_PADDING = 0.06
    _BRIDGE_SITE_INSET = 0.14
    _STRUCTURE_AREA_RATIO_CAP = 0.05
    _STRUCTURE_ANCHOR_RADIAL_FACTOR = 0.62

    STATE_IDLE = "idle"
    STATE_STARTING = "starting"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_STOPPED = "stopped"
    EXPERIMENT_MAX_CONTENT_WIDTH = 940
    EXPERIMENT_PANEL_BORDER = "#586271"
    BACKEND_DEFAULTS = {
        "brain_backend": "rule_brain",
        "local_model": "qwen2.5:3b-instruct",
        "local_base_url": "http://127.0.0.1:11434",
        "timeout_s": 240.0,
        "fallback_backend": "rule_brain",
    }
    LOCAL_MODEL_SHORTLIST = [
        "qwen2.5:3b-instruct",
        "qwen2.5:7b",
        "qwen2.5:7b-instruct",
        "llama3.2",
        "qwen2.5:3b",
        "gemma3:4b",
        "mistral-small3.1",
        "qwen3.5:9b",
    ]
    PLANNER_DEFAULTS = {
        "planner_interval_steps": 16,
        "planner_timeout_seconds": 240.0,
        "planner_max_retries": 1,
        "backend_timeout_s": 240.0,
        "backend_max_retries": 1,
        "degraded_consecutive_failures_threshold": 24,
        "degraded_cooldown_seconds": 300.0,
        "degraded_step_interval_multiplier": 8.0,
        "enable_startup_llm_sanity": False,
        "startup_llm_sanity_timeout_seconds": 240.0,
        "startup_llm_sanity_max_sources": 1,
        "startup_llm_sanity_max_items_per_type": 2,
        "startup_llm_sanity_completion_max_tokens": 2048,
        "planner_completion_max_tokens": 4096,
        "warmup_timeout_seconds": 180.0,
        "startup_llm_sanity_raw_response_max_chars": 4000,
        "enable_bootstrap_summary_reuse": True,
        "bootstrap_summary_max_chars": 280,
        "high_latency_local_llm_mode": False,
        "unrestricted_local_qwen_mode": False,
        "high_latency_stale_result_grace_s": 1800.0,
        "planner_request_policy": "cadence_with_dik_integration",
        "planning_interval_steps": 60,
        "dik_integration_cooldown_steps": 12,
        "dik_integration_batch_threshold": 2,
    }
    RETRY_HELP_TEXT = {
        "backend_max_retries": "0 means make one attempt and then rely on fallback/degraded behavior instead of retrying immediately.",
        "planner_max_retries": "0 means the planning step is not immediately retried after a failure.",
    }
    MAX_AGENT_PANELS = 6
    DEFAULT_AGENT_COUNT = 3
    DEFAULT_AGENT_IDENTITIES = [
        {"role": "Architect", "display_name": "Architect", "label": "Architect", "template_id": "mars_architect"},
        {"role": "Engineer", "display_name": "Engineer", "label": "Engineer", "template_id": "mars_engineer"},
        {"role": "Botanist", "display_name": "Botanist", "label": "Botanist", "template_id": "mars_botanist"},
        {"role": "Agent 4", "display_name": "Agent 4", "label": "Agent 4", "template_id": None},
        {"role": "Agent 5", "display_name": "Agent 5", "label": "Agent 5", "template_id": None},
        {"role": "Agent 6", "display_name": "Agent 6", "label": "Agent 6", "template_id": None},
    ]
    PLOTTING_INIT_ERROR = (
        "Plot initialization failed.\n\n"
        "This interface expects a Tk-compatible Matplotlib environment.\n"
        "Mixed NumPy/Matplotlib/PySide installations can break plotting imports.\n\n"
        "Run the repository preflight repair command before launching again:\n"
        "  py -3 scripts\\preflight_check.py --repair"
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mars Colony Simulation")

        self.speed_multiplier = DoubleVar(value=1.0)
        self.flash_mode = BooleanVar(value=False)

        self.sim = None
        self.construction_defaults = self._load_construction_defaults()
        self.run_state = self.STATE_IDLE
        self._run_loop_job = None
        self._startup_poll_job = None
        self._startup_worker = None
        self._startup_queue = None
        self._startup_dialog = None
        self._startup_status_var = None
        self._startup_progress = None
        self._startup_progressbar = None
        self._batch_worker = None
        self._batch_queue = None
        self._batch_poll_job = None
        self.run_loop_interval_ms = 100
        self.base_dt = 0.1
        self.wallclock_elapsed_var = StringVar(value="Wall-clock: 00:00:00.0")
        self.sim_time_elapsed_var = StringVar(value="Sim-time: 00:00:00.0")
        self.current_pause_elapsed_var = StringVar(value="Current cognition-pause: none")
        self.cumulative_pause_wait_var = StringVar(value="Cumulative cognition wait: 00:00:00.0")
        self.barrier_summary_var = StringVar(value="Barrier: none")
        self.pause_count_var = StringVar(value="Cognition pauses: 0")
        self.last_barrier_duration_var = StringVar(value="Last barrier: none")
        self.last_planner_latency_var = StringVar(value="Last planner latency: n/a")

        self.create_widgets()
        self._update_control_states()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.control_frame = ttk.Frame(self.root, padding=(8, 6))
        self.control_frame.pack(fill="x")
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_experiment)
        self.run_batch_button = ttk.Button(self.control_frame, text="Run Batch", command=self.run_batch_experiment)
        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_experiment)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_experiment)
        self.lifecycle_label = ttk.Label(self.control_frame, text="State: idle")
        self.observability_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        self.observability_line_primary = ttk.Label(self.observability_frame, textvariable=self.wallclock_elapsed_var)
        self.observability_line_primary_2 = ttk.Label(self.observability_frame, textvariable=self.sim_time_elapsed_var)
        self.observability_line_primary_3 = ttk.Label(self.observability_frame, textvariable=self.current_pause_elapsed_var)
        self.observability_line_primary_4 = ttk.Label(self.observability_frame, textvariable=self.cumulative_pause_wait_var)
        self.observability_line_secondary = ttk.Label(self.observability_frame, textvariable=self.barrier_summary_var)
        self.observability_line_secondary_2 = ttk.Label(self.observability_frame, textvariable=self.pause_count_var)
        self.observability_line_secondary_3 = ttk.Label(self.observability_frame, textvariable=self.last_barrier_duration_var)
        self.observability_line_secondary_4 = ttk.Label(self.observability_frame, textvariable=self.last_planner_latency_var)

        self.start_button.pack(side="left", padx=(0, 6))
        self.run_batch_button.pack(side="left", padx=6)
        self.pause_button.pack(side="left", padx=6)
        self.stop_button.pack(side="left", padx=6)
        self.lifecycle_label.pack(side="right")
        self.observability_frame.pack(fill="x")
        self.observability_line_primary.pack(side="left", padx=(0, 12))
        self.observability_line_primary_2.pack(side="left", padx=(0, 12))
        self.observability_line_primary_3.pack(side="left", padx=(0, 12))
        self.observability_line_primary_4.pack(side="left", padx=(0, 12))
        self.observability_line_secondary.pack(side="left", padx=(0, 12))
        self.observability_line_secondary_2.pack(side="left", padx=(0, 12))
        self.observability_line_secondary_3.pack(side="left", padx=(0, 12))
        self.observability_line_secondary_4.pack(side="left")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.create_experiment_tab()
        self.create_dashboard_tab()
        self.create_main_tab()
        self.create_construction_tab()
        self.create_agents_tab()
        self.create_event_monitor_tab()
        self.create_interaction_tab()
        self.create_brain_tab()
        self.create_dik_derivations_tab()
        self.create_rule_derivations_tab()

    def _build_environment_canvas(self, parent):
        try:
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            return fig, ax, canvas
        except Exception as exc:
            messagebox.showerror("Plot Initialization Error", self.PLOTTING_INIT_ERROR)
            raise RuntimeError(self.PLOTTING_INIT_ERROR) from exc

    def create_dashboard_tab(self):
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="Dashboard")

        self.dashboard_panes = ttk.PanedWindow(self.tab_dashboard, orient="horizontal")
        self.dashboard_panes.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        left_frame = ttk.Frame(self.dashboard_panes)
        right_frame = ttk.Frame(self.dashboard_panes)
        self.dashboard_panes.add(left_frame, weight=3)
        self.dashboard_panes.add(right_frame, weight=2)

        env_frame = ttk.LabelFrame(left_frame, text="Environment Overview", padding=6)
        env_frame.grid(row=0, column=0, sticky="nsew")
        self.dashboard_fig, self.dashboard_ax, self.dashboard_canvas = self._build_environment_canvas(env_frame)

        bottom_frame = ttk.Frame(left_frame)
        bottom_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        summary_pane = ttk.PanedWindow(bottom_frame, orient="horizontal")
        summary_pane.pack(fill="both", expand=True)

        agent_frame = ttk.LabelFrame(summary_pane, text="Agent State Summary", padding=6)
        self.dashboard_agent_state = ttk.Treeview(
            agent_frame,
            columns=("Agent", "Goal", "HR", "CO2"),
            show="headings",
            height=6,
        )
        for col in self.dashboard_agent_state["columns"]:
            self.dashboard_agent_state.heading(col, text=col)
            self.dashboard_agent_state.column(col, width=90, anchor="center")
        self.dashboard_agent_state.pack(fill="both", expand=True)

        construction_frame = ttk.LabelFrame(summary_pane, text="Construction Summary", padding=6)
        self.dashboard_construction_text = tk.Text(construction_frame, height=8, wrap="word")
        self.dashboard_construction_text.pack(side="left", fill="both", expand=True)
        cons_scroll = ttk.Scrollbar(construction_frame, orient="vertical", command=self.dashboard_construction_text.yview)
        cons_scroll.pack(side="right", fill="y")
        self.dashboard_construction_text.configure(yscrollcommand=cons_scroll.set)

        summary_pane.add(agent_frame, weight=3)
        summary_pane.add(construction_frame, weight=2)

        left_frame.rowconfigure(0, weight=4)
        left_frame.rowconfigure(1, weight=2)
        left_frame.columnconfigure(0, weight=1)

        event_frame = ttk.LabelFrame(right_frame, text="Event Monitor", padding=6)
        event_frame.grid(row=0, column=0, sticky="nsew")
        self.dashboard_agent_activity_text = tk.Text(event_frame, wrap="word", height=8)
        self.dashboard_interaction_state_text = tk.Text(event_frame, wrap="word", height=8)
        self.dashboard_zone_state_text = tk.Text(event_frame, wrap="word", height=6)

        ttk.Label(event_frame, text="Agent Activities").grid(row=0, column=0, sticky="w")
        ttk.Label(event_frame, text="Interaction State Machine").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(event_frame, text="Agent Zone States").grid(row=4, column=0, sticky="w", pady=(4, 0))

        self.dashboard_agent_activity_text.grid(row=1, column=0, sticky="nsew")
        self.dashboard_interaction_state_text.grid(row=3, column=0, sticky="nsew")
        self.dashboard_zone_state_text.grid(row=5, column=0, sticky="nsew")

        for r, widget in [(1, self.dashboard_agent_activity_text), (3, self.dashboard_interaction_state_text), (5, self.dashboard_zone_state_text)]:
            scroll = ttk.Scrollbar(event_frame, orient="vertical", command=widget.yview)
            scroll.grid(row=r, column=1, sticky="ns")
            widget.configure(yscrollcommand=scroll.set)

        event_frame.rowconfigure(1, weight=3)
        event_frame.rowconfigure(3, weight=2)
        event_frame.rowconfigure(5, weight=1)
        event_frame.columnconfigure(0, weight=1)

        system_log_frame = ttk.LabelFrame(right_frame, text="Simulator / System Log", padding=6)
        system_log_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.system_log_text = tk.Text(system_log_frame, wrap="word", height=10)
        self.system_log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(system_log_frame, orient="vertical", command=self.system_log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.system_log_text.configure(yscrollcommand=log_scroll.set)

        self.backend_status_var = StringVar(value="Backend (configured/effective): rule_brain / rule_brain")
        ttk.Label(right_frame, textvariable=self.backend_status_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=2)
        right_frame.columnconfigure(0, weight=1)

        self.tab_dashboard.rowconfigure(0, weight=1)
        self.tab_dashboard.columnconfigure(0, weight=1)

    def create_main_tab(self):
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Environment")

        self.fig, self.ax, self.canvas = self._build_environment_canvas(self.tab_main)

    def create_construction_tab(self):
        self.tab_construction = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_construction, text="Construction")

        panes = ttk.PanedWindow(self.tab_construction, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=6)

        visual_frame = ttk.LabelFrame(panes, text="Construction Sites", padding=6)
        text_frame = ttk.LabelFrame(panes, text="Construction Summary", padding=6)
        panes.add(visual_frame, weight=3)
        panes.add(text_frame, weight=2)

        self.construction_fig, self.construction_ax, self.construction_canvas = self._build_construction_canvas(visual_frame)

        self.construction_text = tk.Text(text_frame, wrap="word")
        self.construction_text.pack(side="left", fill="both", expand=True)
        cons_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.construction_text.yview)
        cons_scroll.pack(side="right", fill="y")
        self.construction_text.configure(yscrollcommand=cons_scroll.set)

    def _build_construction_canvas(self, parent):
        return self._build_environment_canvas(parent)

    def create_agents_tab(self):
        self.tab_agents = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_agents, text="Agent States")

        self.agent_state_table = ttk.Treeview(self.tab_agents, columns=("Heart Rate", "GSR", "Temp", "CO2"), show="headings")
        for col in self.agent_state_table["columns"]:
            self.agent_state_table.heading(col, text=col)
        self.agent_state_table.pack(fill="both", expand=True)


    def create_interaction_tab(self):
        self.tab_interaction = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_interaction, text="Interactions")

        controls = ttk.Frame(self.tab_interaction, padding=6)
        controls.pack(fill="x")
        ttk.Label(controls, text="Agent").pack(side="left")
        self.interaction_agent_filter = StringVar(value="All")
        ttk.Combobox(controls, textvariable=self.interaction_agent_filter, values=["All", "Architect", "Engineer", "Botanist"], width=12, state="readonly").pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="Type").pack(side="left")
        self.interaction_type_filter = StringVar(value="All")
        ttk.Entry(controls, textvariable=self.interaction_type_filter, width=20).pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="Window (s)").pack(side="left")
        self.interaction_window = DoubleVar(value=20.0)
        ttk.Entry(controls, textvariable=self.interaction_window, width=8).pack(side="left", padx=(4, 10))

        panes = ttk.PanedWindow(self.tab_interaction, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        self.interaction_canvas = tk.Canvas(left, bg="#10141b", height=560)
        self.interaction_canvas.pack(fill="both", expand=True)

        self.interaction_list = tk.Text(right, wrap="word", height=30)
        self.interaction_list.pack(fill="both", expand=True)

    @staticmethod
    def _safe_pretty_json(value):
        if value is None:
            return "(none)"
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return "(empty)"
            try:
                return json.dumps(json.loads(text), indent=2, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                return text
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _brain_normalize_lifecycle_row(row):
        normalized = dict(row or {})
        request_data = dict(normalized.get("request") or {})
        response_data = dict(normalized.get("response") or {})
        interpretation_data = dict(normalized.get("interpretation") or {})

        request_data.setdefault("status", "submitted")
        request_data.setdefault("configured_backend", normalized.get("configured_backend"))
        request_data.setdefault("effective_backend", normalized.get("effective_backend"))
        request_data.setdefault("model", normalized.get("model"))
        request_data.setdefault("trigger_reason", normalized.get("trigger_reason"))
        request_data.setdefault("request_payload", normalized.get("agent_brain_request_payload"))

        response_status = response_data.get("status")
        if not response_status:
            if response_data.get("http_response_received"):
                response_status = "response_received"
            elif response_data.get("timeout_occurred"):
                response_status = "timeout"
            elif response_data.get("error"):
                response_status = "transport_error"
            else:
                response_status = "pending_or_no_response"
        response_data.setdefault("status", response_status)
        response_data.setdefault("http_response_received", bool(response_data.get("http_response_received")))
        response_data.setdefault("json_parsed", bool(response_data.get("json_parsed")))
        response_data.setdefault("normalized_payload_exists", bool(response_data.get("normalized_payload") or response_data.get("normalized_payload_exists")))
        response_data.setdefault("repair_retry_attempted", bool(response_data.get("repair_retry_attempted")))
        response_data.setdefault("raw_response_available", bool(response_data.get("raw_response")))

        interpretation_status = (
            interpretation_data.get("status")
            or interpretation_data.get("runtime_disposition")
            or interpretation_data.get("planner_result")
            or "pending_or_unusable"
        )
        interpretation_data.setdefault("status", interpretation_status)
        interpretation_data.setdefault("runtime_disposition", interpretation_data.get("runtime_disposition") or "unknown")
        interpretation_data.setdefault("planner_result", interpretation_data.get("planner_result") or "none")
        interpretation_data.setdefault("adopted_action", interpretation_data.get("adopted_action") or "none")
        interpretation_data.setdefault("fallback_used", bool(interpretation_data.get("fallback_used")))
        interpretation_data.setdefault("failure_mode", interpretation_data.get("failure_mode") or interpretation_data.get("stale_discard_reason") or "none")

        normalized["request"] = request_data
        normalized["response"] = response_data
        normalized["interpretation"] = interpretation_data
        normalized.setdefault("request_kind", "planner")
        return normalized

    @staticmethod
    def _brain_lifecycle_rows_for_agent(traces, agent_filter):
        rows = [MarsColonyInterface._brain_normalize_lifecycle_row(row) for row in (traces or [])]
        if agent_filter and agent_filter != "All":
            lowered = agent_filter.strip().lower()
            rows = [
                row
                for row in rows
                if str(row.get("agent_id", "")).lower() == lowered
                or str(row.get("display_name", "")).lower() == lowered
                or str(row.get("agent", "")).lower() == lowered
            ]
        rows.sort(
            key=lambda row: (
                float(row.get("sim_time") or 0.0),
                int(row.get("tick") or 0),
                str(row.get("request_id") or ""),
            )
        )
        return rows

    @staticmethod
    def _brain_lifecycle_rows_filtered(traces, *, agent_filter="All", disposition_filter="All", source_filter="All", search_text=""):
        rows = MarsColonyInterface._brain_lifecycle_rows_for_agent(traces, agent_filter)
        needle = str(search_text or "").strip().lower()
        filtered = []
        for row in rows:
            interpretation = row.get("interpretation") or {}
            request_data = row.get("request") or {}
            disposition = str(interpretation.get("runtime_disposition") or interpretation.get("planner_result") or interpretation.get("status") or "").strip()
            source = str(interpretation.get("result_source") or request_data.get("effective_backend") or request_data.get("configured_backend") or "").strip()
            if disposition_filter and disposition_filter != "All" and disposition != disposition_filter:
                continue
            if source_filter and source_filter != "All" and source != source_filter:
                continue
            if needle:
                blob = " ".join(
                    [
                        str(row.get("request_id") or ""),
                        str(row.get("trace_id") or ""),
                        str(interpretation.get("runtime_disposition") or ""),
                        str(interpretation.get("planner_result") or ""),
                        str(interpretation.get("adopted_action") or ""),
                        str((row.get("response") or {}).get("raw_response") or ""),
                    ]
                ).lower()
                if needle not in blob:
                    continue
            filtered.append(row)
        return filtered

    @staticmethod
    def _brain_request_summary_line(row):
        sim_time = row.get("sim_time")
        tick = row.get("tick")
        agent = row.get("display_name") or row.get("agent_id") or "?"
        request_data = row.get("request") or {}
        source = request_data.get("effective_backend") or request_data.get("configured_backend") or "unknown"
        return (
            f"{row.get('request_id')} t={sim_time} tick={tick} {agent} "
            f"kind={row.get('request_kind') or 'planner'} src={source} model={request_data.get('model') or '-'} "
            f"trigger={request_data.get('trigger_reason') or '-'} [{request_data.get('status')}]"
        )

    @staticmethod
    def _brain_response_summary_line(row):
        response = row.get("response") or {}
        error_marker = response.get("error") or response.get("failure_mode") or ""
        return (
            f"{row.get('request_id')} [{response.get('status')}] "
            f"http={bool(response.get('http_response_received'))} parsed={bool(response.get('json_parsed'))} "
            f"normalized={bool(response.get('normalized_payload_exists'))} retry={bool(response.get('repair_retry_attempted'))} "
            f"raw={bool(response.get('raw_response_available') or response.get('raw_response'))}"
            f"{f' err={error_marker}' if error_marker else ''}"
        )

    @staticmethod
    def _brain_interpretation_summary_line(row):
        interpretation = row.get("interpretation") or {}
        failure = ""
        if interpretation.get("failure_mode") and interpretation.get("failure_mode") != "none":
            failure = f" failure={interpretation.get('failure_mode')}"
        stale = ""
        if interpretation.get("stale_discard_reason"):
            stale = f" stale={interpretation.get('stale_discard_reason')}"
        return (
            f"{row.get('request_id')} [{interpretation.get('status')}] "
            f"disp={interpretation.get('runtime_disposition')} result={interpretation.get('planner_result')} "
            f"action={interpretation.get('adopted_action')} fallback={bool(interpretation.get('fallback_used'))}"
            f"{failure}"
            f"{stale}"
        )

    @staticmethod
    def _brain_visible_signature(rows):
        return tuple(
            (
                row.get("request_id"),
                row.get("request_kind"),
                (row.get("request") or {}).get("status"),
                (row.get("request") or {}).get("effective_backend"),
                (row.get("request") or {}).get("model"),
                (row.get("response") or {}).get("status"),
                bool((row.get("response") or {}).get("http_response_received")),
                bool((row.get("response") or {}).get("json_parsed")),
                bool((row.get("response") or {}).get("normalized_payload_exists")),
                (row.get("interpretation") or {}).get("status"),
                (row.get("interpretation") or {}).get("runtime_disposition"),
                (row.get("interpretation") or {}).get("planner_result"),
                (row.get("interpretation") or {}).get("adopted_action"),
            )
            for row in (rows or [])
        )

    @staticmethod
    def _brain_lifecycle_detail_sections(row):
        request_data = row.get("request") or {}
        response_data = row.get("response") or {}
        interpretation_data = row.get("interpretation") or {}
        request_summary = {
            "request_id": row.get("request_id"),
            "trace_id": row.get("trace_id"),
            "tick": row.get("tick"),
            "sim_time": row.get("sim_time"),
            "request_kind": row.get("request_kind"),
            "configured_backend": request_data.get("configured_backend"),
            "effective_backend": request_data.get("effective_backend"),
            "model": request_data.get("model"),
            "trigger_reason": request_data.get("trigger_reason"),
            "request_status": request_data.get("status"),
            "status_badges": [
                badge
                for badge, present in [
                    ("fallback", bool(interpretation_data.get("fallback_used"))),
                    ("timeout", "timeout" in str(response_data.get("status") or "") or "timeout" in str(interpretation_data.get("status") or "")),
                    ("repair_retry", bool(response_data.get("repair_retry_attempted"))),
                    ("late_accepted", bool(interpretation_data.get("late_result_accepted"))),
                ]
                if present
            ],
        }
        interpretation = {
            "runtime_disposition": interpretation_data.get("runtime_disposition"),
            "planner_result": interpretation_data.get("planner_result"),
            "result_source": interpretation_data.get("result_source"),
            "schema_validation_errors": interpretation_data.get("schema_validation_errors"),
            "plan_grounding_status": interpretation_data.get("plan_grounding_status"),
            "plan_grounding_notes": interpretation_data.get("plan_grounding_notes"),
            "fallback_used": interpretation_data.get("fallback_used"),
            "fallback_source": interpretation_data.get("fallback_source"),
            "adopted_action": interpretation_data.get("adopted_action"),
            "late_result_arrived": interpretation_data.get("late_result_arrived"),
            "late_result_accepted": interpretation_data.get("late_result_accepted"),
            "stale_discard_reason": interpretation_data.get("stale_discard_reason"),
            "failure_mode": interpretation_data.get("failure_mode"),
        }
        errors_notes = {
            "exception": response_data.get("error") or interpretation_data.get("error"),
            "normalization_steps": response_data.get("normalization_steps"),
            "repair_retry_attempted": response_data.get("repair_retry_attempted"),
            "grounding_notes": interpretation_data.get("plan_grounding_notes"),
        }
        return {
            "request_summary": request_summary,
            "request_payload": request_data.get("request_payload"),
            "response_summary": {
                "response_status": response_data.get("status"),
                "http_response_received": response_data.get("http_response_received"),
                "json_parsed": response_data.get("json_parsed"),
                "normalized_payload_exists": response_data.get("normalized_payload_exists"),
                "repair_retry_attempted": response_data.get("repair_retry_attempted"),
                "raw_response_available": response_data.get("raw_response_available"),
                "error": response_data.get("error"),
            },
            "raw_response": response_data.get("raw_response"),
            "parsed_pre_normalized": response_data.get("parsed_payload"),
            "normalized_response": response_data.get("normalized_payload"),
            "system_interpretation": interpretation,
            "errors_notes": errors_notes,
        }

    @staticmethod
    def _brain_trace_detail_sections(row):
        """Backward-compatible alias for older helper names."""
        return MarsColonyInterface._brain_lifecycle_detail_sections(row)

    @staticmethod
    def _brain_trace_rows_for_agent(traces, agent_filter):
        return MarsColonyInterface._brain_lifecycle_rows_for_agent(traces, agent_filter)

    @staticmethod
    def _brain_trace_rows_filtered(traces, *, agent_filter="All", disposition_filter="All", source_filter="All", search_text=""):
        return MarsColonyInterface._brain_lifecycle_rows_filtered(
            traces,
            agent_filter=agent_filter,
            disposition_filter=disposition_filter,
            source_filter=source_filter,
            search_text=search_text,
        )

    @staticmethod
    def _brain_trace_summary_line(row):
        return MarsColonyInterface._brain_interpretation_summary_line(row)

    def create_brain_tab(self):
        self.tab_brain = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_brain, text="Brain")
        self.tab_brain.rowconfigure(1, weight=3)
        self.tab_brain.rowconfigure(2, weight=2)
        self.tab_brain.columnconfigure(0, weight=1)

        controls = ttk.Frame(self.tab_brain, padding=6)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Agent").pack(side="left")
        self.brain_agent_filter = StringVar(value="All")
        self.brain_agent_combo = ttk.Combobox(controls, textvariable=self.brain_agent_filter, values=["All"], width=20, state="readonly")
        self.brain_agent_combo.pack(side="left", padx=(4, 10))
        self.brain_agent_combo.bind("<<ComboboxSelected>>", lambda *_: self.update_brain_tab())
        ttk.Label(controls, text="Disposition").pack(side="left")
        self.brain_disposition_filter = StringVar(value="All")
        self.brain_disposition_combo = ttk.Combobox(controls, textvariable=self.brain_disposition_filter, values=["All"], width=20, state="readonly")
        self.brain_disposition_combo.pack(side="left", padx=(4, 10))
        self.brain_disposition_combo.bind("<<ComboboxSelected>>", lambda *_: self.update_brain_tab())
        ttk.Label(controls, text="Source").pack(side="left")
        self.brain_source_filter = StringVar(value="All")
        self.brain_source_combo = ttk.Combobox(controls, textvariable=self.brain_source_filter, values=["All"], width=16, state="readonly")
        self.brain_source_combo.pack(side="left", padx=(4, 10))
        self.brain_source_combo.bind("<<ComboboxSelected>>", lambda *_: self.update_brain_tab())
        ttk.Label(controls, text="Search").pack(side="left")
        self.brain_search_var = StringVar(value="")
        search_entry = ttk.Entry(controls, textvariable=self.brain_search_var, width=18)
        search_entry.pack(side="left", padx=(4, 8))
        search_entry.bind("<KeyRelease>", lambda *_: self.update_brain_tab())
        ttk.Button(controls, text="Refresh", command=lambda: self.update_brain_tab(force=True)).pack(side="left", padx=(2, 4))
        self.brain_auto_refresh_var = BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto-refresh", variable=self.brain_auto_refresh_var).pack(side="left", padx=4)
        self.brain_follow_latest_var = BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Follow latest", variable=self.brain_follow_latest_var).pack(side="left", padx=4)
        ttk.Button(controls, text="Copy detail", command=self._copy_brain_detail_text).pack(side="left", padx=(8, 4))
        ttk.Button(controls, text="Copy selected JSON", command=self._copy_brain_selected_json).pack(side="left", padx=4)
        ttk.Button(controls, text="Copy raw", command=self._copy_brain_raw_response).pack(side="left", padx=4)
        ttk.Button(controls, text="Copy normalized", command=self._copy_brain_normalized_response).pack(side="left", padx=4)
        ttk.Button(controls, text="Copy attempts", command=self._copy_brain_provider_attempts).pack(side="left", padx=4)
        ttk.Button(controls, text="Copy lifecycle bundle", command=self._copy_brain_lifecycle_bundle).pack(side="left", padx=4)
        self.brain_copy_status_var = StringVar(value="")
        ttk.Label(controls, textvariable=self.brain_copy_status_var, foreground="#3f556e").pack(side="right")

        lifecycle_frame = ttk.Frame(self.tab_brain, padding=(6, 0, 6, 6))
        lifecycle_frame.grid(row=1, column=0, sticky="nsew")
        lifecycle_frame.rowconfigure(0, weight=1)
        lifecycle_frame.columnconfigure(0, weight=1)
        lifecycle_frame.columnconfigure(1, weight=1)
        lifecycle_frame.columnconfigure(2, weight=1)

        req_frame = ttk.LabelFrame(lifecycle_frame, text="Requests", padding=4)
        resp_frame = ttk.LabelFrame(lifecycle_frame, text="Responses", padding=4)
        interp_frame = ttk.LabelFrame(lifecycle_frame, text="Interpretation", padding=4)
        req_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        resp_frame.grid(row=0, column=1, sticky="nsew", padx=4)
        interp_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        self.brain_request_list = tk.Listbox(req_frame, exportselection=False)
        self.brain_request_list.pack(side="left", fill="both", expand=True)
        req_scroll = ttk.Scrollbar(req_frame, orient="vertical", command=lambda *args: self._brain_sync_lists_yview(*args))
        req_scroll.pack(side="right", fill="y")
        self.brain_request_list.configure(yscrollcommand=req_scroll.set)
        self.brain_request_list.bind("<<ListboxSelect>>", lambda *_: self._on_brain_list_selection_changed("request"))

        self.brain_response_list = tk.Listbox(resp_frame, exportselection=False)
        self.brain_response_list.pack(side="left", fill="both", expand=True)
        resp_scroll = ttk.Scrollbar(resp_frame, orient="vertical", command=lambda *args: self._brain_sync_lists_yview(*args))
        resp_scroll.pack(side="right", fill="y")
        self.brain_response_list.configure(yscrollcommand=resp_scroll.set)
        self.brain_response_list.bind("<<ListboxSelect>>", lambda *_: self._on_brain_list_selection_changed("response"))

        self.brain_interpretation_list = tk.Listbox(interp_frame, exportselection=False)
        self.brain_interpretation_list.pack(side="left", fill="both", expand=True)
        interp_scroll = ttk.Scrollbar(interp_frame, orient="vertical", command=lambda *args: self._brain_sync_lists_yview(*args))
        interp_scroll.pack(side="right", fill="y")
        self.brain_interpretation_list.configure(yscrollcommand=interp_scroll.set)
        self.brain_interpretation_list.bind("<<ListboxSelect>>", lambda *_: self._on_brain_list_selection_changed("interpretation"))

        for lb in (self.brain_request_list, self.brain_response_list, self.brain_interpretation_list):
            lb.bind("<MouseWheel>", self._brain_on_listbox_mousewheel)
            lb.bind("<Button-4>", self._brain_on_listbox_mousewheel)
            lb.bind("<Button-5>", self._brain_on_listbox_mousewheel)

        detail_frame = ttk.LabelFrame(self.tab_brain, text="Request Detail", padding=6)
        detail_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.brain_detail_text = tk.Text(detail_frame, wrap="word")
        self.brain_detail_text.pack(side="left", fill="both", expand=True)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.brain_detail_text.yview)
        detail_scroll.pack(side="right", fill="y")
        self.brain_detail_text.configure(yscrollcommand=detail_scroll.set)
        self._brain_visible_rows = []
        self._brain_visible_signature_cache = ()
        self._brain_last_detail_key = None
        self._brain_user_inspecting = False
        self._brain_syncing_scroll = False

    def _brain_sync_lists_yview(self, *args):
        if getattr(self, "_brain_syncing_scroll", False):
            return
        self._brain_syncing_scroll = True
        try:
            for list_name in ("brain_request_list", "brain_response_list", "brain_interpretation_list"):
                lb = getattr(self, list_name, None)
                if lb is not None:
                    lb.yview(*args)
        finally:
            self._brain_syncing_scroll = False

    def _brain_on_listbox_mousewheel(self, event):
        delta = getattr(event, "delta", 0)
        if delta:
            units = -1 if delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            return None
        self._brain_sync_lists_yview("scroll", units, "units")
        return "break"

    @staticmethod
    def _brain_detail_bundle_for_row(row):
        return MarsColonyInterface._brain_lifecycle_detail_sections(row)

    @staticmethod
    def _brain_lifecycle_bundle_for_request(rows, request_id):
        for row in rows or []:
            if str(row.get("request_id")) == str(request_id):
                return MarsColonyInterface._brain_lifecycle_detail_sections(row)
        return {}

    @staticmethod
    def _brain_aligned_request_ids(rows):
        return [str((row or {}).get("request_id") or "") for row in (rows or [])]

    def _brain_selected_index(self):
        for list_name in ("brain_request_list", "brain_response_list", "brain_interpretation_list"):
            lb = getattr(self, list_name, None)
            if lb is None:
                continue
            cur = lb.curselection()
            if cur:
                return cur[0]
        return None

    def _on_brain_list_selection_changed(self, source=None):
        if not hasattr(self, "brain_request_list"):
            return
        mapping = {
            "request": getattr(self, "brain_request_list", None),
            "response": getattr(self, "brain_response_list", None),
            "interpretation": getattr(self, "brain_interpretation_list", None),
        }
        selected = []
        if source in mapping and mapping[source] is not None:
            selected = mapping[source].curselection()
        if not selected:
            idx = self._brain_selected_index()
            selected = [idx] if idx is not None else []
        rows = self._brain_visible_rows or []
        if selected and selected[0] < len(rows):
            is_latest = selected[0] == len(rows) - 1
            self._brain_user_inspecting = not is_latest
            self._sync_brain_list_selection(selected[0])
        self._update_brain_detail()

    def _sync_brain_list_selection(self, index):
        for list_name in ("brain_request_list", "brain_response_list", "brain_interpretation_list"):
            lb = getattr(self, list_name, None)
            if lb is None:
                continue
            lb.selection_clear(0, tk.END)
            if index < lb.size():
                lb.selection_set(index)
                lb.activate(index)

    def _format_brain_detail_text(self, row):
        sections = self._brain_lifecycle_detail_sections(row)
        ordered = [
            ("1. Request summary", sections["request_summary"]),
            ("2. Request payload", sections["request_payload"]),
            ("3. Response summary", sections["response_summary"]),
            ("4. Raw response", sections["raw_response"]),
            ("5. Parsed / normalized response (pre-normalized)", sections["parsed_pre_normalized"]),
            ("5b. Parsed / normalized response (normalized)", sections["normalized_response"]),
            ("6. System interpretation", sections["system_interpretation"]),
            ("7. Errors / notes", sections["errors_notes"]),
        ]
        rendered = []
        for heading, value in ordered:
            rendered.append(f"{heading}\n{self._safe_pretty_json(value)}\n")
        return "\n".join(rendered)

    def _update_brain_detail(self, *, force=False):
        if not hasattr(self, "brain_request_list"):
            return
        selected_index = self._brain_selected_index()
        row = self._brain_visible_rows[selected_index] if selected_index is not None and selected_index < len(self._brain_visible_rows) else None
        if not row:
            self.brain_detail_text.delete("1.0", tk.END)
            self.brain_detail_text.insert(tk.END, "No request selected.")
            self._brain_last_detail_key = None
            return
        rendered = self._format_brain_detail_text(row)
        detail_key = (row.get("request_id"), rendered)
        if not force and detail_key == self._brain_last_detail_key:
            return
        detail_scroll = self.brain_detail_text.yview()
        self.brain_detail_text.delete("1.0", tk.END)
        self.brain_detail_text.insert(tk.END, rendered)
        if detail_scroll:
            self.brain_detail_text.yview_moveto(detail_scroll[0])
        self._brain_last_detail_key = detail_key

    def _copy_to_clipboard(self, content, *, status_message):
        text = str(content or "").strip()
        if not text:
            self.brain_copy_status_var.set("Nothing to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.brain_copy_status_var.set(status_message)
        self.root.after(1200, lambda: self.brain_copy_status_var.set(""))

    def _copy_brain_detail_text(self):
        if not hasattr(self, "brain_detail_text"):
            return
        text = self.brain_detail_text.get("1.0", "end-1c")
        self._copy_to_clipboard(text, status_message="Detail copied.")

    def _copy_brain_selected_json(self):
        row = self._brain_selected_row()
        if not row:
            self._copy_to_clipboard("", status_message="Nothing selected.")
            return
        bundle = self._brain_detail_bundle_for_row(row)
        self._copy_to_clipboard(json.dumps(bundle, indent=2, sort_keys=True, default=str), status_message="Selected JSON copied.")

    def _brain_selected_row(self):
        if not hasattr(self, "brain_request_list"):
            return None
        selected_index = self._brain_selected_index()
        return self._brain_visible_rows[selected_index] if selected_index is not None and selected_index < len(self._brain_visible_rows) else None

    def _copy_brain_raw_response(self):
        row = self._brain_selected_row()
        self._copy_to_clipboard(((row or {}).get("response") or {}).get("raw_response"), status_message="Raw response copied.")

    def _copy_brain_normalized_response(self):
        row = self._brain_selected_row()
        normalized = ((row or {}).get("response") or {}).get("normalized_payload")
        self._copy_to_clipboard(self._safe_pretty_json(normalized), status_message="Normalized response copied.")

    def _copy_brain_provider_attempts(self):
        row = self._brain_selected_row()
        attempts = (((row or {}).get("response") or {}).get("provider_trace") or {}).get("attempts") or []
        self._copy_to_clipboard(self._safe_pretty_json(attempts), status_message="Provider attempts copied.")

    def _copy_brain_lifecycle_bundle(self):
        row = self._brain_selected_row()
        bundle = self._brain_detail_bundle_for_row(row) if row else {}
        self._copy_to_clipboard(self._safe_pretty_json(bundle), status_message="Lifecycle bundle copied.")

    def _copy_brain_trace_bundle(self):
        """Backward-compatible alias for older tests/callers."""
        self._copy_brain_lifecycle_bundle()

    def update_brain_tab(self, *, force=False):
        if not hasattr(self, "brain_request_list"):
            return
        if not force and hasattr(self, "brain_auto_refresh_var") and not self.brain_auto_refresh_var.get():
            return
        lifecycle_rows = []
        if self.sim and hasattr(self.sim, "logger"):
            if hasattr(self.sim.logger, "get_recent_brain_lifecycle"):
                lifecycle_rows = list(self.sim.logger.get_recent_brain_lifecycle(500))
            elif hasattr(self.sim.logger, "get_recent_planner_traces"):
                raw_traces = list(self.sim.logger.get_recent_planner_traces(500))
                lifecycle_rows = [{"request_id": row.get("request_id"), "trace_id": row.get("trace_id"), "agent_id": row.get("agent_id"), "display_name": row.get("display_name"), "sim_time": row.get("sim_time"), "tick": row.get("tick"), "request_kind": "planner", "request": {"status": "completed", "configured_backend": row.get("configured_backend"), "effective_backend": row.get("effective_backend"), "model": row.get("model"), "trigger_reason": row.get("trigger_reason"), "request_payload": row.get("agent_brain_request_payload")}, "response": {"status": "response_received" if row.get("llm_response_received") else ("no_response_timeout" if row.get("timeout_occurred") else "transport_error"), "http_response_received": row.get("llm_response_received"), "json_parsed": row.get("llm_response_parsed"), "normalized_payload_exists": bool(row.get("normalized_agent_brain_response")), "raw_response": row.get("raw_http_response_text"), "parsed_payload": row.get("extracted_response_payload"), "normalized_payload": row.get("normalized_agent_brain_response"), "provider_trace": row.get("provider_trace")}, "interpretation": {"status": row.get("runtime_disposition") or row.get("planner_result"), "runtime_disposition": row.get("runtime_disposition"), "planner_result": row.get("planner_result"), "result_source": row.get("result_source"), "adopted_action": (row.get("next_action_summary") or {}).get("action_type"), "fallback_used": row.get("fallback_used"), "fallback_source": row.get("fallback_source"), "schema_validation_errors": row.get("schema_validation_errors"), "plan_grounding_status": row.get("plan_grounding_status"), "plan_grounding_notes": row.get("plan_grounding_notes"), "late_result_arrived": row.get("late_result_arrived"), "late_result_accepted": row.get("late_result_accepted"), "stale_discard_reason": row.get("stale_discard_reason")}} for row in raw_traces]
        agent_names = sorted(
            {
                str(row.get("display_name") or row.get("agent_id"))
                for row in lifecycle_rows
                if row.get("display_name") or row.get("agent_id")
            }
        )
        current = self.brain_agent_filter.get() if hasattr(self, "brain_agent_filter") else "All"
        values = ["All"] + agent_names
        self.brain_agent_combo.configure(values=values)
        if current not in values:
            current = "All"
            self.brain_agent_filter.set(current)
        selected_index = self._brain_selected_index()
        selected_id = None
        if selected_index is not None and selected_index < len(self._brain_visible_rows):
            selected_id = self._brain_visible_rows[selected_index].get("request_id")

        disposition_values = sorted({str((r.get("interpretation") or {}).get("runtime_disposition") or (r.get("interpretation") or {}).get("planner_result") or (r.get("interpretation") or {}).get("status") or "") for r in lifecycle_rows if (r.get("interpretation") or {}).get("status") or (r.get("interpretation") or {}).get("runtime_disposition") or (r.get("interpretation") or {}).get("planner_result")})
        source_values = sorted({str((r.get("interpretation") or {}).get("result_source") or (r.get("request") or {}).get("effective_backend") or (r.get("request") or {}).get("configured_backend") or "") for r in lifecycle_rows if (r.get("request") or {}).get("configured_backend") or (r.get("interpretation") or {}).get("result_source")})
        self.brain_disposition_combo.configure(values=["All"] + disposition_values)
        if self.brain_disposition_filter.get() not in (["All"] + disposition_values):
            self.brain_disposition_filter.set("All")
        self.brain_source_combo.configure(values=["All"] + source_values)
        if self.brain_source_filter.get() not in (["All"] + source_values):
            self.brain_source_filter.set("All")
        rows = self._brain_lifecycle_rows_filtered(
            lifecycle_rows,
            agent_filter=self.brain_agent_filter.get(),
            disposition_filter=self.brain_disposition_filter.get(),
            source_filter=self.brain_source_filter.get(),
            search_text=self.brain_search_var.get() if hasattr(self, "brain_search_var") else "",
        )
        new_signature = self._brain_visible_signature(rows)
        request_scroll = self.brain_request_list.yview()
        rows_changed = force or new_signature != self._brain_visible_signature_cache
        self._brain_visible_rows = rows
        if rows_changed:
            self.brain_request_list.delete(0, tk.END)
            if hasattr(self, "brain_response_list"):
                self.brain_response_list.delete(0, tk.END)
            if hasattr(self, "brain_interpretation_list"):
                self.brain_interpretation_list.delete(0, tk.END)
            for row in rows:
                self.brain_request_list.insert(tk.END, self._brain_request_summary_line(row))
                if hasattr(self, "brain_response_list"):
                    self.brain_response_list.insert(tk.END, self._brain_response_summary_line(row))
                if hasattr(self, "brain_interpretation_list"):
                    self.brain_interpretation_list.insert(tk.END, self._brain_interpretation_summary_line(row))
            if request_scroll:
                self._brain_sync_lists_yview("moveto", request_scroll[0])
            self._brain_visible_signature_cache = new_signature

        if rows:
            reselect = 0
            if selected_id:
                for idx, row in enumerate(rows):
                    if row.get("request_id") == selected_id:
                        reselect = idx
                        break
            elif self.brain_follow_latest_var.get() and not self._brain_user_inspecting:
                reselect = len(rows) - 1
            if self.brain_follow_latest_var.get() and not self._brain_user_inspecting and not selected_id:
                reselect = len(rows) - 1
            self._sync_brain_list_selection(reselect)
        self._update_brain_detail(force=force or rows_changed)

    def update_interaction_tab(self):
        if not self.sim or not hasattr(self.sim, "logger"):
            return
        interactions = list(self.sim.logger.get_recent_interactions(180))
        agent_filter = self.interaction_agent_filter.get()
        type_filter = self.interaction_type_filter.get().strip().lower()
        window_s = max(1.0, float(self.interaction_window.get() or 20.0))
        now_t = float(getattr(self.sim, "time", 0.0))

        filtered = []
        for row in interactions:
            if now_t - float(row.get("time", 0.0) or 0.0) > window_s:
                continue
            aid = str(row.get("agent_id") or "")
            if agent_filter != "All" and agent_filter.lower() not in aid.lower():
                continue
            itype = str(row.get("interaction_type") or "")
            if type_filter and type_filter != "all" and type_filter not in itype.lower():
                continue
            filtered.append(row)

        self._draw_interaction_graph(filtered)
        self.interaction_list.delete("1.0", tk.END)
        for row in filtered[-40:]:
            self.interaction_list.insert(tk.END, f"t={row.get('time')} {row.get('interaction_type')} {row.get('source_node')} -> {row.get('target_node')} [{row.get('status')}] {row.get('payload_summary')}\n")

    def _draw_interaction_graph(self, interactions):
        canvas = self.interaction_canvas
        canvas.delete("all")
        w = max(200, canvas.winfo_width() or 900)
        h = max(200, canvas.winfo_height() or 560)

        active_nodes = set()
        active_edges = set()
        for row in interactions[-30:]:
            src = row.get("source_node")
            tgt = row.get("target_node")
            if src:
                active_nodes.add(src)
            if tgt:
                active_nodes.add(tgt)
            if src and tgt:
                active_edges.add((src, tgt))

        for src, tgt in active_edges:
            src_node = next((n for n in CANONICAL_NODES if n.node_id == src), None)
            tgt_node = next((n for n in CANONICAL_NODES if n.node_id == tgt), None)
            if not src_node or not tgt_node:
                continue
            x1, y1 = src_node.pos[0] * w, src_node.pos[1] * h
            x2, y2 = tgt_node.pos[0] * w, tgt_node.pos[1] * h
            canvas.create_line(x1, y1, x2, y2, fill="#3fb0ff", width=2, arrow=tk.LAST)

        for node in CANONICAL_NODES:
            x, y = node.pos[0] * w, node.pos[1] * h
            r = 18
            fill = "#ffd166" if node.node_id in active_nodes else "#303846"
            canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill, outline="#dddddd")
            canvas.create_text(x, y-26, text=node.label, fill="#e5e7eb", font=("Arial", 8))

    def _get_epistemic_task_model(self):
        if self.sim and getattr(self.sim, "task_model", None) is not None:
            return self.sim.task_model
        if not hasattr(self, "_epistemic_task_model_cache"):
            self._epistemic_task_model_cache = load_task_model(task_id="mars_colony")
        return self._epistemic_task_model_cache

    def _epistemic_selector_values(self):
        names = ["Team Union"]
        if self.sim:
            names.extend([str(getattr(a, "name", f"Agent {idx+1}")) for idx, a in enumerate(self.sim.agents)])
        return names

    def _refresh_epistemic_selector_values(self):
        values = self._epistemic_selector_values()
        if not hasattr(self, "epistemic_view_var"):
            return
        if self.epistemic_view_var.get() not in values:
            self.epistemic_view_var.set("Team Union")
        for name in ("dik_view_combo", "rule_view_combo"):
            combo = getattr(self, name, None)
            if combo is not None:
                combo.configure(values=values)

    def _stable_data_color(self, data_id):
        digest = hashlib.md5(str(data_id).encode("utf-8")).hexdigest()
        hue = (int(digest[:8], 16) % 360) / 360.0
        sat = 0.62
        val = 0.88
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    @staticmethod
    def _agent_held_sets(agent):
        data_ids = {getattr(item, "id", str(item)) for item in list(agent.mental_model.get("data", []))}
        info_ids = {getattr(item, "id", str(item)) for item in list(agent.mental_model.get("information", []))}
        knowledge_ids = {str(item) for item in list(getattr(agent.mental_model.get("knowledge"), "rules", []))}
        normalized_rule_ids = {normalize_rule_token(r) for r in knowledge_ids}
        return {
            "data": data_ids,
            "information": info_ids,
            "knowledge": knowledge_ids,
            "rules": normalized_rule_ids | knowledge_ids,
        }

    def _collect_epistemic_state(self):
        view = self.epistemic_view_var.get() if hasattr(self, "epistemic_view_var") else "Team Union"
        state = {"data": set(), "information": set(), "knowledge": set(), "rules": set()}
        if not self.sim:
            return state
        if view == "Team Union":
            for agent in self.sim.agents:
                held = self._agent_held_sets(agent)
                for key in state:
                    state[key].update(held[key])
            return state
        for agent in self.sim.agents:
            if str(getattr(agent, "name", "")) == view:
                held = self._agent_held_sets(agent)
                for key in state:
                    state[key].update(held[key])
                break
        return state

    def _element_is_held(self, element_id, element_type, state):
        if element_type == "data":
            return element_id in state["data"]
        if element_type == "information":
            return element_id in state["information"]
        if element_type == "knowledge":
            return element_id in state["knowledge"] or normalize_rule_token(element_id) in state["rules"]
        return element_id in state["data"] or element_id in state["information"] or element_id in state["knowledge"] or normalize_rule_token(element_id) in state["rules"]

    @staticmethod
    def _build_derivation_groups(task_model, *, output_type):
        groups = {}
        for derivation in task_model.derivations.values():
            if not derivation.enabled or str(derivation.output_type) != str(output_type):
                continue
            group = groups.setdefault(derivation.output_element_id, [])
            group.append(derivation)
        return [(output_id, groups[output_id]) for output_id in sorted(groups.keys())]

    def _pathway_satisfied(self, derivation, state, task_model):
        req_ids = [str(x) for x in (derivation.required_inputs or []) if str(x)]
        held = 0
        for req_id in req_ids:
            element = task_model.dik_elements.get(req_id)
            etype = element.element_type if element else None
            if self._element_is_held(req_id, etype, state):
                held += 1
        if derivation.min_required_count:
            return held >= int(derivation.min_required_count)
        return held == len(req_ids)

    @staticmethod
    def _abbreviated_element_id(element_id, element_type=None):
        text = str(element_id or "")
        prefixes = []
        if element_type == "data":
            prefixes.append("D_")
        elif element_type == "information":
            prefixes.append("I_")
        elif element_type == "knowledge":
            prefixes.append("K_")
        elif element_type == "rule":
            prefixes.append("R_")
        prefixes.extend(["D_", "I_", "K_", "R_"])
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    @staticmethod
    def _wrap_flow_rows(widths, max_width, hgap):
        rows = []
        current = []
        row_width = 0
        for idx, width in enumerate(widths):
            candidate = width if not current else row_width + hgap + width
            if current and candidate > max_width:
                rows.append(current)
                current = [idx]
                row_width = width
            else:
                current.append(idx)
                row_width = candidate
        if current:
            rows.append(current)
        return rows

    def _ensure_epistemic_fonts(self):
        if hasattr(self, "_epi_font_chip"):
            return
        self._epi_font_chip = tkfont.Font(root=self.root, family="Arial", size=8)
        self._epi_font_title = tkfont.Font(root=self.root, family="Arial", size=9, weight="bold")
        self._epi_font_section = tkfont.Font(root=self.root, family="Arial", size=11, weight="bold")

    def _measure_text(self, text, *, font_key="chip"):
        self._ensure_epistemic_fonts()
        font = self._epi_font_chip
        if font_key == "title":
            font = self._epi_font_title
        elif font_key == "section":
            font = self._epi_font_section
        return int(font.measure(str(text)))

    def _display_label(self, task_model, element_id, element_type=None):
        element = getattr(task_model, "dik_elements", {}).get(element_id)
        candidate = str(getattr(element, "label", "") or "").strip() if element else ""
        if candidate:
            return candidate
        return self._abbreviated_element_id(element_id, element_type=element_type)

    def _chip_style(self, chip_type, held, canonical_id):
        if chip_type == "data":
            accent = self._stable_data_color(canonical_id)
            return {"outline": accent, "fill": "#eff7ff" if held else "#f8fafc", "accent": accent}
        if chip_type == "knowledge":
            return {"outline": "#6b4f8c", "fill": "#e6dcff" if held else "#f6f8fa", "accent": "#6b4f8c"}
        if chip_type == "information":
            return {"outline": "#2f6f9f", "fill": "#d6ecff" if held else "#f6f8fa", "accent": "#2f6f9f"}
        return {"outline": "#7a8188", "fill": "#f6f8fa", "accent": "#7a8188"}

    def _chip_metrics(self, label):
        square = 8
        pad_x = 6
        inner_gap = 6
        outer_gap = 6
        text_w = self._measure_text(label, font_key="chip")
        width = pad_x + square + inner_gap + text_w + pad_x
        height = 20
        return {"square": square, "pad_x": pad_x, "inner_gap": inner_gap, "outer_gap": outer_gap, "width": width, "height": height}

    def _bind_epistemic_detail(self, canvas, item_id, detail_text):
        if not hasattr(self, "_epistemic_detail_var"):
            return
        canvas.tag_bind(item_id, "<Button-1>", lambda _e, txt=detail_text: self._epistemic_detail_var.set(txt))

    def _draw_chip(self, canvas, x, y, chip):
        metrics = self._chip_metrics(chip["display_label"])
        style = self._chip_style(chip["chip_type"], chip["held"], chip["canonical_id"])
        rect = canvas.create_rectangle(
            x,
            y,
            x + metrics["width"],
            y + metrics["height"],
            outline=style["outline"],
            fill=style["fill"],
            width=1.0,
        )
        square_y = y + (metrics["height"] - metrics["square"]) / 2
        canvas.create_rectangle(
            x + metrics["pad_x"],
            square_y,
            x + metrics["pad_x"] + metrics["square"],
            square_y + metrics["square"],
            outline=style["accent"],
            fill=style["accent"] if chip["held"] else "#ffffff",
            width=1.0,
        )
        text_x = x + metrics["pad_x"] + metrics["square"] + metrics["inner_gap"]
        canvas.create_text(text_x, y + metrics["height"] / 2, text=chip["display_label"], anchor="w", font=("Arial", 8), fill="#1f2933")
        detail = f"{chip['chip_type']} | {chip['display_label']} | canonical: {chip['canonical_id']} | {'held' if chip['held'] else 'not held'}"
        self._bind_epistemic_detail(canvas, rect, detail)
        return metrics

    def _draw_chip_flow(self, canvas, x, y, max_width, chips, *, draw=True):
        if not chips:
            return 0
        widths = [self._chip_metrics(chip["display_label"])["width"] for chip in chips]
        hgap = 6
        vgap = 6
        rows = self._wrap_flow_rows(widths, max(80, int(max_width)), hgap)
        row_h = 20
        if draw:
            for row_idx, row in enumerate(rows):
                cur_x = x
                row_y = y + row_idx * (row_h + vgap)
                for idx_in_row, chip_idx in enumerate(row):
                    chip = chips[chip_idx]
                    self._draw_chip(canvas, cur_x, row_y, chip)
                    cur_x += widths[chip_idx] + (hgap if idx_in_row < len(row) - 1 else 0)
        return len(rows) * row_h + max(0, len(rows) - 1) * vgap

    def _canvas_content_width(self, canvas):
        width = int(canvas.winfo_width() or 0)
        if width <= 1:
            width = int(canvas.winfo_reqwidth() or 900)
        return max(420, width - 24)

    def create_dik_derivations_tab(self):
        self.tab_dik_derivations = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dik_derivations, text="DIK Derivations")
        controls = ttk.Frame(self.tab_dik_derivations, padding=(6, 6, 6, 0))
        controls.pack(fill="x")
        ttk.Label(controls, text="View").pack(side="left")
        self.epistemic_view_var = StringVar(value="Team Union")
        self.epistemic_unsatisfied_only_var = BooleanVar(value=False)
        self.dik_view_combo = ttk.Combobox(controls, textvariable=self.epistemic_view_var, values=["Team Union"], state="readonly", width=24)
        self.dik_view_combo.pack(side="left", padx=(6, 0))
        self.dik_view_combo.bind("<<ComboboxSelected>>", lambda *_: (self.update_dik_derivations_tab(), self.update_rule_derivations_tab()))
        ttk.Checkbutton(
            controls,
            text="Show only unsatisfied outputs",
            variable=self.epistemic_unsatisfied_only_var,
            command=lambda: (self.update_dik_derivations_tab(), self.update_rule_derivations_tab()),
        ).pack(side="left", padx=(12, 0))

        canvas_wrap = ttk.Frame(self.tab_dik_derivations, padding=6)
        canvas_wrap.pack(fill="both", expand=True)
        self.dik_derivations_canvas = tk.Canvas(canvas_wrap, background="#ffffff")
        y_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.dik_derivations_canvas.yview)
        self.dik_derivations_canvas.configure(yscrollcommand=y_scroll.set)
        self.dik_derivations_canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        self.dik_derivations_canvas.bind("<Configure>", lambda *_: self.update_dik_derivations_tab())
        self._epistemic_detail_var = StringVar(value="Select an item to see canonical details.")
        ttk.Label(self.tab_dik_derivations, textvariable=self._epistemic_detail_var, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

    def create_rule_derivations_tab(self):
        self.tab_rule_derivations = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rule_derivations, text="Rule Derivations")
        controls = ttk.Frame(self.tab_rule_derivations, padding=(6, 6, 6, 0))
        controls.pack(fill="x")
        ttk.Label(controls, text="View").pack(side="left")
        self.rule_view_combo = ttk.Combobox(controls, textvariable=self.epistemic_view_var, values=["Team Union"], state="readonly", width=24)
        self.rule_view_combo.pack(side="left", padx=(6, 0))
        self.rule_view_combo.bind("<<ComboboxSelected>>", lambda *_: (self.update_dik_derivations_tab(), self.update_rule_derivations_tab()))
        ttk.Checkbutton(
            controls,
            text="Show only unsatisfied outputs",
            variable=self.epistemic_unsatisfied_only_var,
            command=lambda: (self.update_dik_derivations_tab(), self.update_rule_derivations_tab()),
        ).pack(side="left", padx=(12, 0))

        canvas_wrap = ttk.Frame(self.tab_rule_derivations, padding=6)
        canvas_wrap.pack(fill="both", expand=True)
        self.rule_derivations_canvas = tk.Canvas(canvas_wrap, background="#ffffff")
        y_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.rule_derivations_canvas.yview)
        self.rule_derivations_canvas.configure(yscrollcommand=y_scroll.set)
        self.rule_derivations_canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        self.rule_derivations_canvas.bind("<Configure>", lambda *_: self.update_rule_derivations_tab())

    def update_dik_derivations_tab(self):
        if not hasattr(self, "dik_derivations_canvas"):
            return
        self._refresh_epistemic_selector_values()
        canvas = self.dik_derivations_canvas
        canvas.delete("all")
        task_model = self._get_epistemic_task_model()
        state = self._collect_epistemic_state()
        left = 12
        width = self._canvas_content_width(canvas)
        y = 10
        canvas.create_text(left, y, text="Data Elements", anchor="nw", font=("Arial", 11, "bold"))
        y += 24
        data_ids = sorted([eid for eid, elem in task_model.dik_elements.items() if elem.enabled and elem.element_type == "data"])
        data_chips = [
            {
                "canonical_id": data_id,
                "display_label": self._display_label(task_model, data_id, "data"),
                "chip_type": "data",
                "held": self._element_is_held(data_id, "data", state),
            }
            for data_id in data_ids
        ]
        y += self._draw_chip_flow(canvas, left + 4, y, width - 20, data_chips, draw=True) + 18

        for section_name, output_type, border_color, fill_active, fill_inactive in [
            ("Information Derivations", "information", "#2f6f9f", "#d6ecff", "#f8fbff"),
            ("Knowledge Derivations", "knowledge", "#6b4f8c", "#e6dcff", "#fbf8ff"),
        ]:
            canvas.create_text(left, y, text=section_name, anchor="nw", font=("Arial", 11, "bold"))
            y += 22
            groups = self._build_derivation_groups(task_model, output_type=output_type)
            if not groups:
                canvas.create_text(left + 8, y, text="(no derivations)", anchor="nw", font=("Arial", 9), fill="#666666")
                y += 22
                continue
            for output_id, pathways in groups:
                output_held = self._element_is_held(output_id, output_type, state)
                if self.epistemic_unsatisfied_only_var.get() and output_held:
                    continue
                display_output = self._display_label(task_model, output_id, output_type)
                heading = canvas.create_text(
                    left + 6,
                    y,
                    text=display_output,
                    anchor="nw",
                    font=("Arial", 10, "bold"),
                    fill="#14395f" if output_held else "#7a8188",
                )
                heading_w = self._measure_text(display_output, font_key="title")
                tag_x = left + 14 + heading_w
                canvas.create_rectangle(tag_x, y + 1, tag_x + 82, y + 17, outline="#9aa4ae", fill="#e9f6ea" if output_held else "#f5e9ea")
                canvas.create_text(tag_x + 41, y + 9, text="achieved" if output_held else "not achieved", anchor="center", font=("Arial", 8))
                self._bind_epistemic_detail(
                    canvas,
                    heading,
                    f"{output_type} output | {display_output} | canonical: {output_id} | {'achieved' if output_held else 'not achieved'}",
                )
                y += 24
                for derivation in pathways:
                    satisfied = self._pathway_satisfied(derivation, state, task_model)
                    req_chips = []
                    for req_id in derivation.required_inputs:
                        element = task_model.dik_elements.get(req_id)
                        req_type = element.element_type if element else "unknown"
                        req_chips.append(
                            {
                                "canonical_id": req_id,
                                "display_label": self._display_label(task_model, req_id, req_type),
                                "chip_type": req_type,
                                "held": self._element_is_held(req_id, req_type, state),
                            }
                        )
                    card_x = left + 18
                    card_w = width - 36
                    chips_h = self._draw_chip_flow(canvas, card_x + 14, y + 34, card_w - 28, req_chips, draw=False)
                    card_h = 40 + max(20, chips_h)
                    fill = fill_active if satisfied else fill_inactive
                    canvas.create_rectangle(card_x, y, card_x + card_w, y + card_h, outline=border_color, fill=fill, width=1.5)
                    title = canvas.create_text(
                        card_x + 8,
                        y + 6,
                        text=f"{derivation.derivation_id} ({'path satisfied' if satisfied else 'path unsatisfied'})",
                        anchor="nw",
                        font=("Arial", 9, "bold"),
                    )
                    self._bind_epistemic_detail(
                        canvas,
                        title,
                        f"derivation | {derivation.derivation_id} | output: {output_id} | {'path satisfied' if satisfied else 'path unsatisfied'}",
                    )
                    self._draw_chip_flow(canvas, card_x + 14, y + 34, card_w - 28, req_chips, draw=True)
                    y += card_h + 8
                y += 8
            y += 10
        canvas.configure(scrollregion=(0, 0, width + 24, max(500, y + 20)))

    def update_rule_derivations_tab(self):
        if not hasattr(self, "rule_derivations_canvas"):
            return
        self._refresh_epistemic_selector_values()
        canvas = self.rule_derivations_canvas
        canvas.delete("all")
        task_model = self._get_epistemic_task_model()
        state = self._collect_epistemic_state()
        left = 12
        width = self._canvas_content_width(canvas)
        y = 10
        canvas.create_text(left, y, text="Rule Derivations", anchor="nw", font=("Arial", 11, "bold"))
        y += 24
        for rule_id in sorted(task_model.rules.keys()):
            rule = task_model.rules[rule_id]
            if not rule.enabled:
                continue
            req_info = [r for r in rule.required_information if r]
            req_knowledge = [r for r in rule.required_knowledge if r]
            prereq_ok = all(self._element_is_held(i, "information", state) for i in req_info) and all(
                self._element_is_held(k, "knowledge", state) for k in req_knowledge
            )
            achieved = normalize_rule_token(rule.rule_id) in state["rules"] or rule.rule_id in state["knowledge"]
            if self.epistemic_unsatisfied_only_var.get() and achieved:
                continue
            info_chips = [
                {
                    "canonical_id": req_id,
                    "display_label": self._display_label(task_model, req_id, "information"),
                    "chip_type": "information",
                    "held": self._element_is_held(req_id, "information", state),
                }
                for req_id in req_info
            ]
            knowledge_chips = [
                {
                    "canonical_id": req_id,
                    "display_label": self._display_label(task_model, req_id, "knowledge"),
                    "chip_type": "knowledge",
                    "held": self._element_is_held(req_id, "knowledge", state),
                }
                for req_id in req_knowledge
            ]
            card_x = left + 4
            card_w = width - 8
            info_h = self._draw_chip_flow(canvas, card_x + 16, y + 62, card_w - 32, info_chips, draw=False) if info_chips else 0
            info_section_h = (16 + info_h + 6) if info_chips else 0
            know_h = self._draw_chip_flow(canvas, card_x + 16, y + 62 + info_section_h, card_w - 32, knowledge_chips, draw=False) if knowledge_chips else 0
            know_section_h = (16 + know_h + 6) if knowledge_chips else 0
            card_h = 62 + info_section_h + know_section_h + 8
            canvas.create_rectangle(card_x, y, card_x + card_w, y + card_h, outline="#444444", width=1.6, fill="#edf9ed" if achieved else "#fafafa")
            display_rule = self._abbreviated_element_id(rule.rule_id, element_type="rule")
            title = canvas.create_text(card_x + 8, y + 8, text=f"{display_rule} ({'achieved' if achieved else 'not achieved'})", anchor="nw", font=("Arial", 10, "bold"))
            self._bind_epistemic_detail(canvas, title, f"rule | {display_rule} | canonical: {rule.rule_id} | {'achieved' if achieved else 'not achieved'}")
            canvas.create_text(card_x + 8, y + 26, text=f"Prerequisites: {'satisfied' if prereq_ok else 'unsatisfied'}", anchor="nw", font=("Arial", 9), fill="#2f7a2f" if prereq_ok else "#9a3c3c")
            section_y = y + 44
            if info_chips:
                canvas.create_text(card_x + 12, section_y, text="Required Information", anchor="nw", font=("Arial", 8, "bold"))
                self._draw_chip_flow(canvas, card_x + 16, section_y + 16, card_w - 32, info_chips, draw=True)
                section_y += info_section_h
            if knowledge_chips:
                canvas.create_text(card_x + 12, section_y, text="Required Knowledge", anchor="nw", font=("Arial", 8, "bold"))
                self._draw_chip_flow(canvas, card_x + 16, section_y + 16, card_w - 32, knowledge_chips, draw=True)
            y += card_h + 12
        canvas.configure(scrollregion=(0, 0, width + 24, max(500, y + 20)))

    def create_event_monitor_tab(self):
        self.tab_event = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_event, text="Event Monitor")

        # Section 1: Agent Activities
        frame_agent_activities = ttk.LabelFrame(self.tab_event, text="Agent Activities", padding=10)
        frame_agent_activities.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.agent_activity_text = tk.Text(frame_agent_activities, width=45, height=20, wrap="word")
        self.agent_activity_text.pack(expand=True, fill="both")

        # Section 2: Interaction State Machines
        frame_interaction_states = ttk.LabelFrame(self.tab_event, text="Interaction State Machine", padding=10)
        frame_interaction_states.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.interaction_state_text = tk.Text(frame_interaction_states, width=45, height=20, wrap="word")
        self.interaction_state_text.pack(expand=True, fill="both")

        # Section 3: Spatial Location States
        frame_location_states = ttk.LabelFrame(self.tab_event, text="Agent Zone States", padding=10)
        frame_location_states.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.zone_state_text = tk.Text(frame_location_states, width=92, height=10, wrap="word")
        self.zone_state_text.pack(expand=True, fill="both")

        self.tab_event.rowconfigure(0, weight=3)
        self.tab_event.rowconfigure(1, weight=1)
        self.tab_event.columnconfigure(0, weight=1)
        self.tab_event.columnconfigure(1, weight=1)

    def build_agent_configs(self):
        agent_configs = []
        profile_constructs = {
            "High": 1.0,
            "Low": 0.0,
        }
        profile_traits = {
            "High_Team": {
                "communication_propensity": 1.0,
                "goal_alignment": 1.0,
                "help_tendency": 1.0
            },
            "Low_Team": {
                "communication_propensity": 0.3,
                "goal_alignment": 0.4,
                "help_tendency": 0.2
            },
            "High_Task": {
                "build_speed": 1.0,
                "rule_accuracy": 1.0
            },
            "Low_Task": {
                "build_speed": 0.5,
                "rule_accuracy": 0.5
            }
        }

        selected_count = max(1, min(self.MAX_AGENT_PANELS, int(self.num_agents_var.get())))
        for role in self.agent_card_order[:selected_count]:
            if self.active_roles[role].get():
                team_pot = self.agent_profiles[role]["team"].get()
                task_pot = self.agent_profiles[role]["task"].get()

                traits = {}
                traits.update(profile_traits[f"{team_pot}_Team"])
                traits.update(profile_traits[f"{task_pot}_Task"])

                for trait_key, var in self.agent_traits[role].items():
                    traits[trait_key] = var.get()  # override with current slider value

                selected_packets = [
                    packet_name
                    for packet_name, is_enabled in self.packet_access[role].items()
                    if is_enabled.get()
                ]

                identity = self.agent_identity[role]
                brain_settings = self.agent_brain_settings[role]
                planner_settings = self.agent_planner_settings[role]
                display_name = identity["display_name"].get().strip() or role
                alias = identity["alias"].get().strip()
                backend = brain_settings["backend"].get().strip()
                model = brain_settings["local_model"].get().strip()
                fallback_backend = brain_settings["fallback_backend"].get().strip()
                planner_cadence = max(1, int(planner_settings["planner_interval_steps"].get()))
                planner_timeout = max(0.1, float(planner_settings["planner_timeout_seconds"].get()))
                planner_max_retries = max(0, int(planner_settings["planner_max_retries"].get()))
                planner_request_policy = str(planner_settings["planner_request_policy"].get() or self.PLANNER_DEFAULTS["planner_request_policy"]).strip() or self.PLANNER_DEFAULTS["planner_request_policy"]
                planning_interval_steps = max(1, int(planner_settings["planning_interval_steps"].get()))
                dik_integration_cooldown_steps = max(1, int(planner_settings["dik_integration_cooldown_steps"].get()))
                dik_integration_batch_threshold = max(1, int(planner_settings["dik_integration_batch_threshold"].get()))
                degraded_threshold = max(1, int(planner_settings["degraded_consecutive_failures_threshold"].get()))
                degraded_cooldown = max(0.0, float(planner_settings["degraded_cooldown_seconds"].get()))
                degraded_interval_multiplier = max(1.0, float(planner_settings["degraded_step_interval_multiplier"].get()))
                per_agent_timeout = max(0.1, float(brain_settings["timeout_s"].get()))
                per_agent_max_retries = max(0, int(brain_settings["max_retries"].get()))

                brain_config = {
                    "backend": backend,
                    "local_model": model,
                    "fallback_backend": fallback_backend,
                    "timeout_s": per_agent_timeout,
                    "max_retries": per_agent_max_retries,
                }
                planner_config = {
                    "planner_interval_steps": planner_cadence,
                    "planner_timeout_seconds": planner_timeout,
                    "planner_max_retries": planner_max_retries,
                    "planner_request_policy": planner_request_policy,
                    "planning_interval_steps": planning_interval_steps,
                    "dik_integration_cooldown_steps": dik_integration_cooldown_steps,
                    "dik_integration_batch_threshold": dik_integration_batch_threshold,
                    "degraded_consecutive_failures_threshold": degraded_threshold,
                    "degraded_cooldown_seconds": degraded_cooldown,
                    "degraded_step_interval_multiplier": degraded_interval_multiplier,
                }

                agent_configs.append({
                    "name": role,
                    "display_name": display_name,
                    "alias": alias,
                    "label": alias or role,
                    "role": role,
                    "template_id": identity["template_id"],
                    "constructs": {
                        "teamwork_potential": profile_constructs[team_pot],
                        "taskwork_potential": profile_constructs[task_pot],
                    },
                    "mechanism_overrides": dict(traits),
                    "traits": traits,
                    "packet_access": selected_packets,
                    "brain_config": {k: v for k, v in brain_config.items() if v},
                    "planner_config": planner_config,
                })

        return agent_configs

    def _collect_brain_backend_config(self):
        backend = (self.brain_backend_var.get() or self.BACKEND_DEFAULTS["brain_backend"]).strip() or self.BACKEND_DEFAULTS["brain_backend"]
        options = {
            "local_model": self.local_model_var.get().strip() or self.BACKEND_DEFAULTS["local_model"],
            "local_base_url": self.local_base_url_var.get().strip() or self.BACKEND_DEFAULTS["local_base_url"],
            "timeout_s": max(0.1, float(self.local_timeout_var.get())),
            "fallback_backend": (self.fallback_backend_var.get() or self.BACKEND_DEFAULTS["fallback_backend"]).strip() or self.BACKEND_DEFAULTS["fallback_backend"],
        }
        return backend, options


    def _collect_global_planner_config(self):
        return {
            "enable_startup_llm_sanity": bool(self.enable_startup_llm_sanity_var.get()),
            "startup_llm_sanity_timeout_seconds": max(0.1, float(self.startup_llm_sanity_timeout_var.get())),
            "startup_llm_sanity_max_sources": max(1, int(self.startup_llm_sanity_max_sources_var.get())),
            "startup_llm_sanity_max_items_per_type": max(1, int(self.startup_llm_sanity_max_items_var.get())),
            "startup_llm_sanity_raw_response_max_chars": max(500, int(self.startup_llm_sanity_raw_max_chars_var.get())),
            "enable_bootstrap_summary_reuse": bool(self.bootstrap_reuse_enabled_var.get()),
            "bootstrap_summary_max_chars": max(80, int(self.bootstrap_summary_max_chars_var.get())),
            "high_latency_local_llm_mode": bool(self.high_latency_local_llm_mode_var.get()),
            "unrestricted_local_qwen_mode": bool(self.unrestricted_local_qwen_mode_var.get()),
            "planner_timeout_seconds": max(0.1, float(self.planner_timeout_seconds_var.get())),
            "planner_request_policy": str(self.planner_request_policy_var.get() or self.PLANNER_DEFAULTS["planner_request_policy"]).strip() or self.PLANNER_DEFAULTS["planner_request_policy"],
            "planning_interval_steps": max(1, int(self.planning_interval_steps_var.get())),
            "dik_integration_cooldown_steps": max(1, int(self.dik_integration_cooldown_steps_var.get())),
            "dik_integration_batch_threshold": max(1, int(self.dik_integration_batch_threshold_var.get())),
            "warmup_timeout_seconds": max(0.1, float(self.warmup_timeout_var.get())),
            "startup_llm_sanity_completion_max_tokens": max(256, int(self.startup_llm_sanity_completion_tokens_var.get())),
            "planner_completion_max_tokens": max(256, int(self.planner_completion_tokens_var.get())),
            "high_latency_stale_result_grace_s": max(0.0, float(self.high_latency_stale_result_grace_var.get())),
        }

    @staticmethod
    def _load_construction_defaults():
        path = Path(__file__).resolve().parent / "config" / "tasks" / "mars_colony" / "construction_parameters.json"
        if not path.exists():
            return dict(ConstructionManager.DEFAULT_PARAMETERS)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return dict(ConstructionManager.DEFAULT_PARAMETERS)
        defaults = dict(ConstructionManager.DEFAULT_PARAMETERS)
        if isinstance(payload, dict):
            defaults.update(payload)
        return defaults

    def _collect_construction_parameters(self):
        return {
            "pile_a_quantity": max(0, int(self.construction_param_vars["pile_a_quantity"].get())),
            "pile_c_quantity": max(0, int(self.construction_param_vars["pile_c_quantity"].get())),
            "housing_cost": max(1, int(self.construction_param_vars["housing_cost"].get())),
            "greenhouse_cost": max(1, int(self.construction_param_vars["greenhouse_cost"].get())),
            "water_generator_cost": max(1, int(self.construction_param_vars["water_generator_cost"].get())),
            "bridge_bc_cost": max(1, int(self.construction_param_vars["bridge_bc_cost"].get())),
            "site_a_capacity": max(1, int(self.construction_param_vars["site_a_capacity"].get())),
            "site_b_capacity": max(1, int(self.construction_param_vars["site_b_capacity"].get())),
            "site_c_capacity": max(1, int(self.construction_param_vars["site_c_capacity"].get())),
            "move_time_per_unit": max(1, int(self.construction_param_vars["move_time_per_unit"].get())),
            "carry_capacity": max(1, int(self.construction_param_vars["carry_capacity"].get())),
        }

    def _collect_experiment_settings(self):
        selected_backend, backend_options = self._collect_brain_backend_config()
        planner_config = self._collect_global_planner_config()
        construction_parameters = self._collect_construction_parameters()
        return {
            "agent_configs": self.build_agent_configs(),
            "num_runs": max(1, int(self.num_runs.get())),
            "timesteps_per_run": max(1, int(self.timesteps_per_run_var.get())),
            "speed": self.speed_multiplier.get(),
            "experiment_name": self.experiment_name_var.get(),
            "phases": MISSION_PHASES,
            "flash_mode": self.flash_mode.get(),
            "brain_backend": selected_backend,
            "brain_backend_options": backend_options,
            "planner_config": planner_config,
            "construction_parameters": construction_parameters,
            "base_dt": 1.0,
        }

    @staticmethod
    def _resolve_default_packet_access(default_packet_access):
        if default_packet_access:
            return {str(packet_name) for packet_name in default_packet_access}
        return {"Team_Packet"}

    @staticmethod
    def _build_batch_run_experiment_name(experiment_name, run_index, num_runs):
        base_name = (experiment_name or "experiment").strip() or "experiment"
        suffix = f"_run{run_index:03d}"
        return f"{base_name}{suffix}" if num_runs > 1 else base_name

    def _add_help_text(self, parent, row, text):
        ttk.Label(parent, text=text, foreground="#5f6b7a", wraplength=620, justify="left").grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            padx=(0, 8),
            pady=(0, 3),
        )

    def _resolve_agent_effective_brain_settings(self, role):
        backend_override = self.agent_brain_settings[role]["backend"].get().strip()
        model_override = self.agent_brain_settings[role]["local_model"].get().strip()
        fallback_override = self.agent_brain_settings[role]["fallback_backend"].get().strip()
        return {
            "backend": backend_override or self.brain_backend_var.get().strip() or self.BACKEND_DEFAULTS["brain_backend"],
            "local_model": model_override or self.local_model_var.get().strip() or self.BACKEND_DEFAULTS["local_model"],
            "fallback_backend": fallback_override or self.fallback_backend_var.get().strip() or self.BACKEND_DEFAULTS["fallback_backend"],
        }

    def _refresh_all_agent_inheritance_display(self):
        for role in getattr(self, "agent_card_order", []):
            self._update_agent_inheritance_display(role)

    def _update_agent_inheritance_display(self, role):
        note_vars = self.agent_inheritance_note_vars.get(role)
        if not note_vars:
            return
        effective = self._resolve_agent_effective_brain_settings(role)
        for key, global_label in [
            ("backend", "Backend"),
            ("local_model", "Model"),
            ("fallback_backend", "Fallback"),
        ]:
            override_value = self.agent_brain_settings[role][key].get().strip()
            if override_value:
                note_vars[key].set(f"Override active: {override_value}")
            else:
                note_vars[key].set(f"Inherited from global {global_label}: {effective[key]}")
        summary_vars = self.agent_effective_summary_vars.get(role)
        if summary_vars:
            summary_vars["backend"].set(f"Effective Backend: {effective['backend']}")
            summary_vars["local_model"].set(f"Effective Model: {effective['local_model']}")
            summary_vars["fallback_backend"].set(f"Effective Fallback: {effective['fallback_backend']}")

    def _update_visible_agent_cards(self):
        if not hasattr(self, "agent_cards"):
            return
        selected_count = max(1, min(self.MAX_AGENT_PANELS, int(self.num_agents_var.get())))
        for idx, role in enumerate(self.agent_card_order):
            card = self.agent_cards[role]
            if idx < selected_count:
                card.grid()
            else:
                card.grid_remove()

    def _is_local_backend_selected(self):
        selected = (self.brain_backend_var.get() or "").strip().lower()
        return selected in {"local_http", "openai_compatible_local", "ollama_local", "ollama"}

    def _update_backend_field_states(self):
        state = "normal" if self._is_local_backend_selected() else "disabled"
        for widget in getattr(self, "_local_backend_widgets", []):
            widget.configure(state=state)
        self._refresh_all_agent_inheritance_display()

    def _update_backend_status_display(self):
        if not hasattr(self, "backend_status_var"):
            return
        if not self.sim:
            configured = self.brain_backend_var.get() if hasattr(self, "brain_backend_var") else "rule_brain"
            self.backend_status_var.set(f"Backend (configured/effective): {configured} / {configured}")
            return
        configured = getattr(self.sim, "configured_brain_backend", "unknown")
        effective = getattr(self.sim, "effective_brain_backend", configured)
        fallback_count = getattr(self.sim, "backend_fallback_count", 0)
        suffix = f" (fallbacks={fallback_count})" if fallback_count else ""
        self.backend_status_var.set(f"Backend (configured/effective): {configured} / {effective}{suffix}")

    @staticmethod
    def _format_elapsed_duration(seconds):
        total = max(0.0, float(seconds or 0.0))
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = total - (hours * 3600 + minutes * 60)
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"

    @staticmethod
    def _format_barrier_summary(status):
        if not status or not status.get("barrier_active"):
            return "Barrier: none"
        request_count = len(status.get("blocking_request_ids") or [])
        agent_count = len(status.get("blocking_agent_ids") or [])
        return f"Barrier: active ({request_count} requests, {agent_count} agents)"

    @staticmethod
    def _resolve_last_planner_latency_text(sim):
        logger = getattr(sim, "logger", None)
        if logger is None or not hasattr(logger, "get_recent_planner_traces"):
            return "Last planner latency: n/a"
        rows = list(logger.get_recent_planner_traces(40))
        for row in reversed(rows):
            for key in ("planner_latency_s", "llm_latency_s", "latency_s", "duration_s"):
                if key in row and row.get(key) is not None:
                    return f"Last planner latency: {MarsColonyInterface._format_elapsed_duration(float(row.get(key)))}"
        return "Last planner latency: n/a"

    def _update_observability_status_display(self):
        required_vars = [
            "wallclock_elapsed_var",
            "sim_time_elapsed_var",
            "current_pause_elapsed_var",
            "cumulative_pause_wait_var",
            "barrier_summary_var",
            "pause_count_var",
            "last_barrier_duration_var",
            "last_planner_latency_var",
        ]
        if any(not hasattr(self, name) for name in required_vars):
            return
        if not self.sim or not hasattr(self.sim, "get_observability_status"):
            self.wallclock_elapsed_var.set("Wall-clock: 00:00:00.0")
            self.sim_time_elapsed_var.set("Sim-time: 00:00:00.0")
            self.current_pause_elapsed_var.set("Current cognition-pause: none")
            self.cumulative_pause_wait_var.set("Cumulative cognition wait: 00:00:00.0")
            self.barrier_summary_var.set("Barrier: none")
            self.pause_count_var.set("Cognition pauses: 0")
            self.last_barrier_duration_var.set("Last barrier: none")
            self.last_planner_latency_var.set("Last planner latency: n/a")
            return
        status = self.sim.get_observability_status(now_wallclock=time.perf_counter())
        self.wallclock_elapsed_var.set(f"Wall-clock: {self._format_elapsed_duration(status.get('run_wallclock_elapsed_s', 0.0))}")
        self.sim_time_elapsed_var.set(f"Sim-time: {self._format_elapsed_duration(status.get('sim_time_elapsed_s', 0.0))}")
        if status.get("barrier_active"):
            self.current_pause_elapsed_var.set(
                f"Current cognition-pause: {self._format_elapsed_duration(status.get('current_cognition_pause_elapsed_s', 0.0))}"
            )
        else:
            self.current_pause_elapsed_var.set("Current cognition-pause: none")
        self.cumulative_pause_wait_var.set(
            f"Cumulative cognition wait: {self._format_elapsed_duration(status.get('cumulative_cognition_wait_s', 0.0))}"
        )
        self.barrier_summary_var.set(self._format_barrier_summary(status))
        self.pause_count_var.set(f"Cognition pauses: {int(status.get('barrier_pause_count', 0) or 0)}")
        last_barrier_s = float(status.get("last_barrier_duration_s", 0.0) or 0.0)
        if last_barrier_s > 0.0:
            self.last_barrier_duration_var.set(f"Last barrier: {self._format_elapsed_duration(last_barrier_s)}")
        else:
            self.last_barrier_duration_var.set("Last barrier: none")
        self.last_planner_latency_var.set(self._resolve_last_planner_latency_text(self.sim))

    def apply_experiment_settings(self):
        sim = self._build_simulation_from_settings()
        self._finalize_simulation_install(sim)

    def _build_simulation_from_settings(self, startup_progress_callback=None):
        settings = self._collect_experiment_settings()
        print("=== Experiment Settings ===")
        print("Speed Multiplier:", settings["speed"])
        print("Flash Mode:", settings["flash_mode"])
        print("Number of Runs:", settings["num_runs"])
        print("Timesteps per Run:", settings["timesteps_per_run"])
        print("Brain Backend:", settings["brain_backend"])
        print("Brain Backend Options:", settings["brain_backend_options"])
        print("Planner Config:", settings["planner_config"])
        print("Construction Parameters:", settings["construction_parameters"])

        agent_configs = settings["agent_configs"]
        for agent in settings["agent_configs"]:
            print(f"\n{agent['name']} ({agent['role']})")
            for k, v in agent["traits"].items():
                print(f"  {k}: {v}")
            print(f"  Packet Access: {agent['packet_access']}")

        return SimulationState(
            agent_configs=agent_configs,
            num_runs=settings["num_runs"],
            speed=settings["speed"],
            experiment_name=settings["experiment_name"],
            phases=settings["phases"],
            flash_mode=settings["flash_mode"],
            brain_backend=settings["brain_backend"],
            brain_backend_options=settings["brain_backend_options"],
            planner_config=settings["planner_config"],
            construction_parameters=settings["construction_parameters"],
            startup_progress_callback=startup_progress_callback,
        )

    def _finalize_simulation_install(self, sim):
        self.sim = sim
        self.run_state = self.STATE_IDLE
        self._cancel_run_loop()
        self._update_control_states()

        self.update_environment_plot()
        self._sync_construction_summaries()
        self._update_system_log()
        self._update_backend_status_display()
        self._update_observability_status_display()
        self.update_interaction_tab()
        self.update_brain_tab()
        self.update_dik_derivations_tab()
        self.update_rule_derivations_tab()

    def _create_startup_dialog(self):
        self._close_startup_dialog()
        dialog = tk.Toplevel(self.root)
        dialog.title("Initializing Simulation")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        self._startup_status_var = StringVar(value="Preparing startup…")
        ttk.Label(frame, textvariable=self._startup_status_var).pack(fill="x", pady=(0, 8))
        self._startup_progress = DoubleVar(value=0.0)
        self._startup_progressbar = ttk.Progressbar(frame, mode="indeterminate", length=280)
        self._startup_progressbar.pack(fill="x")
        self._startup_progressbar.start(12)
        self._startup_dialog = dialog

    def _close_startup_dialog(self):
        if self._startup_progressbar is not None:
            try:
                self._startup_progressbar.stop()
            except Exception:
                pass
        if self._startup_dialog is not None:
            try:
                self._startup_dialog.grab_release()
            except Exception:
                pass
            try:
                self._startup_dialog.destroy()
            except Exception:
                pass
        self._startup_dialog = None
        self._startup_status_var = None
        self._startup_progress = None
        self._startup_progressbar = None

    def _startup_progress_callback_factory(self):
        def _callback(stage, current, total, agent_name, detail):
            if not self._startup_queue:
                return
            self._startup_queue.put(
                {
                    "type": "progress",
                    "stage": stage,
                    "current": int(current or 0),
                    "total": max(0, int(total or 0)),
                    "agent_name": agent_name,
                    "detail": detail,
                }
            )

        return _callback

    def _startup_worker_main(self):
        try:
            if self._startup_queue:
                self._startup_queue.put({"type": "startup_begin"})
            sim = self._build_simulation_from_settings(
                startup_progress_callback=self._startup_progress_callback_factory(),
            )
            if self._startup_queue:
                self._startup_queue.put({"type": "startup_success", "sim": sim})
        except Exception as exc:  # noqa: BLE001
            if self._startup_queue:
                self._startup_queue.put(
                    {
                        "type": "startup_failure",
                        "error_message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

    def _begin_async_startup(self):
        self.run_state = self.STATE_STARTING
        self._cancel_run_loop()
        self._update_control_states()
        self._create_startup_dialog()
        self._startup_queue = queue.Queue()
        self._startup_worker = threading.Thread(target=self._startup_worker_main, daemon=True)
        self._startup_worker.start()
        self._schedule_startup_poll()

    def _schedule_startup_poll(self):
        if self.run_state == self.STATE_STARTING:
            self._startup_poll_job = self.root.after(75, self._poll_startup_queue)

    def _poll_startup_queue(self):
        self._startup_poll_job = None
        if self._startup_queue is None:
            return
        should_reschedule = self.run_state == self.STATE_STARTING
        while True:
            try:
                event = self._startup_queue.get_nowait()
            except queue.Empty:
                break
            should_reschedule = self._handle_startup_event(event)
            if not should_reschedule:
                break
        if should_reschedule and self.run_state == self.STATE_STARTING:
            self._schedule_startup_poll()

    def _handle_startup_event(self, event):
        event_type = event.get("type")
        if event_type in {"startup_begin", "progress"}:
            self._update_startup_progress(event)
            return True
        if event_type == "startup_success":
            self._finalize_startup_success(event.get("sim"))
            return False
        if event_type == "startup_failure":
            self._finalize_startup_failure(event.get("error_message"), event.get("traceback"))
            return False
        return True

    def _update_startup_progress(self, event):
        stage = event.get("stage") or event.get("type") or "startup"
        current = max(0, int(event.get("current", 0) or 0))
        total = max(0, int(event.get("total", 0) or 0))
        agent_name = event.get("agent_name")
        detail = event.get("detail")

        if self._startup_status_var is not None:
            label = detail or stage.replace("_", " ").title()
            if agent_name:
                label = f"{label} ({agent_name})"
            self._startup_status_var.set(label)

        if self._startup_progressbar is None:
            return
        if total > 0:
            if str(self._startup_progressbar.cget("mode")) != "determinate":
                self._startup_progressbar.stop()
                self._startup_progressbar.configure(mode="determinate", maximum=total, variable=self._startup_progress)
            if self._startup_progress is not None:
                self._startup_progress.set(min(current, total))
        elif str(self._startup_progressbar.cget("mode")) != "indeterminate":
            self._startup_progressbar.configure(mode="indeterminate", variable="")
            self._startup_progressbar.start(12)

    def _finalize_startup_success(self, sim):
        self._close_startup_dialog()
        self._startup_queue = None
        self._startup_worker = None
        self._finalize_simulation_install(sim)
        self.run_state = self.STATE_RUNNING
        self._update_control_states()
        self._schedule_next_tick()

    def _finalize_startup_failure(self, error_message, tb_text=None):
        if tb_text:
            print(tb_text)
        self._close_startup_dialog()
        self._startup_queue = None
        self._startup_worker = None
        self.sim = None
        self.run_state = self.STATE_IDLE
        self._cancel_run_loop()
        self._update_control_states()
        messagebox.showerror("Startup Failed", error_message or "Simulation initialization failed.")

    def _cancel_run_loop(self):
        if self._run_loop_job is not None:
            self.root.after_cancel(self._run_loop_job)
            self._run_loop_job = None

    def _schedule_next_tick(self):
        self._cancel_run_loop()
        self._run_loop_job = self.root.after(self.run_loop_interval_ms, self._run_loop_tick)

    def _run_loop_tick(self):
        self._run_loop_job = None
        if self.run_state != self.STATE_RUNNING or not self.sim:
            return

        self.sim.update(self.base_dt)
        self.update_environment_plot()
        self.update_agent_table()
        self.update_event_monitor()
        self.update_dashboard()
        self._sync_construction_summaries()
        self._update_backend_status_display()
        self._update_observability_status_display()
        self.update_interaction_tab()
        self.update_brain_tab()
        self.update_dik_derivations_tab()
        self.update_rule_derivations_tab()
        self._schedule_next_tick()

    def _update_control_states(self):
        start_enabled = self.run_state in {self.STATE_IDLE, self.STATE_PAUSED, self.STATE_STOPPED}
        pause_enabled = self.run_state == self.STATE_RUNNING
        stop_enabled = self.run_state in {self.STATE_RUNNING, self.STATE_PAUSED}
        batch_enabled = self.run_state in {self.STATE_IDLE, self.STATE_STOPPED}
        self.start_button.config(state="normal" if start_enabled else "disabled")
        self.run_batch_button.config(state="normal" if batch_enabled else "disabled")
        self.pause_button.config(state="normal" if pause_enabled else "disabled")
        self.stop_button.config(state="normal" if stop_enabled else "disabled")
        self.lifecycle_label.config(text=f"State: {self.run_state}")
        self._update_observability_status_display()

    def start_experiment(self):
        if self.run_state in {self.STATE_RUNNING, self.STATE_STARTING}:
            return

        if self.run_state in {self.STATE_IDLE, self.STATE_STOPPED} or self.sim is None:
            self._begin_async_startup()
            return

        self.run_state = self.STATE_RUNNING
        self._update_control_states()
        self._schedule_next_tick()

    def pause_experiment(self):
        if self.run_state != self.STATE_RUNNING:
            return
        self.run_state = self.STATE_PAUSED
        self._cancel_run_loop()
        self._update_control_states()

    def _batch_progress_callback(self, event):
        if self._batch_queue is not None:
            self._batch_queue.put(dict(event))

    def _batch_worker_main(self, settings):
        try:
            run_batch_experiment(
                settings=settings,
                progress_callback=self._batch_progress_callback,
                run_name_builder=self._build_batch_run_experiment_name,
            )
            if self._batch_queue is not None:
                self._batch_queue.put({"type": "batch_complete"})
        except Exception as exc:  # noqa: BLE001
            if self._batch_queue is not None:
                self._batch_queue.put(
                    {
                        "type": "batch_error",
                        "error_message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

    def _schedule_batch_poll(self):
        if self._batch_queue is not None:
            self._batch_poll_job = self.root.after(75, self._poll_batch_queue)

    def _poll_batch_queue(self):
        self._batch_poll_job = None
        if self._batch_queue is None:
            return
        while True:
            try:
                event = self._batch_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_batch_event(event)
            if self._batch_queue is None:
                return
        self._schedule_batch_poll()

    def _handle_batch_event(self, event):
        event_type = event.get("type")
        if event_type == "batch_error":
            if event.get("traceback"):
                print(event["traceback"])
            self.batch_status_var.set(f"Batch failed: {event.get('error_message')}")
            self._batch_queue = None
            self._batch_worker = None
            self.run_state = self.STATE_IDLE
            self._update_control_states()
            messagebox.showerror("Batch Run Failed", event.get("error_message") or "Batch execution failed.")
            return
        if event_type == "batch_complete":
            self.batch_status_var.set("Batch run complete.")
            self._batch_queue = None
            self._batch_worker = None
            self.run_state = self.STATE_IDLE
            self._update_control_states()
            return

        run_idx = max(0, int(event.get("run_index", 0)))
        num_runs = max(1, int(event.get("num_runs", 1)))
        step_idx = max(0, int(event.get("timestep", 0)))
        total_steps = max(1, int(event.get("timesteps_per_run", 1)))
        if event_type == "batch_timestep":
            self.batch_status_var.set(
                f"Batch running: run {run_idx}/{num_runs}, timestep {step_idx}/{total_steps}"
            )
        elif event_type == "batch_run_start":
            self.batch_status_var.set(f"Starting run {run_idx}/{num_runs}")
        elif event_type == "batch_run_complete":
            self.batch_status_var.set(f"Completed run {run_idx}/{num_runs}")

    def run_batch_experiment(self):
        if self.run_state not in {self.STATE_IDLE, self.STATE_STOPPED}:
            return
        settings = self._collect_experiment_settings()
        self.batch_status_var.set(
            f"Batch queued: {settings['num_runs']} run(s), {settings['timesteps_per_run']} timesteps each."
        )
        self.run_state = self.STATE_STARTING
        self._cancel_run_loop()
        self._update_control_states()
        self._batch_queue = queue.Queue()
        self._batch_worker = threading.Thread(target=self._batch_worker_main, args=(settings,), daemon=True)
        self._batch_worker.start()
        self._schedule_batch_poll()

    def _render_environment_plot(self, ax, canvas):
        ax.clear()
        (x_min, x_max), (y_min, y_max) = self.sim.environment.get_viewport_bounds()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title("Mars Colony Environment")

        for obj in self.sim.environment.objects.values():
            if obj["type"] == "circle":
                x, y = obj["position"]
                r = obj["radius"]
                ax.add_patch(Circle((x, y), r, edgecolor='black', facecolor='lightblue'))
                ax.text(x, y, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "rect":
                x, y = obj["position"]
                w, h = obj["size"]
                ax.add_patch(Rectangle((x, y), w, h, edgecolor='black', facecolor='lightgray'))
                ax.text(x + w / 2, y + h / 2, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "line":
                sx, sy = obj["start"]
                ex, ey = obj["end"]
                ax.plot([sx, ex], [sy, ey], color='gray', linewidth=4)
            elif obj["type"] == "blocked":
                (x1, y1), (x2, y2) = obj["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='black', facecolor='darkgray'))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, obj.get("label", ""), ha='center', va='center', fontsize=9)

        for project in self.sim.environment.construction.get_visual_data():
            cx, cy = project["position"]
            r = project["radius"]
            border = project["border_color"]
            fill = project["fill_color"]
            fill_pct = project["progress"]

            ax.add_patch(Circle((cx, cy), r, edgecolor=border, facecolor=fill, linewidth=2))
            ax.add_patch(Circle((cx, cy), r * fill_pct, color=border, alpha=0.3))
            ax.text(cx, cy, project["label"], ha='center', va='center', fontsize=7)

        for agent in self.sim.agents:
            agent.draw(ax)

        canvas.draw()

    def _populate_event_monitor_widgets(self, activity_widget, interaction_widget, zone_widget):
        activity_widget.delete("1.0", tk.END)
        for agent in self.sim.agents:
            activity_widget.insert(tk.END, f"--- {agent.name} ({agent.role}) ---\n")
            activity_widget.insert(tk.END, f"Goal: {agent.goal}\n")
            activity_widget.insert(tk.END, f"Target: {agent.target}\n")
            last_action = getattr(agent, "status_last_action", "") or (agent.activity_log[-1] if agent.activity_log else "")
            if last_action:
                activity_widget.insert(tk.END, f"Last Action: {last_action}\n")
            if agent.mental_model["data"]:
                activity_widget.insert(tk.END, f"Data: {[d.id for d in agent.mental_model['data']]}\n")
            if agent.mental_model["information"]:
                activity_widget.insert(tk.END, f"Info: {[i.id for i in agent.mental_model['information']]}\n")
            if agent.mental_model["knowledge"].rules:
                activity_widget.insert(tk.END, f"Rules: {[r for r in agent.mental_model['knowledge'].rules]}\n")
            activity_widget.insert(tk.END, "\n")

        interaction_widget.delete("1.0", tk.END)
        for agent in self.sim.agents:
            current_states = []
            if agent.target:
                current_states.append("Moving")
            if agent.goal == "share" and not agent.has_shared:
                current_states.append("Communicating")
            if agent.goal == "get_team_info":
                current_states.append("Accessing Info")
            if agent.goal == "build":
                current_states.append("Building")

            interaction_widget.insert(tk.END, f"{agent.name} ({agent.role}):\n")
            interaction_widget.insert(tk.END, f"  Current State(s): {', '.join(current_states) or 'Idle'}\n")
            interaction_widget.insert(tk.END, "  Transitions:\n")
            interaction_widget.insert(tk.END, "    - Idle → Accessing Info\n")
            interaction_widget.insert(tk.END, "    - Accessing Info → Sharing\n")
            interaction_widget.insert(tk.END, "    - Sharing → Building\n")
            interaction_widget.insert(tk.END, "    - Any → Moving (if target exists)\n\n")

        zone_widget.delete("1.0", tk.END)
        for agent in self.sim.agents:
            current_zone = "Transition"
            for zone_name, obj in self.sim.environment.zones.items():
                if "corners" not in obj:
                    continue
                if self.sim.environment._point_in_zone(agent.position, obj["corners"]):
                    current_zone = zone_name
                    break
            zone_widget.insert(tk.END, f"{agent.name} ({agent.role}): {current_zone}\n")

    def _sync_construction_summaries(self):
        self.construction_text.delete("1.0", tk.END)
        self.dashboard_construction_text.delete("1.0", tk.END)
        construction = self.sim.environment.construction
        scene = construction.get_construction_scene_data()
        started_projects = [p for p in construction.projects.values() if p.get("started")]
        if not started_projects:
            line = "No structures started yet.\n"
            self.construction_text.insert(tk.END, line)
            self.dashboard_construction_text.insert(tk.END, line)
        for project in started_projects:
            req = project.get("required_resources", {}).get("bricks", 0)
            delivered = project.get("delivered_resources", {}).get("bricks", 0)
            status = project.get("status", "unknown")
            builders = sorted(project.get("builders", []))
            line = f"{project.get('id', 'unknown')}: status={status}, bricks={delivered}/{req}, builders={', '.join(builders) or 'none'}\n"
            self.construction_text.insert(tk.END, line)
            self.dashboard_construction_text.insert(tk.END, line)
        bridge = construction.bridges.get("bridge_bc")
        bridge_line = f"bridge_bc: status={bridge.status}, resources={bridge.delivered_resources}/{bridge.required_resources}\n"
        self.construction_text.insert(tk.END, bridge_line)
        self.dashboard_construction_text.insert(tk.END, bridge_line)
        for pile in scene.get("resource_piles", []):
            pile_line = f"{pile['pile_id']}: remaining={pile['remaining']}/{pile['max_quantity']}\n"
            self.construction_text.insert(tk.END, pile_line)
            self.dashboard_construction_text.insert(tk.END, pile_line)

        self._render_construction_tab(scene)

    @staticmethod
    def _map_structure_visual(structure):
        structure_type = str(structure.get("structure_type") or structure.get("type") or "").lower()
        if structure_type in {"house", "housing"}:
            return {"symbol": "square", "color": "#c6362f"}
        if structure_type == "greenhouse":
            return {"symbol": "square", "color": "#2f8f46"}
        if structure_type == "water_generator":
            return {"symbol": "square", "color": "#2f6fbf"}
        return {"symbol": "square", "color": str(structure.get("color") or "#666666")}

    @staticmethod
    def _progress_fill_fraction(structure):
        raw = structure.get("progress", 0.0)
        try:
            return max(0.0, min(1.0, float(raw or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _project_overlay_state(structure):
        if not structure.get("correct", True) or structure.get("status") == "needs_repair":
            return "invalid"
        if structure.get("validated_complete"):
            return "validated"
        if structure.get("resource_complete") and not structure.get("validated_complete"):
            return "ready_for_validation"
        return "in_progress"

    @staticmethod
    def _site_circle_style():
        return {
            "radius": 0.86,
            "edgecolor": "#b8c1cd",
            "facecolor": "none",
            "linewidth": 1.2,
            "linestyle": "solid",
        }

    @staticmethod
    def _site_label_text(site):
        return str(site.get("label") or site.get("site_id") or "site")

    @staticmethod
    def _structure_label_text(_structure):
        return ""

    @staticmethod
    def _draw_progress_square(ax, x, y, size, progress, color, zorder=3):
        left, bottom = x - size / 2.0, y - size / 2.0
        outline = Rectangle((left, bottom), size, size, edgecolor=color, facecolor="none", linewidth=1.7, zorder=zorder)
        ax.add_patch(outline)
        if progress > 0:
            ax.add_patch(
                Rectangle(
                    (left, bottom),
                    size,
                    size * progress,
                    edgecolor="none",
                    facecolor=color,
                    zorder=zorder - 0.2,
                )
            )

    @staticmethod
    def _draw_structure_symbol(ax, structure, x, y, size=0.56):
        visual = MarsColonyInterface._map_structure_visual(structure)
        color = visual["color"]
        progress = MarsColonyInterface._progress_fill_fraction(structure)
        MarsColonyInterface._draw_progress_square(ax, x, y, size, progress, color)

    @staticmethod
    def _structure_draw_size(site_radius):
        max_size_from_area = math.sqrt(MarsColonyInterface._STRUCTURE_AREA_RATIO_CAP * math.pi * site_radius * site_radius)
        return min(0.28, max_size_from_area)

    @staticmethod
    def _structure_anchor_offsets(site_radius, draw_size):
        half_diag = draw_size * math.sqrt(2.0) / 2.0
        anchor_radius = max(0.0, site_radius - half_diag - 0.02) * MarsColonyInterface._STRUCTURE_ANCHOR_RADIAL_FACTOR
        anchor_angles_deg = [90.0, 30.0, -30.0, -90.0, -150.0, 150.0]
        offsets = []
        for angle_deg in anchor_angles_deg:
            radians = math.radians(angle_deg)
            offsets.append((anchor_radius * math.cos(radians), anchor_radius * math.sin(radians)))
        return offsets

    @staticmethod
    def _resource_pile_center(site_center, pile_id, site_radius):
        sx, sy = site_center
        corner_by_pile = {
            "pile_a": (1.0, -1.0),   # lower-right of site_a
            "pile_c": (-1.0, -1.0),  # lower-left of site_c
        }
        dx, dy = corner_by_pile.get(str(pile_id or ""), (0.0, 0.0))
        mag = math.hypot(dx, dy)
        if mag == 0.0:
            return sx, sy
        ux, uy = dx / mag, dy / mag
        center_offset = (
            site_radius
            + (MarsColonyInterface._PILE_DRAW_SIZE * math.sqrt(2.0) / 2.0)
            + MarsColonyInterface._PILE_OUTSIDE_PADDING
        )
        return sx + ux * center_offset, sy + uy * center_offset

    @staticmethod
    def _trimmed_bridge_endpoints(start_site_center, end_site_center, site_radius):
        sx, sy = start_site_center
        ex, ey = end_site_center
        vx, vy = ex - sx, ey - sy
        dist = math.hypot(vx, vy)
        if dist <= 0.0:
            return (sx, sy), (ex, ey)
        ux, uy = vx / dist, vy / dist
        trim = max(0.0, site_radius - MarsColonyInterface._BRIDGE_SITE_INSET)
        return (sx + ux * trim, sy + uy * trim), (ex - ux * trim, ey - uy * trim)

    @staticmethod
    def _square_corners(center_x, center_y, size):
        half = size / 2.0
        return [
            (center_x - half, center_y - half),
            (center_x - half, center_y + half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
        ]

    @staticmethod
    def _nearest_corner_pair(bounds_a, bounds_b):
        ax, ay, asize = bounds_a
        bx, by, bsize = bounds_b
        corners_a = MarsColonyInterface._square_corners(ax, ay, asize)
        corners_b = MarsColonyInterface._square_corners(bx, by, bsize)
        closest = min(
            ((ca, cb) for ca in corners_a for cb in corners_b),
            key=lambda pair: math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]),
        )
        return closest

    @staticmethod
    def _draw_construction_scene(ax, scene_data):
        ax.clear()
        ax.set_xlim(2.0, 8.0)
        ax.set_ylim(2.0, 6.0)
        ax.set_aspect("equal")
        ax.set_title("Construction Visual State")
        ax.grid(True, alpha=0.15)

        structures = scene_data.get("structures", [])
        sites = scene_data.get("sites", [])
        resource_piles = scene_data.get("resource_piles", [])
        bridges = scene_data.get("bridges", [])
        site_lookup = {
            site.get("site_id"): {
                "site_id": site.get("site_id"),
                "position": tuple(site.get("position", (0.0, 0.0))),
                "label": site.get("label"),
            }
            for site in sites
            if site.get("site_id")
        }

        grouped_by_site = {site_id: [] for site_id in site_lookup}
        for structure in structures:
            site_id = structure.get("site_id")
            if site_id in grouped_by_site:
                grouped_by_site[site_id].append(structure)

        # 1) Empty site circles
        for site_id, site in site_lookup.items():
            sx, sy = site["position"]
            site_style = MarsColonyInterface._site_circle_style()
            site_circle = Circle(
                (sx, sy),
                site_style["radius"],
                edgecolor=site_style["edgecolor"],
                facecolor=site_style["facecolor"],
                linewidth=site_style["linewidth"],
                linestyle=site_style["linestyle"],
                zorder=0,
            )
            ax.add_patch(site_circle)

            site_name = MarsColonyInterface._site_label_text(site)
            ax.text(
                sx,
                sy - (site_style["radius"] + 0.10),
                site_name,
                ha="center",
                va="top",
                fontsize=6.0,
                color="#8c95a3",
            )

        site_radius = MarsColonyInterface._site_circle_style()["radius"]

        # 2) Resource piles as black squares with proportional fill
        for pile in resource_piles:
            x, y = tuple(pile.get("position", (0.0, 0.0)))
            site_id = pile.get("site_id")
            if site_id in site_lookup:
                x, y = MarsColonyInterface._resource_pile_center(
                    site_lookup[site_id]["position"],
                    pile.get("pile_id"),
                    site_radius,
                )
            fill = max(0.0, min(1.0, float(pile.get("fill_fraction", 0.0) or 0.0)))
            MarsColonyInterface._draw_progress_square(
                ax,
                x,
                y,
                size=MarsColonyInterface._PILE_DRAW_SIZE,
                progress=fill,
                color="black",
                zorder=2,
            )

        # 3) Bridges
        for bridge in bridges:
            start = bridge.get("start")
            end = bridge.get("end")
            if not start or not end:
                continue
            start_site = site_lookup.get(bridge.get("start_site_id"))
            end_site = site_lookup.get(bridge.get("end_site_id"))
            if start_site and end_site:
                start, end = MarsColonyInterface._trimmed_bridge_endpoints(
                    start_site["position"],
                    end_site["position"],
                    site_radius,
                )
            status = str(bridge.get("status") or "complete")
            width = 2.0 if status == "complete" else 1.3
            alpha = 1.0 if status == "complete" else 0.6
            ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=width, alpha=alpha, zorder=1.5)

        # 4) Structure squares inside site circles (only started structures provided in scene data)
        structure_bounds_by_project = {}
        draw_size = MarsColonyInterface._structure_draw_size(site_radius)
        for site_id, site_structures in grouped_by_site.items():
            if not site_structures:
                continue
            sx, sy = site_lookup[site_id]["position"]
            offsets = MarsColonyInterface._structure_anchor_offsets(site_radius, draw_size)
            ordered_structures = sorted(site_structures, key=lambda row: str(row.get("project_id") or row.get("name") or ""))
            for idx, structure in enumerate(ordered_structures):
                ox, oy = offsets[idx % len(offsets)]
                px, py = sx + ox, sy + oy
                MarsColonyInterface._draw_structure_symbol(ax, structure, px, py, size=draw_size)
                project_id = structure.get("project_id")
                if project_id:
                    structure_bounds_by_project[project_id] = (px, py, draw_size)

        # 5) Connectors
        for connector in scene_data.get("connectors", []):
            start = connector.get("start")
            end = connector.get("end")
            start_project_id = connector.get("start_project_id")
            end_project_id = connector.get("end_project_id")
            if start_project_id in structure_bounds_by_project and end_project_id in structure_bounds_by_project:
                start, end = MarsColonyInterface._nearest_corner_pair(
                    structure_bounds_by_project[start_project_id],
                    structure_bounds_by_project[end_project_id],
                )
            if not start or not end:
                continue
            ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=1.5, zorder=1)

    def _render_construction_tab(self, scene_data):
        self._draw_construction_scene(self.construction_ax, scene_data)
        self.construction_canvas.draw()

    def update_dashboard(self):
        self._render_environment_plot(self.dashboard_ax, self.dashboard_canvas)
        self._populate_event_monitor_widgets(
            self.dashboard_agent_activity_text,
            self.dashboard_interaction_state_text,
            self.dashboard_zone_state_text,
        )

        for i in self.dashboard_agent_state.get_children():
            self.dashboard_agent_state.delete(i)
        for agent in self.sim.agents:
            self.dashboard_agent_state.insert(
                "",
                "end",
                values=(agent.name, agent.goal or "None", agent.heart_rate, round(agent.co2_output, 3)),
            )

        self._update_system_log()

    def _format_system_event(self, event):
        try:
            import json
            payload = json.loads(event.get("payload", "{}"))
        except Exception:
            payload = {}

        event_type = event.get("event_type", "event")
        sim_time = event.get("time", 0.0)

        if event_type == "session_initialized":
            return f"t={sim_time:05.2f} Session created: {payload.get('session_folder', 'unknown')}"
        if event_type == "outputs_saved":
            return f"t={sim_time:05.2f} Outputs saved ({payload.get('rows', 0)} state rows, {payload.get('event_rows', 0)} events)."
        if event_type == "brain_backend_selected":
            return f"t={sim_time:05.2f} backend configured={payload.get('configured_brain_backend')} effective={payload.get('effective_brain_backend')} provider={payload.get('provider_class')} model={payload.get('local_model_name')} url={payload.get('local_base_url')} endpoint={payload.get('local_endpoint')} timeout={payload.get('timeout_s')}."
        if event_type == "brain_backend_runtime_status":
            return f"t={sim_time:05.2f} backend runtime configured={payload.get('configured_brain_backend')} effective={payload.get('effective_brain_backend')} fallback={payload.get('fallback_backend')} model={payload.get('local_model_name')} timeout={payload.get('timeout_s')}."
        if event_type == "brain_decision_query":
            meta = payload.get("context_meta", {})
            return f"t={sim_time:05.2f} {payload.get('agent')} query via {payload.get('provider')} cfg={payload.get('configured_brain_backend')} eff={payload.get('effective_brain_backend')} ({payload.get('trigger_reason')}); affordances={meta.get('affordance_count', 0)}."
        if event_type == "brain_decision_outcome":
            return f"t={sim_time:05.2f} {payload.get('agent')} decision {payload.get('decision_status')} -> {payload.get('selected_action')} ({payload.get('provider')}, cfg={payload.get('configured_brain_backend')}, eff={payload.get('effective_brain_backend')})."
        if event_type == "brain_provider_fallback":
            return f"t={sim_time:05.2f} WARNING fallback cfg={payload.get('configured_brain_backend')} eff={payload.get('effective_brain_backend')} {payload.get('provider')} -> {payload.get('fallback_provider')}: {payload.get('reason')} model={payload.get('local_model_name')} hint={payload.get('fallback_hint')}"
        if event_type == "effective_brain_backend_updated":
            return f"t={sim_time:05.2f} backend effective updated to {payload.get('effective_brain_backend')} (configured={payload.get('configured_brain_backend')}, reason={payload.get('reason')})."
        if event_type == "brain_plan_continued":
            return f"t={sim_time:05.2f} {payload.get('agent')} continuing {payload.get('plan_id')} (remaining={payload.get('remaining_executions')})."
        return f"t={sim_time:05.2f} {event_type}: {payload}"

    def _update_system_log(self):
        if not self.sim or not hasattr(self.sim, "logger"):
            return

        events = self.sim.logger.get_recent_events(60)
        self.system_log_text.delete("1.0", tk.END)
        for event in events:
            self.system_log_text.insert(tk.END, self._format_system_event(event) + "\n")

        if self.system_log_text.tag_ranges("sel"):
            return
        self.system_log_text.see(tk.END)

    def update_environment_plot(self, frame=None):
        if not self.sim:
            return  # Don't try to update before simulation starts
        self._render_environment_plot(self.ax, self.canvas)

    def update_agent_table(self):
        for i in self.agent_state_table.get_children():
            self.agent_state_table.delete(i)
        for agent in self.sim.agents:
            self.agent_state_table.insert("", "end", values=(agent.heart_rate, round(agent.gsr, 3), round(agent.temperature, 2), round(agent.co2_output, 3)))

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        self.stop_experiment()
        self.root.quit()
        self.root.destroy()

    def update_event_monitor(self):
        self._populate_event_monitor_widgets(
            self.agent_activity_text,
            self.interaction_state_text,
            self.zone_state_text,
        )

    def _create_experiment_global_settings(self, parent):
        settings_panel = tk.LabelFrame(
            parent,
            text="Global Experiment Settings",
            padx=10,
            pady=8,
            relief="groove",
            borderwidth=2,
            highlightbackground=self.EXPERIMENT_PANEL_BORDER,
            highlightthickness=1,
        )
        settings_panel.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        settings_frame = ttk.Frame(settings_panel)
        settings_frame.grid(row=0, column=0, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Speed Multiplier").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        tk.Scale(
            settings_frame,
            variable=self.speed_multiplier,
            from_=0.5,
            to=10.0,
            resolution=0.1,
            orient="horizontal",
            length=260,
        ).grid(row=0, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 1, "Controls how quickly simulated time advances.")

        ttk.Label(settings_frame, text="Flash Mode").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Checkbutton(settings_frame, text="Enable (no animation)", variable=self.flash_mode).grid(row=2, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 3, "Runs the simulation without rendering each animation step.")

        ttk.Label(settings_frame, text="Number of Agents").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=3)
        self.num_agents_var = IntVar(value=self.DEFAULT_AGENT_COUNT)
        num_agents_combo = ttk.Combobox(settings_frame, textvariable=self.num_agents_var, values=[1, 2, 3, 4, 5, 6], state="readonly", width=10)
        num_agents_combo.grid(row=4, column=1, sticky="w", pady=3)
        num_agents_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visible_agent_cards())
        self._add_help_text(settings_frame, 5, "Select how many agent panels are active for this run (1-6).")

        ttk.Label(settings_frame, text="Experiment Name").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=3)
        self.experiment_name_var = StringVar()
        ttk.Entry(settings_frame, textvariable=self.experiment_name_var, width=34).grid(row=6, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 7, "Optional label used in output folders and logs.")

        ttk.Label(settings_frame, text="Number of Simulation Runs").grid(row=8, column=0, sticky="w", padx=(0, 8), pady=(6, 3))
        self.num_runs = IntVar(value=1)
        ttk.Entry(settings_frame, textvariable=self.num_runs, width=10).grid(row=8, column=1, sticky="w", pady=(6, 3))
        self._add_help_text(settings_frame, 9, "How many repeated runs to execute with the same setup.")

        ttk.Label(settings_frame, text="Timesteps per Run").grid(row=10, column=0, sticky="w", padx=(0, 8), pady=(6, 3))
        self.timesteps_per_run_var = IntVar(value=300)
        ttk.Entry(settings_frame, textvariable=self.timesteps_per_run_var, width=10).grid(row=10, column=1, sticky="w", pady=(6, 3))
        self._add_help_text(settings_frame, 11, "Exact number of simulation updates per run in batch mode.")

        ttk.Label(settings_frame, text="Brain Backend").grid(row=12, column=0, sticky="w", padx=(0, 8), pady=(6, 3))
        self.brain_backend_var = StringVar(value=self.BACKEND_DEFAULTS["brain_backend"])
        self.brain_backend_var.trace_add("write", lambda *_: self._refresh_all_agent_inheritance_display())
        backend_combo = ttk.Combobox(settings_frame, textvariable=self.brain_backend_var, values=["rule_brain", "local_http", "ollama"], state="readonly", width=22)
        backend_combo.grid(row=12, column=1, sticky="w", pady=(6, 3))
        backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_backend_field_states())
        self._add_help_text(settings_frame, 13, "Select which decision system agents use by default.")

        self.local_model_var = StringVar(value=self.BACKEND_DEFAULTS["local_model"])
        self.local_model_var.trace_add("write", lambda *_: self._refresh_all_agent_inheritance_display())
        self.local_base_url_var = StringVar(value=self.BACKEND_DEFAULTS["local_base_url"])
        self.local_timeout_var = DoubleVar(value=self.BACKEND_DEFAULTS["timeout_s"])
        self.fallback_backend_var = StringVar(value=self.BACKEND_DEFAULTS["fallback_backend"])
        self.fallback_backend_var.trace_add("write", lambda *_: self._refresh_all_agent_inheritance_display())

        ttk.Label(settings_frame, text="Local Model").grid(row=14, column=0, sticky="w", padx=(0, 8), pady=3)
        local_model_entry = ttk.Combobox(settings_frame, textvariable=self.local_model_var, values=self.LOCAL_MODEL_SHORTLIST, width=31)
        local_model_entry.grid(row=14, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 15, "Local model name to use for Ollama-backed agents. Recommended default on weaker machines: qwen2.5:3b-instruct.")

        ttk.Label(settings_frame, text="Local Base URL").grid(row=16, column=0, sticky="w", padx=(0, 8), pady=3)
        local_base_url_entry = ttk.Entry(settings_frame, textvariable=self.local_base_url_var, width=34)
        local_base_url_entry.grid(row=16, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 17, "Base URL for the local backend endpoint.")

        ttk.Label(settings_frame, text="Local Timeout (s)").grid(row=18, column=0, sticky="w", padx=(0, 8), pady=3)
        local_timeout_entry = ttk.Entry(settings_frame, textvariable=self.local_timeout_var, width=10)
        local_timeout_entry.grid(row=18, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 19, "Maximum request time for the selected default backend.")

        ttk.Label(settings_frame, text="Fallback Backend").grid(row=20, column=0, sticky="w", padx=(0, 8), pady=3)
        fallback_combo = ttk.Combobox(settings_frame, textvariable=self.fallback_backend_var, values=["rule_brain"], state="readonly", width=22)
        fallback_combo.grid(row=20, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 21, "Used if the selected backend fails or times out.")

        self.enable_startup_llm_sanity_var = BooleanVar(value=bool(self.PLANNER_DEFAULTS["enable_startup_llm_sanity"]))
        self.startup_llm_sanity_timeout_var = DoubleVar(value=float(self.PLANNER_DEFAULTS["startup_llm_sanity_timeout_seconds"]))
        self.startup_llm_sanity_max_sources_var = IntVar(value=int(self.PLANNER_DEFAULTS["startup_llm_sanity_max_sources"]))
        self.startup_llm_sanity_max_items_var = IntVar(value=int(self.PLANNER_DEFAULTS["startup_llm_sanity_max_items_per_type"]))
        self.startup_llm_sanity_raw_max_chars_var = IntVar(value=int(self.PLANNER_DEFAULTS["startup_llm_sanity_raw_response_max_chars"]))
        self.bootstrap_reuse_enabled_var = BooleanVar(value=bool(self.PLANNER_DEFAULTS["enable_bootstrap_summary_reuse"]))
        self.bootstrap_summary_max_chars_var = IntVar(value=int(self.PLANNER_DEFAULTS["bootstrap_summary_max_chars"]))
        self.planner_timeout_seconds_var = DoubleVar(value=float(self.PLANNER_DEFAULTS["planner_timeout_seconds"]))
        self.planner_request_policy_var = StringVar(value=str(self.PLANNER_DEFAULTS["planner_request_policy"]))
        self.planning_interval_steps_var = IntVar(value=int(self.PLANNER_DEFAULTS["planning_interval_steps"]))
        self.dik_integration_cooldown_steps_var = IntVar(value=int(self.PLANNER_DEFAULTS["dik_integration_cooldown_steps"]))
        self.dik_integration_batch_threshold_var = IntVar(value=int(self.PLANNER_DEFAULTS["dik_integration_batch_threshold"]))
        self.high_latency_local_llm_mode_var = BooleanVar(value=bool(self.PLANNER_DEFAULTS["high_latency_local_llm_mode"]))
        self.unrestricted_local_qwen_mode_var = BooleanVar(value=bool(self.PLANNER_DEFAULTS["unrestricted_local_qwen_mode"]))
        self.warmup_timeout_var = DoubleVar(value=float(self.PLANNER_DEFAULTS["warmup_timeout_seconds"]))
        self.startup_llm_sanity_completion_tokens_var = IntVar(value=int(self.PLANNER_DEFAULTS["startup_llm_sanity_completion_max_tokens"]))
        self.planner_completion_tokens_var = IntVar(value=int(self.PLANNER_DEFAULTS["planner_completion_max_tokens"]))
        self.high_latency_stale_result_grace_var = DoubleVar(value=float(self.PLANNER_DEFAULTS["high_latency_stale_result_grace_s"]))

        ttk.Label(settings_frame, text="Startup LLM Sanity / Bootstrap").grid(row=22, column=0, sticky="w", padx=(0, 8), pady=(8, 3))
        ttk.Checkbutton(settings_frame, text="Enable Startup LLM Sanity / Bootstrap", variable=self.enable_startup_llm_sanity_var).grid(row=22, column=1, sticky="w", pady=(8, 3))
        self._add_help_text(settings_frame, 23, "Runs once per agent at startup with bounded role/task/DIK context. It validates local model responsiveness and seeds explicit simulator-side bootstrap context for reuse.")

        ttk.Label(settings_frame, text="Startup Sanity Timeout (s)").grid(row=24, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.startup_llm_sanity_timeout_var, width=10).grid(row=24, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 25, "Per-agent timeout for startup sanity/bootstrap requests.")

        ttk.Label(settings_frame, text="Max Sources in Startup Prompt").grid(row=26, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.startup_llm_sanity_max_sources_var, width=10).grid(row=26, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 27, "Bound the number of role-relevant sources included in startup context.")

        ttk.Label(settings_frame, text="Max Items Per Type in Startup Prompt").grid(row=28, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.startup_llm_sanity_max_items_var, width=10).grid(row=28, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 29, "Bound data/information/knowledge/rule examples included per type.")

        ttk.Label(settings_frame, text="Startup Completion Budget (tokens)").grid(row=30, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.startup_llm_sanity_completion_tokens_var, width=10).grid(row=30, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 31, "Maximum tokens allowed for startup sanity completions.")

        ttk.Label(settings_frame, text="Planner Completion Budget (tokens)").grid(row=32, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.planner_completion_tokens_var, width=10).grid(row=32, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 33, "Maximum tokens allowed for planner completions.")

        ttk.Label(settings_frame, text="Planner Timeout (s)").grid(row=34, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.planner_timeout_seconds_var, width=10).grid(row=34, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 35, "Per-request planner timeout budget for local LLM calls.")

        ttk.Label(settings_frame, text="Planner Request Policy").grid(row=36, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(settings_frame, textvariable=self.planner_request_policy_var, values=["cadence_with_dik_integration", "legacy"], state="readonly", width=30).grid(row=36, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 37, "Split mode runs planning on cadence and DIK integration on epistemic triggers. Legacy preserves prior mixed cadence logic.")

        ttk.Label(settings_frame, text="Split Planning Interval (steps)").grid(row=38, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.planning_interval_steps_var, width=10).grid(row=38, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 39, "When split mode is enabled, submit planner requests every N simulation ticks.")

        ttk.Label(settings_frame, text="DIK Cooldown (steps)").grid(row=40, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.dik_integration_cooldown_steps_var, width=10).grid(row=40, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 41, "Minimum ticks between DIK integration requests while in split mode.")

        ttk.Label(settings_frame, text="DIK Batch Threshold").grid(row=42, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.dik_integration_batch_threshold_var, width=10).grid(row=42, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 43, "Minimum epistemic delta required before split-mode DIK integration is submitted.")

        ttk.Label(settings_frame, text="Warmup Timeout (s)").grid(row=44, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.warmup_timeout_var, width=10).grid(row=44, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 45, "Startup backend warmup timeout budget.")

        ttk.Label(settings_frame, text="Raw Response Max Chars").grid(row=46, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.startup_llm_sanity_raw_max_chars_var, width=10).grid(row=46, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 47, "Truncate captured raw startup responses to keep artifacts bounded.")

        ttk.Label(settings_frame, text="Reuse Bootstrap Summary in Planner Requests").grid(row=48, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Checkbutton(settings_frame, text="Include compact bootstrap summary on future planner requests", variable=self.bootstrap_reuse_enabled_var).grid(row=48, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 49, "Agents remain persistent simulator entities for the session, but model calls stay explicit stateless requests. Reuse adds a compact inspectable summary field; no hidden model-side memory is assumed.")

        ttk.Label(settings_frame, text="Bootstrap Summary Max Chars").grid(row=50, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.bootstrap_summary_max_chars_var, width=10).grid(row=50, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 51, "Upper bound for compact bootstrap summaries attached to planner requests.")

        ttk.Label(settings_frame, text="High-Latency Local LLM Mode").grid(row=52, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Checkbutton(settings_frame, text="Let local LLM requests run with relaxed timing and stale-result tolerance", variable=self.high_latency_local_llm_mode_var).grid(row=52, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 53, "Diagnostic mode for slow local inference: relaxed timeouts, reduced planner pressure, and less eager stale-result discards.")

        ttk.Label(settings_frame, text="Unrestricted Local Qwen Mode").grid(row=54, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Checkbutton(settings_frame, text="Very permissive mode: multi-minute waits + very large completion budgets (with safety ceilings)", variable=self.unrestricted_local_qwen_mode_var).grid(row=54, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 55, "Intentionally patient diagnostics mode for slow local Qwen runs.")

        ttk.Label(settings_frame, text="Stale Result Grace (s)").grid(row=56, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(settings_frame, textvariable=self.high_latency_stale_result_grace_var, width=10).grid(row=56, column=1, sticky="w", pady=3)
        self._add_help_text(settings_frame, 57, "Additional grace window to accept late but still relevant planner responses in high-latency mode.")

        self.construction_param_vars = {
            "pile_a_quantity": IntVar(value=int(self.construction_defaults["pile_a_quantity"])),
            "pile_c_quantity": IntVar(value=int(self.construction_defaults["pile_c_quantity"])),
            "housing_cost": IntVar(value=int(self.construction_defaults["housing_cost"])),
            "greenhouse_cost": IntVar(value=int(self.construction_defaults["greenhouse_cost"])),
            "water_generator_cost": IntVar(value=int(self.construction_defaults["water_generator_cost"])),
            "bridge_bc_cost": IntVar(value=int(self.construction_defaults["bridge_bc_cost"])),
            "site_a_capacity": IntVar(value=int(self.construction_defaults["site_a_capacity"])),
            "site_b_capacity": IntVar(value=int(self.construction_defaults["site_b_capacity"])),
            "site_c_capacity": IntVar(value=int(self.construction_defaults["site_c_capacity"])),
            "move_time_per_unit": IntVar(value=int(self.construction_defaults["move_time_per_unit"])),
            "carry_capacity": IntVar(value=int(self.construction_defaults["carry_capacity"])),
        }
        row = 58
        ttk.Label(settings_frame, text="Construction Parameters").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(8, 3))
        self._add_help_text(settings_frame, row + 1, "Site/resource/bridge settings loaded from task defaults and overridable before run start.")
        row += 2
        construction_fields = [
            ("Pile A Quantity", "pile_a_quantity"),
            ("Pile C Quantity", "pile_c_quantity"),
            ("Housing Cost", "housing_cost"),
            ("Greenhouse Cost", "greenhouse_cost"),
            ("Water Generator Cost", "water_generator_cost"),
            ("Bridge BC Cost", "bridge_bc_cost"),
            ("Site A Capacity", "site_a_capacity"),
            ("Site B Capacity", "site_b_capacity"),
            ("Site C Capacity", "site_c_capacity"),
            ("Move Time Per Unit", "move_time_per_unit"),
            ("Carry Capacity Per Trip", "carry_capacity"),
        ]
        for label, key in construction_fields:
            ttk.Label(settings_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(settings_frame, textvariable=self.construction_param_vars[key], width=10).grid(row=row, column=1, sticky="w", pady=3)
            row += 1

        self._local_backend_widgets = [local_model_entry, local_base_url_entry, local_timeout_entry, fallback_combo]
        self._update_backend_field_states()

    def _create_trait_slider(self, parent, row, col, label, variable):
        item = ttk.Frame(parent)
        item.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
        item.columnconfigure(0, weight=1)

        ttk.Label(item, text=label).grid(row=0, column=0, sticky="w")
        tk.Scale(
            item,
            variable=variable,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            length=180,
        ).grid(row=1, column=0, sticky="w")

    def _create_agent_card(self, parent, agent_row, row, trait_labels, update_traits_from_profile):
        role = agent_row["role"]
        self.agent_identity[role] = {
            "display_name": StringVar(value=agent_row.get("display_name") or role),
            "alias": StringVar(value=agent_row.get("label") or role),
            "template_id": agent_row.get("template_id"),
        }

        card = tk.LabelFrame(
            parent,
            text=f"Agent {row + 1} Configuration",
            padx=10,
            pady=8,
            relief="groove",
            borderwidth=2,
            highlightbackground=self.EXPERIMENT_PANEL_BORDER,
            highlightthickness=1,
        )
        card.grid(row=row, column=0, sticky="ew", padx=10, pady=6)
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        self.active_roles[role] = BooleanVar(value=True)
        header = ttk.Frame(card)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        header.columnconfigure(1, weight=1)

        ttk.Checkbutton(header, text="Enable", variable=self.active_roles[role]).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.agent_identity[role]["display_name"]).grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(header, text="Display Name").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(header, textvariable=self.agent_identity[role]["display_name"], width=20).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(4, 0))
        ttk.Label(header, text="Alias").grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(header, textvariable=self.agent_identity[role]["alias"], width=20).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(2, 0))

        ttk.Label(card, text="Teamwork Potential").grid(row=1, column=0, sticky="w")
        ttk.Label(card, text="Taskwork Potential").grid(row=1, column=1, sticky="w")

        team_potential = StringVar(value="High")
        task_potential = StringVar(value="High")
        self.agent_profiles[role] = {"team": team_potential, "task": task_potential}

        ttk.OptionMenu(card, team_potential, "High", "High", "Low", command=lambda _, r=role: update_traits_from_profile(r)).grid(row=2, column=0, sticky="w", pady=(2, 6))
        ttk.OptionMenu(card, task_potential, "High", "High", "Low", command=lambda _, r=role: update_traits_from_profile(r)).grid(row=2, column=1, sticky="w", pady=(2, 6))

        traits_frame = ttk.LabelFrame(card, text="Traits", padding=(8, 6))
        traits_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        traits_frame.columnconfigure(0, weight=1)
        traits_frame.columnconfigure(1, weight=1)

        left_traits = ["communication_propensity", "help_tendency", "rule_accuracy"]
        right_traits = ["goal_alignment", "build_speed"]

        self.agent_traits[role] = {}

        for idx, trait in enumerate(left_traits):
            self.agent_traits[role][trait] = DoubleVar(value=0.5)
            self._create_trait_slider(traits_frame, row=idx, col=0, label=trait_labels[trait], variable=self.agent_traits[role][trait])

        for idx, trait in enumerate(right_traits):
            self.agent_traits[role][trait] = DoubleVar(value=0.5)
            self._create_trait_slider(traits_frame, row=idx, col=1, label=trait_labels[trait], variable=self.agent_traits[role][trait])

        packet_frame = ttk.LabelFrame(card, text="Packet Access", padding=(8, 6))
        packet_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        packet_frame.columnconfigure(0, weight=1)

        packet_names = ["Team_Packet", "Architect_Packet", "Engineer_Packet", "Botanist_Packet"]
        default_packet_access = self._resolve_default_packet_access(agent_row.get("default_packet_access"))
        self.packet_access[role] = {}
        for idx, pkt in enumerate(packet_names):
            pkt_enabled = BooleanVar(value=pkt in default_packet_access)
            self.packet_access[role][pkt] = pkt_enabled
            ttk.Checkbutton(packet_frame, text=pkt, variable=pkt_enabled).grid(row=idx, column=0, sticky="w", pady=1)

        settings_frame = ttk.LabelFrame(card, text="Per-Agent Brain/Planner", padding=(8, 6))
        settings_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        settings_frame.columnconfigure(1, weight=1)

        default_brain = dict(agent_row.get("brain_config", {}))
        default_planner = dict(agent_row.get("planner_config", {}))
        self.agent_brain_settings[role] = {
            "backend": StringVar(value=default_brain.get("backend", "")),
            "local_model": StringVar(value=default_brain.get("local_model", "")),
            "fallback_backend": StringVar(value=default_brain.get("fallback_backend", "")),
            "timeout_s": DoubleVar(value=float(default_brain.get("timeout_s", self.PLANNER_DEFAULTS["backend_timeout_s"]))),
            "max_retries": IntVar(value=int(default_brain.get("max_retries", self.PLANNER_DEFAULTS["backend_max_retries"]))),
        }
        self.agent_planner_settings[role] = {
            "planner_interval_steps": IntVar(value=int(default_planner.get("planner_interval_steps", self.PLANNER_DEFAULTS["planner_interval_steps"]))),
            "planner_timeout_seconds": DoubleVar(value=float(default_planner.get("planner_timeout_seconds", self.PLANNER_DEFAULTS["planner_timeout_seconds"]))),
            "planner_max_retries": IntVar(value=int(default_planner.get("planner_max_retries", self.PLANNER_DEFAULTS["planner_max_retries"]))),
            "planner_request_policy": StringVar(value=str(default_planner.get("planner_request_policy", self.PLANNER_DEFAULTS["planner_request_policy"]))),
            "planning_interval_steps": IntVar(value=int(default_planner.get("planning_interval_steps", self.PLANNER_DEFAULTS["planning_interval_steps"]))),
            "dik_integration_cooldown_steps": IntVar(value=int(default_planner.get("dik_integration_cooldown_steps", self.PLANNER_DEFAULTS["dik_integration_cooldown_steps"]))),
            "dik_integration_batch_threshold": IntVar(value=int(default_planner.get("dik_integration_batch_threshold", self.PLANNER_DEFAULTS["dik_integration_batch_threshold"]))),
            "degraded_consecutive_failures_threshold": IntVar(value=int(default_planner.get("degraded_consecutive_failures_threshold", self.PLANNER_DEFAULTS["degraded_consecutive_failures_threshold"]))),
            "degraded_cooldown_seconds": DoubleVar(value=float(default_planner.get("degraded_cooldown_seconds", self.PLANNER_DEFAULTS["degraded_cooldown_seconds"]))),
            "degraded_step_interval_multiplier": DoubleVar(value=float(default_planner.get("degraded_step_interval_multiplier", self.PLANNER_DEFAULTS["degraded_step_interval_multiplier"]))),
        }

        self.agent_inheritance_note_vars[role] = {
            "backend": StringVar(value=""),
            "local_model": StringVar(value=""),
            "fallback_backend": StringVar(value=""),
        }
        self.agent_effective_summary_vars[role] = {
            "backend": StringVar(value=""),
            "local_model": StringVar(value=""),
            "fallback_backend": StringVar(value=""),
        }

        fields = [
            ("Backend Override", ttk.Combobox(settings_frame, textvariable=self.agent_brain_settings[role]["backend"], values=["", "rule_brain", "local_http", "ollama"], state="readonly", width=18), "Optional per-agent backend override. Leave blank to inherit global default.", "backend"),
            ("Model Override", ttk.Entry(settings_frame, textvariable=self.agent_brain_settings[role]["local_model"], width=20), "Optional model override for this agent. Leave blank to inherit global model.", "local_model"),
            ("Fallback Override", ttk.Combobox(settings_frame, textvariable=self.agent_brain_settings[role]["fallback_backend"], values=["", "rule_brain"], state="readonly", width=18), "Fallback used by this agent if its selected backend fails. Leave blank to inherit global fallback.", "fallback_backend"),
            ("Planner Cadence (steps)", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["planner_interval_steps"], width=8), "Higher values reduce how often this agent's brain is queried.", None),
            ("Planner Timeout (s)", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["planner_timeout_seconds"], width=8), "Maximum time allowed for this agent's planning step before it is treated as failed.", None),
            ("Backend Timeout (s)", ttk.Entry(settings_frame, textvariable=self.agent_brain_settings[role]["timeout_s"], width=8), "Maximum backend request time for this agent.", None),
            ("Backend Max Retries", ttk.Entry(settings_frame, textvariable=self.agent_brain_settings[role]["max_retries"], width=8), self.RETRY_HELP_TEXT["backend_max_retries"], None),
            ("Planner Max Retries", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["planner_max_retries"], width=8), self.RETRY_HELP_TEXT["planner_max_retries"], None),
            ("Planner Request Policy", ttk.Combobox(settings_frame, textvariable=self.agent_planner_settings[role]["planner_request_policy"], values=["cadence_with_dik_integration", "legacy"], state="readonly", width=28), "Optional per-agent policy override for split DIK/planning cadence.", None),
            ("Split Planning Interval", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["planning_interval_steps"], width=8), "Per-agent split mode planner interval in ticks.", None),
            ("DIK Cooldown (steps)", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["dik_integration_cooldown_steps"], width=8), "Per-agent minimum spacing between DIK integration requests.", None),
            ("DIK Batch Threshold", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["dik_integration_batch_threshold"], width=8), "Per-agent minimum epistemic delta before DIK integration is requested.", None),
            ("Degraded Threshold", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["degraded_consecutive_failures_threshold"], width=8), "Number of consecutive backend failures before degraded mode begins.", None),
            ("Degraded Cooldown (s)", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["degraded_cooldown_seconds"], width=8), "How long this agent waits before retrying the backend after repeated failures.", None),
            ("Degraded Step Multiplier", ttk.Entry(settings_frame, textvariable=self.agent_planner_settings[role]["degraded_step_interval_multiplier"], width=8), "In degraded mode, increases the interval between planning attempts.", None),
        ]

        current_row = 0
        for label, widget, help_text, inheritance_key in fields:
            ttk.Label(settings_frame, text=label).grid(row=current_row, column=0, sticky="w", pady=(2, 0))
            widget.grid(row=current_row, column=1, sticky="w", pady=(2, 0))
            current_row += 1
            self._add_help_text(settings_frame, current_row, help_text)
            current_row += 1
            if inheritance_key:
                ttk.Label(
                    settings_frame,
                    textvariable=self.agent_inheritance_note_vars[role][inheritance_key],
                    foreground="#3f556e",
                    wraplength=620,
                    justify="left",
                ).grid(row=current_row, column=0, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 4))
                current_row += 1

        effective_frame = ttk.Frame(settings_frame)
        effective_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(effective_frame, text="Effective Settings", foreground="#1f3247").grid(row=0, column=0, sticky="w")
        ttk.Label(effective_frame, textvariable=self.agent_effective_summary_vars[role]["backend"], foreground="#3f556e").grid(row=1, column=0, sticky="w")
        ttk.Label(effective_frame, textvariable=self.agent_effective_summary_vars[role]["local_model"], foreground="#3f556e").grid(row=2, column=0, sticky="w")
        ttk.Label(effective_frame, textvariable=self.agent_effective_summary_vars[role]["fallback_backend"], foreground="#3f556e").grid(row=3, column=0, sticky="w")

        for key in ("backend", "local_model", "fallback_backend"):
            self.agent_brain_settings[role][key].trace_add("write", lambda *_args, r=role: self._update_agent_inheritance_display(r))
        self._update_agent_inheritance_display(role)
        return card

    def create_experiment_tab(self):
        container = ttk.Frame(self.notebook)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.tab_experiment = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=self.tab_experiment, anchor="nw")

        def _scroll_experiment_canvas(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        def _bind_mousewheel(_):
            canvas.bind_all("<MouseWheel>", _scroll_experiment_canvas)
            canvas.bind_all("<Button-4>", _scroll_experiment_canvas)
            canvas.bind_all("<Button-5>", _scroll_experiment_canvas)

        def _unbind_mousewheel(_):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        def _update_experiment_scrollregion(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _update_experiment_width(event):
            content_width = min(event.width, self.EXPERIMENT_MAX_CONTENT_WIDTH)
            side_gutter = max((event.width - content_width) // 2, 0)
            canvas.coords(canvas_window, side_gutter, 0)
            canvas.itemconfigure(canvas_window, width=content_width)

        self.tab_experiment.bind("<Configure>", _update_experiment_scrollregion)
        canvas.bind("<Configure>", _update_experiment_width)
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.notebook.add(container, text="Experiment")
        self.tab_experiment.columnconfigure(0, weight=1)

        self.active_roles = {}
        self.agent_profiles = {}
        self.agent_traits = {}
        self.packet_access = {}
        self.agent_identity = {}
        self.agent_brain_settings = {}
        self.agent_planner_settings = {}
        self.agent_inheritance_note_vars = {}
        self.agent_effective_summary_vars = {}
        self.agent_cards = {}
        self.agent_card_order = []

        task_model = load_task_model(task_id="mars_colony")
        default_rows = []
        for d in task_model.agent_defaults:
            default_rows.append(
                {
                    "role": d.role_id,
                    "display_name": d.display_name or d.agent_name,
                    "label": d.agent_label or d.role_id,
                    "template_id": d.template_id,
                    "brain_config": dict(d.brain_config or {}),
                    "planner_config": dict(d.planner_config or {}),
                    "default_packet_access": list(d.source_access_override or []),
                }
            )
        by_role = {row["role"]: row for row in default_rows}
        default_rows = []
        for fallback in self.DEFAULT_AGENT_IDENTITIES:
            role = fallback["role"]
            row = dict(by_role.get(role, {}))
            row.setdefault("role", role)
            row.setdefault("display_name", fallback["display_name"])
            row.setdefault("label", fallback["label"])
            row.setdefault("template_id", fallback["template_id"])
            row.setdefault("brain_config", {})
            row.setdefault("planner_config", {})
            row.setdefault("default_packet_access", [])
            default_rows.append(row)

        # Trait labels
        trait_labels = {
            "communication_propensity": "Tendency to Communicate",
            "goal_alignment": "Agreement with Team Goals",
            "help_tendency": "Willingness to Help",
            "build_speed": "Speed of Building",
            "rule_accuracy": "Rule Interpretation Accuracy"
        }

        # Preset values
        profile_traits = {
            "High_Team": {"communication_propensity": 1.0, "goal_alignment": 1.0, "help_tendency": 1.0},
            "Low_Team": {"communication_propensity": 0.3, "goal_alignment": 0.4, "help_tendency": 0.2},
            "High_Task": {"build_speed": 1.0, "rule_accuracy": 1.0},
            "Low_Task": {"build_speed": 0.5, "rule_accuracy": 0.5}
        }

        def update_traits_from_profile(role):
            team = self.agent_profiles[role]["team"].get()
            task = self.agent_profiles[role]["task"].get()

            for trait, value in profile_traits[f"{team}_Team"].items():
                self.agent_traits[role][trait].set(value)
            for trait, value in profile_traits[f"{task}_Task"].items():
                self.agent_traits[role][trait].set(value)

        self._create_experiment_global_settings(self.tab_experiment)

        cards_container = ttk.Frame(self.tab_experiment)
        cards_container.grid(row=1, column=0, sticky="ew", padx=10)
        cards_container.columnconfigure(0, weight=1)

        for row, agent_row in enumerate(default_rows):
            role = agent_row["role"]
            self.agent_card_order.append(role)
            self.agent_cards[role] = self._create_agent_card(cards_container, agent_row, row, trait_labels, update_traits_from_profile)

        self._refresh_all_agent_inheritance_display()
        self._update_visible_agent_cards()
        self.batch_status_var = StringVar(value="Batch idle.")
        ttk.Label(
            self.tab_experiment,
            textvariable=self.batch_status_var,
            foreground="#35506b",
            wraplength=900,
            justify="left",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(4, 2))

        ttk.Label(
            self.tab_experiment,
            text="Use the shared Start / Pause / Stop controls at the top to run the simulation.",
        ).grid(row=3, column=0, sticky="w", padx=10, pady=(2, 10))



    def stop_experiment(self):
        if self.run_state == self.STATE_STARTING:
            return
        self._cancel_run_loop()
        if self.sim:
            self.sim.stop()
        if self.sim is not None:
            self.run_state = self.STATE_STOPPED
        else:
            self.run_state = self.STATE_IDLE
        self._update_control_states()
        print("Simulation stopped and data saved.")


if __name__ == "__main__":
    app = MarsColonyInterface()
    app.run()
