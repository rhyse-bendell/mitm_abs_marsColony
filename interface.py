# File: interface.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules.simulation import SimulationState
from tkinter import StringVar, BooleanVar, DoubleVar, IntVar
from modules.construction import ConstructionManager
from modules.phase_definitions import MISSION_PHASES


class MarsColonyInterface:
    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_STOPPED = "stopped"
    EXPERIMENT_MAX_CONTENT_WIDTH = 940
    EXPERIMENT_PANEL_BORDER = "#586271"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mars Colony Simulation")

        self.speed_multiplier = DoubleVar(value=1.0)
        self.flash_mode = BooleanVar(value=False)

        self.sim = None
        self.construction = ConstructionManager()
        self.run_state = self.STATE_IDLE
        self._run_loop_job = None
        self.run_loop_interval_ms = 100
        self.base_dt = 0.1

        self.create_widgets()
        self._update_control_states()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.control_frame = ttk.Frame(self.root, padding=(8, 6))
        self.control_frame.pack(fill="x")
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_experiment)
        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_experiment)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_experiment)
        self.lifecycle_label = ttk.Label(self.control_frame, text="State: idle")

        self.start_button.pack(side="left", padx=(0, 6))
        self.pause_button.pack(side="left", padx=6)
        self.stop_button.pack(side="left", padx=6)
        self.lifecycle_label.pack(side="right")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.create_experiment_tab()
        self.create_dashboard_tab()
        self.create_main_tab()
        self.create_construction_tab()
        self.create_agents_tab()
        self.create_event_monitor_tab()

    def _build_environment_canvas(self, parent):
        fig, ax = plt.subplots(figsize=(6, 6))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, ax, canvas

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

        self.construction_text = tk.Text(self.tab_construction, wrap="word")
        self.construction_text.pack(fill="both", expand=True)

    def create_agents_tab(self):
        self.tab_agents = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_agents, text="Agent States")

        self.agent_state_table = ttk.Treeview(self.tab_agents, columns=("Heart Rate", "GSR", "Temp", "CO2"), show="headings")
        for col in self.agent_state_table["columns"]:
            self.agent_state_table.heading(col, text=col)
        self.agent_state_table.pack(fill="both", expand=True)

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
                "communication_propensity": 0.9,
                "goal_alignment": 0.9,
                "help_tendency": 0.8
            },
            "Low_Team": {
                "communication_propensity": 0.3,
                "goal_alignment": 0.4,
                "help_tendency": 0.2
            },
            "High_Task": {
                "build_speed": 1.0,
                "rule_accuracy": 0.95
            },
            "Low_Task": {
                "build_speed": 0.5,
                "rule_accuracy": 0.5
            }
        }

        for role in self.active_roles:
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

                agent_configs.append({
                    "name": role,
                    "role": role,
                    "constructs": {
                        "teamwork_potential": profile_constructs[team_pot],
                        "taskwork_potential": profile_constructs[task_pot],
                    },
                    "mechanism_overrides": dict(traits),
                    "traits": traits,
                    "packet_access": selected_packets
                })

        return agent_configs

    def _collect_brain_backend_config(self):
        backend = (self.brain_backend_var.get() or "rule_brain").strip() or "rule_brain"
        options = {
            "local_model": self.local_model_var.get().strip() or "local-model",
            "local_base_url": self.local_base_url_var.get().strip() or "http://127.0.0.1:11434",
            "timeout_s": max(0.1, float(self.local_timeout_var.get())),
            "fallback_backend": (self.fallback_backend_var.get() or "rule_brain").strip() or "rule_brain",
        }
        return backend, options

    def _is_local_backend_selected(self):
        selected = (self.brain_backend_var.get() or "").strip().lower()
        return selected in {"local_http", "openai_compatible_local", "ollama_local", "ollama"}

    def _update_backend_field_states(self):
        state = "normal" if self._is_local_backend_selected() else "disabled"
        for widget in getattr(self, "_local_backend_widgets", []):
            widget.configure(state=state)

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

    def apply_experiment_settings(self):
        print("=== Experiment Settings ===")
        print("Speed Multiplier:", self.speed_multiplier.get())
        print("Flash Mode:", self.flash_mode.get())

        print("Number of Runs:", self.num_runs.get())
        selected_backend, backend_options = self._collect_brain_backend_config()
        print("Brain Backend:", selected_backend)
        print("Brain Backend Options:", backend_options)

        agent_configs = self.build_agent_configs()
        for agent in agent_configs:
            print(f"\n{agent['name']} ({agent['role']})")
            for k, v in agent["traits"].items():
                print(f"  {k}: {v}")
            print(f"  Packet Access: {agent['packet_access']}")

        # Create new simulation with selected parameters
        from modules.simulation import SimulationState
        self.sim = SimulationState(
            agent_configs=agent_configs,
            num_runs=self.num_runs.get(),
            speed=self.speed_multiplier.get(),
            experiment_name=self.experiment_name_var.get(),
            phases=MISSION_PHASES,
            flash_mode=self.flash_mode.get(),
            brain_backend=selected_backend,
            brain_backend_options=backend_options,
        )

        self.run_state = self.STATE_IDLE
        self._cancel_run_loop()
        self._update_control_states()

        self.update_environment_plot()
        self._sync_construction_summaries()
        self._update_system_log()
        self._update_backend_status_display()

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
        self._schedule_next_tick()

    def _update_control_states(self):
        start_enabled = self.run_state in {self.STATE_IDLE, self.STATE_PAUSED, self.STATE_STOPPED}
        pause_enabled = self.run_state == self.STATE_RUNNING
        stop_enabled = self.run_state in {self.STATE_RUNNING, self.STATE_PAUSED}
        self.start_button.config(state="normal" if start_enabled else "disabled")
        self.pause_button.config(state="normal" if pause_enabled else "disabled")
        self.stop_button.config(state="normal" if stop_enabled else "disabled")
        self.lifecycle_label.config(text=f"State: {self.run_state}")

    def start_experiment(self):
        if self.run_state == self.STATE_RUNNING:
            return

        if self.run_state in {self.STATE_IDLE, self.STATE_STOPPED} or self.sim is None:
            self.apply_experiment_settings()

        self.run_state = self.STATE_RUNNING
        self._update_control_states()
        self._schedule_next_tick()

    def pause_experiment(self):
        if self.run_state != self.STATE_RUNNING:
            return
        self.run_state = self.STATE_PAUSED
        self._cancel_run_loop()
        self._update_control_states()

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
                ax.add_patch(plt.Circle((x, y), r, edgecolor='black', facecolor='lightblue'))
                ax.text(x, y, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "rect":
                x, y = obj["position"]
                w, h = obj["size"]
                ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='black', facecolor='lightgray'))
                ax.text(x + w / 2, y + h / 2, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "line":
                sx, sy = obj["start"]
                ex, ey = obj["end"]
                ax.plot([sx, ex], [sy, ey], color='gray', linewidth=4)
            elif obj["type"] == "blocked":
                (x1, y1), (x2, y2) = obj["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='black', facecolor='darkgray'))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, obj.get("label", ""), ha='center', va='center', fontsize=9)

        for project in self.sim.environment.construction.get_visual_data():
            cx, cy = project["position"]
            r = project["radius"]
            border = project["border_color"]
            fill = project["fill_color"]
            fill_pct = project["progress"]

            ax.add_patch(plt.Circle((cx, cy), r, edgecolor=border, facecolor=fill, linewidth=2))
            ax.add_patch(plt.Circle((cx, cy), r * fill_pct, color=border, alpha=0.3))
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
        projects = list(self.sim.environment.construction.projects.values())
        if not projects:
            msg = "No active construction projects."
            self.construction_text.insert(tk.END, msg)
            self.dashboard_construction_text.insert(tk.END, msg)
            return

        for project in projects:
            req = project.get("required_resources", {}).get("bricks", 0)
            delivered = project.get("delivered_resources", {}).get("bricks", 0)
            status = project.get("status", "unknown")
            builders = sorted(project.get("builders", []))
            line = f"{project.get('id', 'unknown')}: status={status}, bricks={delivered}/{req}, builders={', '.join(builders) or 'none'}\n"
            self.construction_text.insert(tk.END, line)
            self.dashboard_construction_text.insert(tk.END, line)

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
            return f"t={sim_time:05.2f} backend configured={payload.get('configured_brain_backend')} effective={payload.get('effective_brain_backend')} provider={payload.get('provider_class')}."
        if event_type == "brain_backend_runtime_status":
            return f"t={sim_time:05.2f} backend runtime configured={payload.get('configured_brain_backend')} effective={payload.get('effective_brain_backend')} fallback={payload.get('fallback_backend')}."
        if event_type == "brain_decision_query":
            meta = payload.get("context_meta", {})
            return f"t={sim_time:05.2f} {payload.get('agent')} query via {payload.get('provider')} cfg={payload.get('configured_brain_backend')} eff={payload.get('effective_brain_backend')} ({payload.get('trigger_reason')}); affordances={meta.get('affordance_count', 0)}."
        if event_type == "brain_decision_outcome":
            return f"t={sim_time:05.2f} {payload.get('agent')} decision {payload.get('decision_status')} -> {payload.get('selected_action')} ({payload.get('provider')}, cfg={payload.get('configured_brain_backend')}, eff={payload.get('effective_brain_backend')})."
        if event_type == "brain_provider_fallback":
            return f"t={sim_time:05.2f} WARNING fallback cfg={payload.get('configured_brain_backend')} eff={payload.get('effective_brain_backend')} {payload.get('provider')} -> {payload.get('fallback_provider')}: {payload.get('reason')}"
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

        ttk.Label(settings_frame, text="Flash Mode").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Checkbutton(settings_frame, text="Enable (no animation)", variable=self.flash_mode).grid(
            row=1,
            column=1,
            sticky="w",
            pady=3,
        )

        ttk.Label(settings_frame, text="Experiment Name").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.experiment_name_var = StringVar()
        ttk.Entry(settings_frame, textvariable=self.experiment_name_var, width=34).grid(
            row=2,
            column=1,
            sticky="w",
            pady=3,
        )

        ttk.Label(settings_frame, text="Number of Simulation Runs").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 3))
        self.num_runs = IntVar(value=1)
        ttk.Entry(settings_frame, textvariable=self.num_runs, width=10).grid(row=3, column=1, sticky="w", pady=(6, 3))

        ttk.Label(settings_frame, text="Brain Backend").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=(6, 3))
        self.brain_backend_var = StringVar(value="rule_brain")
        backend_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.brain_backend_var,
            values=["rule_brain", "local_http", "ollama"],
            state="readonly",
            width=22,
        )
        backend_combo.grid(row=4, column=1, sticky="w", pady=(6, 3))
        backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_backend_field_states())

        self.local_model_var = StringVar(value="local-model")
        self.local_base_url_var = StringVar(value="http://127.0.0.1:11434")
        self.local_timeout_var = DoubleVar(value=1.5)
        self.fallback_backend_var = StringVar(value="rule_brain")

        ttk.Label(settings_frame, text="Local Model").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=3)
        local_model_entry = ttk.Entry(settings_frame, textvariable=self.local_model_var, width=34)
        local_model_entry.grid(row=5, column=1, sticky="w", pady=3)

        ttk.Label(settings_frame, text="Local Base URL").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=3)
        local_base_url_entry = ttk.Entry(settings_frame, textvariable=self.local_base_url_var, width=34)
        local_base_url_entry.grid(row=6, column=1, sticky="w", pady=3)

        ttk.Label(settings_frame, text="Local Timeout (s)").grid(row=7, column=0, sticky="w", padx=(0, 8), pady=3)
        local_timeout_entry = ttk.Entry(settings_frame, textvariable=self.local_timeout_var, width=10)
        local_timeout_entry.grid(row=7, column=1, sticky="w", pady=3)

        ttk.Label(settings_frame, text="Fallback Backend").grid(row=8, column=0, sticky="w", padx=(0, 8), pady=3)
        fallback_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.fallback_backend_var,
            values=["rule_brain"],
            state="readonly",
            width=22,
        )
        fallback_combo.grid(row=8, column=1, sticky="w", pady=3)

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

    def _create_agent_card(self, parent, role, row, trait_labels, update_traits_from_profile):
        card = tk.LabelFrame(
            parent,
            text=f"{role} Configuration",
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
        ttk.Label(header, text=role).grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(card, text="Teamwork Potential").grid(row=1, column=0, sticky="w")
        ttk.Label(card, text="Taskwork Potential").grid(row=1, column=1, sticky="w")

        team_potential = StringVar(value="High")
        task_potential = StringVar(value="High")
        self.agent_profiles[role] = {"team": team_potential, "task": task_potential}

        ttk.OptionMenu(
            card,
            team_potential,
            "High",
            "High",
            "Low",
            command=lambda _, r=role: update_traits_from_profile(r),
        ).grid(row=2, column=0, sticky="w", pady=(2, 6))
        ttk.OptionMenu(
            card,
            task_potential,
            "High",
            "High",
            "Low",
            command=lambda _, r=role: update_traits_from_profile(r),
        ).grid(row=2, column=1, sticky="w", pady=(2, 6))

        traits_frame = ttk.LabelFrame(card, text="Traits", padding=(8, 6))
        traits_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        traits_frame.columnconfigure(0, weight=1)
        traits_frame.columnconfigure(1, weight=1)

        left_traits = ["communication_propensity", "help_tendency", "rule_accuracy"]
        right_traits = ["goal_alignment", "build_speed"]

        self.agent_traits[role] = {}

        for idx, trait in enumerate(left_traits):
            self.agent_traits[role][trait] = DoubleVar(value=0.5)
            self._create_trait_slider(
                traits_frame,
                row=idx,
                col=0,
                label=trait_labels[trait],
                variable=self.agent_traits[role][trait],
            )

        for idx, trait in enumerate(right_traits):
            self.agent_traits[role][trait] = DoubleVar(value=0.5)
            self._create_trait_slider(
                traits_frame,
                row=idx,
                col=1,
                label=trait_labels[trait],
                variable=self.agent_traits[role][trait],
            )

        packet_frame = ttk.LabelFrame(card, text="Packet Access", padding=(8, 6))
        packet_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        packet_frame.columnconfigure(0, weight=1)

        packet_names = ["Team_Packet", "Architect_Packet", "Engineer_Packet", "Botanist_Packet"]
        self.packet_access[role] = {}
        for idx, pkt in enumerate(packet_names):
            pkt_enabled = BooleanVar(value=(idx == 0))
            self.packet_access[role][pkt] = pkt_enabled
            ttk.Checkbutton(packet_frame, text=pkt, variable=pkt_enabled).grid(
                row=idx,
                column=0,
                sticky="w",
                pady=1,
            )

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
        roles = ["Architect", "Engineer", "Botanist"]

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
            "High_Team": {"communication_propensity": 0.9, "goal_alignment": 0.9, "help_tendency": 0.8},
            "Low_Team": {"communication_propensity": 0.3, "goal_alignment": 0.4, "help_tendency": 0.2},
            "High_Task": {"build_speed": 1.0, "rule_accuracy": 0.95},
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

        for row, role in enumerate(roles):
            self._create_agent_card(cards_container, role, row, trait_labels, update_traits_from_profile)

        ttk.Label(
            self.tab_experiment,
            text="Use the shared Start / Pause / Stop controls at the top to run the simulation.",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(4, 10))



    def stop_experiment(self):
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
