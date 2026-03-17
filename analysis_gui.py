from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.analysis_loader import load_analysis_session
from modules.analysis_models import AnalysisSession
from modules.analysis_plots import PLOT_OPTIONS, build_plot
from modules.analysis_stats import aggregate_statistics, phase_statistics
from modules.analysis_widgets import fill_text_widget, populate_key_value_tree
from modules.replay_engine import ReplayEngine
from modules.interaction_graph import CANONICAL_NODES


class AnalysisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mars Colony Session Analysis")

        self.session: AnalysisSession | None = None
        self.replay_engine: ReplayEngine | None = None
        self.replay_index = 0
        self.replay_job = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        ttk.Button(top, text="Open Session Folder", command=self.open_session_folder).pack(side="left")
        self.session_label = ttk.Label(top, text="No session loaded")
        self.session_label.pack(side="left", padx=8)

        self.warn_text = tk.Text(self.root, height=4, wrap="word")
        self.warn_text.pack(fill="x", padx=6, pady=(0, 6))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self._build_replay_tab()
        self._build_interaction_tab()
        self._build_graphs_tab()
        self._build_aggregate_tab()
        self._build_phase_tab()

    def _build_replay_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Replay")

        controls = ttk.Frame(tab, padding=6)
        controls.pack(fill="x")
        ttk.Button(controls, text="<<", command=self.step_back).pack(side="left")
        ttk.Button(controls, text=">", command=self.step_forward).pack(side="left", padx=4)
        ttk.Button(controls, text="Play", command=self.play).pack(side="left")
        ttk.Button(controls, text="Pause", command=self.pause).pack(side="left", padx=4)
        self.time_label = ttk.Label(controls, text="t=0.00")
        self.time_label.pack(side="left", padx=8)

        self.replay_slider = ttk.Scale(tab, from_=0, to=1, orient="horizontal", command=self.on_slider)
        self.replay_slider.pack(fill="x", padx=8)

        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        import matplotlib.pyplot as plt

        self.replay_fig, self.replay_ax = plt.subplots(figsize=(6, 6))
        self.replay_canvas = FigureCanvasTkAgg(self.replay_fig, master=left)
        self.replay_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.agent_tree = ttk.Treeview(right, columns=("agent", "goal", "x", "y"), show="headings", height=8)
        for c in ("agent", "goal", "x", "y"):
            self.agent_tree.heading(c, text=c)
        self.agent_tree.pack(fill="x")

        self.events_text = tk.Text(right, height=12, wrap="word")
        self.events_text.pack(fill="both", expand=True, pady=(6, 0))

        self.jump_list = tk.Listbox(right, height=8)
        self.jump_list.pack(fill="x", pady=(6, 0))
        self.jump_list.bind("<<ListboxSelect>>", self.on_jump_event)


    def _build_interaction_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Interaction Replay")

        controls = ttk.Frame(tab, padding=6)
        controls.pack(fill="x")
        self.interaction_status = ttk.Label(controls, text="No interaction trace loaded")
        self.interaction_status.pack(side="left")

        self.interaction_slider = ttk.Scale(tab, from_=0, to=1, orient="horizontal", command=self.on_interaction_slider)
        self.interaction_slider.pack(fill="x", padx=8)

        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        self.interaction_canvas = tk.Canvas(left, bg="#10141b", height=520)
        self.interaction_canvas.pack(fill="both", expand=True)

        self.interaction_detail = tk.Text(right, wrap="word")
        self.interaction_detail.pack(fill="both", expand=True)

    def _build_graphs_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Graphs")

        top = ttk.Frame(tab, padding=6)
        top.pack(fill="x")
        self.plot_var = tk.StringVar(value=list(PLOT_OPTIONS.keys())[0])
        ttk.Combobox(top, textvariable=self.plot_var, values=list(PLOT_OPTIONS.keys()), state="readonly").pack(side="left")
        ttk.Button(top, text="Render", command=self.render_plot).pack(side="left", padx=6)

        self.graph_host = ttk.Frame(tab)
        self.graph_host.pack(fill="both", expand=True)
        self.graph_canvas = None

    def _build_aggregate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Aggregate Statistics")

        self.agg_tree = ttk.Treeview(tab, columns=("key", "value"), show="headings")
        self.agg_tree.heading("key", text="Metric")
        self.agg_tree.heading("value", text="Value")
        self.agg_tree.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_phase_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Phase Statistics")

        split = ttk.PanedWindow(tab, orient="horizontal")
        split.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=1)
        split.add(right, weight=2)

        self.phase_list = tk.Listbox(left)
        self.phase_list.pack(fill="both", expand=True)
        self.phase_list.bind("<<ListboxSelect>>", self.on_phase_select)

        self.phase_detail_text = tk.Text(right, wrap="word")
        self.phase_detail_text.pack(fill="both", expand=True)

    def open_session_folder(self):
        selected = filedialog.askdirectory(title="Select session output folder")
        if not selected:
            return
        self.load_session(Path(selected))

    def load_session(self, folder: Path):
        self.session = load_analysis_session(folder)
        self.replay_engine = ReplayEngine(self.session)
        self.replay_index = 0
        self.session_label.config(text=str(folder))

        self.warn_text.delete("1.0", tk.END)
        for msg in self.session.warnings:
            self.warn_text.insert(tk.END, f"- {msg}\n")

        self.refresh_replay()
        self.refresh_interaction_replay()
        self.refresh_aggregate()
        self.refresh_phase()
        self.render_plot()

    def refresh_interaction_replay(self):
        trace = self.session.artifacts.interaction_trace if self.session else []
        if not trace:
            self.interaction_status.config(text="This session has no logs/interaction_trace.jsonl artifact.")
            self.interaction_detail.delete("1.0", tk.END)
            self.interaction_detail.insert(tk.END, "No interaction trace available for replay.")
            self._draw_interaction_state([])
            return
        self.interaction_status.config(text=f"Interaction events: {len(trace)}")
        self.interaction_slider.configure(to=max(0, len(self.replay_engine.frames) - 1 if self.replay_engine else 0))
        self.update_interaction_frame_view()

    def refresh_replay(self):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        self.replay_slider.configure(to=max(0, len(self.replay_engine.frames) - 1))
        self.update_frame_view()
        self.update_interaction_frame_view()

        self.jump_list.delete(0, tk.END)
        for ev in self.replay_engine.important_events():
            self.jump_list.insert(tk.END, f"t={ev.get('time')} {ev.get('event_type')}")

    def update_frame_view(self):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        frame = self.replay_engine.frames[self.replay_index]
        self.replay_slider.set(self.replay_index)
        self.time_label.config(text=f"t={frame.time:.2f} ({frame.index + 1}/{len(self.replay_engine.frames)})")

        self.replay_ax.clear()
        xs, ys = [], []
        for name, row in frame.agent_states.items():
            try:
                x = float(row.get("x", 0.0))
                y = float(row.get("y", 0.0))
            except Exception:
                x, y = 0.0, 0.0
            xs.append(x)
            ys.append(y)
            self.replay_ax.text(x, y, name, fontsize=8)
        if xs:
            self.replay_ax.scatter(xs, ys, c="#1f77b4")
        self.replay_ax.set_title("Environment Replay (from logged state rows)")
        self.replay_ax.set_xlim(0, 100)
        self.replay_ax.set_ylim(0, 100)
        self.replay_ax.grid(alpha=0.2)
        self.replay_canvas.draw()

        for i in self.agent_tree.get_children():
            self.agent_tree.delete(i)
        for name, row in sorted(frame.agent_states.items()):
            self.agent_tree.insert("", "end", values=(name, row.get("goal"), row.get("x"), row.get("y")))

        self.events_text.delete("1.0", tk.END)
        for event in frame.events_at_time[-20:]:
            payload = event.get("payload", "{}")
            self.events_text.insert(tk.END, f"t={event.get('time')} {event.get('event_type')} {payload}\n")

    def step_back(self):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        self.replay_index = max(0, self.replay_index - 1)
        self.update_frame_view()
        self.update_interaction_frame_view()

    def step_forward(self):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        self.replay_index = min(len(self.replay_engine.frames) - 1, self.replay_index + 1)
        self.update_frame_view()
        self.update_interaction_frame_view()

    def on_interaction_slider(self, value):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        self.replay_index = int(float(value))
        self.update_interaction_frame_view()

    def on_slider(self, value):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        self.replay_index = int(float(value))
        self.update_frame_view()
        self.update_interaction_frame_view()

    def update_interaction_frame_view(self):
        if not self.replay_engine or not self.replay_engine.frames:
            return
        frame = self.replay_engine.frames[self.replay_index]
        self.interaction_slider.set(self.replay_index)
        interactions = list(frame.interaction_events_at_time)
        self._draw_interaction_state(interactions)
        self.interaction_detail.delete("1.0", tk.END)
        for row in interactions[-30:]:
            self.interaction_detail.insert(tk.END, f"t={row.get('time')} {row.get('interaction_type')} {row.get('source_node')} -> {row.get('target_node')} [{row.get('status')}]\n{row.get('payload_summary')}\n\n")

    def _draw_interaction_state(self, interactions):
        canvas = self.interaction_canvas
        canvas.delete("all")
        w = max(200, canvas.winfo_width() or 900)
        h = max(200, canvas.winfo_height() or 520)
        active_nodes = set()
        active_edges = set()
        for row in interactions:
            s, t = row.get("source_node"), row.get("target_node")
            if s:
                active_nodes.add(s)
            if t:
                active_nodes.add(t)
            if s and t:
                active_edges.add((s, t))
        for s, t in active_edges:
            sn = next((n for n in CANONICAL_NODES if n.node_id == s), None)
            tn = next((n for n in CANONICAL_NODES if n.node_id == t), None)
            if sn and tn:
                canvas.create_line(sn.pos[0]*w, sn.pos[1]*h, tn.pos[0]*w, tn.pos[1]*h, fill="#3fb0ff", width=2, arrow=tk.LAST)
        for node in CANONICAL_NODES:
            x, y = node.pos[0]*w, node.pos[1]*h
            fill = "#ffd166" if node.node_id in active_nodes else "#303846"
            canvas.create_oval(x-18, y-18, x+18, y+18, fill=fill, outline="#dddddd")
            canvas.create_text(x, y-26, text=node.label, fill="#e5e7eb", font=("Arial", 8))

    def play(self):
        self.pause()

        def _tick():
            if not self.replay_engine or not self.replay_engine.frames:
                return
            if self.replay_index < len(self.replay_engine.frames) - 1:
                self.replay_index += 1
                self.update_frame_view()
                self.update_interaction_frame_view()
                self.replay_job = self.root.after(300, _tick)

        self.replay_job = self.root.after(50, _tick)

    def pause(self):
        if self.replay_job is not None:
            self.root.after_cancel(self.replay_job)
            self.replay_job = None

    def on_jump_event(self, _event):
        if not self.replay_engine:
            return
        selection = self.jump_list.curselection()
        if not selection:
            return
        item = self.jump_list.get(selection[0])
        try:
            marker = item.split()[0].replace("t=", "")
            target_time = float(marker)
        except Exception:
            return
        nearest = min(range(len(self.replay_engine.frames)), key=lambda i: abs(self.replay_engine.frames[i].time - target_time))
        self.replay_index = nearest
        self.update_frame_view()
        self.update_interaction_frame_view()

    def render_plot(self):
        if not self.session:
            return
        fig = build_plot(self.session, self.plot_var.get())
        if self.graph_canvas is not None:
            self.graph_canvas.get_tk_widget().destroy()
        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_host)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.graph_canvas.draw()

    def refresh_aggregate(self):
        if not self.session:
            return
        stats = aggregate_statistics(self.session)
        compact = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                compact[key] = json.dumps(value)[:350]
            elif isinstance(value, list):
                compact[key] = f"{len(value)} rows"
            else:
                compact[key] = value
        populate_key_value_tree(self.agg_tree, compact)

    def refresh_phase(self):
        if not self.session:
            return
        self.phase_rows = phase_statistics(self.session)
        self.phase_list.delete(0, tk.END)
        for row in self.phase_rows:
            name = row.get("phase_name", "unknown")
            dur = row.get("duration_seconds")
            self.phase_list.insert(tk.END, f"{name} (duration={dur})")

    def on_phase_select(self, _event):
        selection = self.phase_list.curselection()
        if not selection:
            return
        idx = selection[0]
        row = self.phase_rows[idx]
        fill_text_widget(self.phase_detail_text, row)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AnalysisGUI().run()
