# File: interface.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules.simulation import SimulationState
from tkinter import StringVar, BooleanVar, DoubleVar, IntVar
from modules.construction import ConstructionManager
from modules.phase_definitions import MISSION_PHASES


class MarsColonyInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mars Colony Simulation")

        self.speed_multiplier = DoubleVar(value=1.0)
        self.flash_mode = BooleanVar(value=False)

        self.sim = None
        self.construction = ConstructionManager()

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.create_experiment_tab()
        self.create_main_tab()
        self.create_construction_tab()
        self.create_agents_tab()
        self.create_event_monitor_tab()

    def create_main_tab(self):
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Environment")

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.anim = None  # Don’t start animation until experiment is started


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

                selected_packets = [self.packet_access[role].get(i) for i in self.packet_access[role].curselection()]

                agent_configs.append({
                    "name": role,
                    "role": role,
                    "traits": traits,
                    "packet_access": selected_packets
                })

        return agent_configs

    def apply_experiment_settings(self):
        print("=== Experiment Settings ===")
        print("Speed Multiplier:", self.speed_multiplier.get())
        print("Flash Mode:", self.flash_mode.get())

        print("Number of Runs:", self.num_runs.get())

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
            flash_mode=self.flash_mode.get()
        )

        # Refresh main tab to reflect new sim state
        # Stop old animation if it exists
        if self.anim:
            self.anim.event_source.stop()

        # Start new animation
        self.anim = animation.FuncAnimation(self.fig, self.update_environment_plot, interval=100)

        self.stop_button.config(state="normal")

    def update_environment_plot(self, frame):
        if not self.sim:
            return  # Don't try to update before simulation starts

        self.ax.clear()
        self.sim.update(0.1)  # This will now be scaled internally

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title("Mars Colony Environment")

        for obj in self.sim.environment.objects.values():
            if obj["type"] == "circle":
                x, y = obj["position"]
                r = obj["radius"]
                self.ax.add_patch(plt.Circle((x, y), r, edgecolor='black', facecolor='lightblue'))
                self.ax.text(x, y, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "rect":
                x, y = obj["position"]
                w, h = obj["size"]
                self.ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='black', facecolor='lightgray'))
                self.ax.text(x + w / 2, y + h / 2, obj.get("label", ""), ha='center', va='center')
            elif obj["type"] == "line":
                sx, sy = obj["start"]
                ex, ey = obj["end"]
                self.ax.plot([sx, ex], [sy, ey], color='gray', linewidth=4)
            elif obj["type"] == "blocked":
                (x1, y1), (x2, y2) = obj["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                self.ax.add_patch(plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor='black',
                    facecolor='darkgray'
                ))
                self.ax.text((x1 + x2) / 2, (y1 + y2) / 2, obj.get("label", ""), ha='center', va='center', fontsize=9)

        for project in self.sim.environment.construction.get_visual_data():
            cx, cy = project["position"]
            r = project["radius"]
            border = project["border_color"]
            fill = project["fill_color"]
            fill_pct = project["progress"]

            self.ax.add_patch(plt.Circle((cx, cy), r, edgecolor=border, facecolor=fill, linewidth=2))
            self.ax.add_patch(plt.Circle((cx, cy), r * fill_pct, color=border, alpha=0.3))
            self.ax.text(cx, cy, project["label"], ha='center', va='center', fontsize=7)

        for agent in self.sim.agents:
            agent.draw(self.ax)

        self.canvas.draw()
        self.update_agent_table()
        self.update_event_monitor()

    def update_agent_table(self):
        for i in self.agent_state_table.get_children():
            self.agent_state_table.delete(i)
        for agent in self.sim.agents:
            self.agent_state_table.insert("", "end", values=(agent.heart_rate, round(agent.gsr, 3), round(agent.temperature, 2), round(agent.co2_output, 3)))

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def update_event_monitor(self):
        # Section 1: Agent Activities
        self.agent_activity_text.delete("1.0", tk.END)
        for agent in self.sim.agents:
            self.agent_activity_text.insert(tk.END, f"--- {agent.name} ({agent.role}) ---\n")
            self.agent_activity_text.insert(tk.END, f"Goal: {agent.goal}\n")
            self.agent_activity_text.insert(tk.END, f"Target: {agent.target}\n")
            if agent.activity_log:
                self.agent_activity_text.insert(tk.END, f"Last Action: {agent.activity_log[-1]}\n")
            if agent.mental_model["data"]:
                self.agent_activity_text.insert(tk.END, f"Data: {[d.id for d in agent.mental_model['data']]}\n")
            if agent.mental_model["information"]:
                self.agent_activity_text.insert(tk.END, f"Info: {[i.id for i in agent.mental_model['information']]}\n")
            if agent.mental_model["knowledge"].rules:
                self.agent_activity_text.insert(tk.END, f"Rules: {[r for r in agent.mental_model['knowledge'].rules]}\n")
            self.agent_activity_text.insert(tk.END, "\n")

        # Section 2: Interaction State Machines
        self.interaction_state_text.delete("1.0", tk.END)
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

            self.interaction_state_text.insert(tk.END, f"{agent.name} ({agent.role}):\n")
            self.interaction_state_text.insert(tk.END, f"  Current State(s): {', '.join(current_states) or 'Idle'}\n")
            self.interaction_state_text.insert(tk.END, "  Transitions:\n")
            self.interaction_state_text.insert(tk.END, "    - Idle → Accessing Info\n")
            self.interaction_state_text.insert(tk.END, "    - Accessing Info → Sharing\n")
            self.interaction_state_text.insert(tk.END, "    - Sharing → Building\n")
            self.interaction_state_text.insert(tk.END, "    - Any → Moving (if target exists)\n")
            self.interaction_state_text.insert(tk.END, "\n")

        # Section 3: Zone States
        self.zone_state_text.delete("1.0", tk.END)
        for agent in self.sim.agents:
            current_zone = "Transition"
            for zone_name, obj in self.sim.environment.zones.items():
                if "corners" not in obj:
                    continue
                if self.sim.environment._point_in_zone(agent.position, obj["corners"]):
                    current_zone = zone_name
                    break
            self.zone_state_text.insert(tk.END, f"{agent.name} ({agent.role}): {current_zone}\n")

    def create_experiment_tab(self):
        container = ttk.Frame(self.notebook)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.tab_experiment = ttk.Frame(canvas)

        self.tab_experiment.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.tab_experiment, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.notebook.add(container, text="Experiment")

        row = 0
        ttk.Label(self.tab_experiment, text="Speed Multiplier (0.5x to 10x):").grid(row=row, column=0, sticky="w")
        speed_slider = tk.Scale(self.tab_experiment, variable=self.speed_multiplier,
                                from_=0.5, to=10.0, resolution=0.1, orient="horizontal", length=200)
        speed_slider.grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(self.tab_experiment, text="Enable Flash Mode (no animation):").grid(row=row, column=0, sticky="w")
        tk.Checkbutton(self.tab_experiment, variable=self.flash_mode).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(self.tab_experiment, text="Experiment Name:").grid(row=row, column=0, sticky="w")
        self.experiment_name_var = StringVar()
        ttk.Entry(self.tab_experiment, textvariable=self.experiment_name_var).grid(row=row, column=1, sticky="ew")
        row += 1

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

        #stop button



        def update_traits_from_profile(role):
            team = self.agent_profiles[role]["team"].get()
            task = self.agent_profiles[role]["task"].get()

            for trait, value in profile_traits[f"{team}_Team"].items():
                self.agent_traits[role][trait].set(value)
            for trait, value in profile_traits[f"{task}_Task"].items():
                self.agent_traits[role][trait].set(value)

        for role in roles:
            agent_frame = ttk.LabelFrame(self.tab_experiment, text=f"{role} Settings", padding=10)
            agent_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
            row += 1

            self.active_roles[role] = BooleanVar(value=True)
            ttk.Label(agent_frame, text=f"{role} Active:").grid(row=0, column=0, sticky="w")
            ttk.Checkbutton(agent_frame, variable=self.active_roles[role]).grid(row=0, column=1)

            ttk.Label(agent_frame, text="Teamwork Potential:").grid(row=1, column=0, sticky="w")
            ttk.Label(agent_frame, text="Taskwork Potential:").grid(row=1, column=1, sticky="w")

            team_potential = StringVar(value="High")
            task_potential = StringVar(value="High")
            self.agent_profiles[role] = {"team": team_potential, "task": task_potential}

            ttk.OptionMenu(agent_frame, team_potential, "High", "High", "Low",
                           command=lambda _, r=role: update_traits_from_profile(r)).grid(row=2, column=0, sticky="ew")
            ttk.OptionMenu(agent_frame, task_potential, "High", "High", "Low",
                           command=lambda _, r=role: update_traits_from_profile(r)).grid(row=2, column=1, sticky="ew")

            self.agent_traits[role] = {}
            trait_names = list(trait_labels.keys())
            for i, trait in enumerate(trait_names):
                ttk.Label(agent_frame, text=trait_labels[trait] + ":").grid(row=3 + i, column=0, sticky="w")
                self.agent_traits[role][trait] = DoubleVar(value=0.5)
                tk.Scale(
                    agent_frame,
                    variable=self.agent_traits[role][trait],
                    from_=0.0,
                    to=1.0,
                    resolution=0.1,
                    orient="horizontal"
                ).grid(row=3 + i, column=1, columnspan=2, sticky="ew")

            # Packet access via listbox
            ttk.Label(agent_frame, text=f"{role} Packet Access:").grid(row=3 + len(trait_names), column=0, sticky="w")
            self.packet_access[role] = tk.Listbox(agent_frame, selectmode="multiple", exportselection=False, height=4)
            for pkt in ["Team_Packet", "Architect_Packet", "Engineer_Packet", "Botanist_Packet"]:
                self.packet_access[role].insert(tk.END, pkt)
            self.packet_access[role].select_set(0)  # Default to "Team_Packet"
            self.packet_access[role].grid(row=3 + len(trait_names), column=1, columnspan=2, sticky="ew")

        # Final controls
        ttk.Label(self.tab_experiment, text="Number of Simulation Runs:").grid(row=row, column=0, sticky="w")
        self.num_runs = IntVar(value=1)
        ttk.Entry(self.tab_experiment, textvariable=self.num_runs).grid(row=row, column=1)
        row += 1

        ttk.Button(self.tab_experiment, text="Start Experiment", command=self.apply_experiment_settings).grid(
            row=row, column=0, columnspan=2, pady=10
        )
        row += 1  # move down one row for the stop button

        self.stop_button = ttk.Button(self.tab_experiment, text="Stop Experiment", command=self.stop_experiment,
                                      state="disabled")
        self.stop_button.grid(row=row, column=0, columnspan=2, pady=10)



    def stop_experiment(self):
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        if hasattr(self.sim, 'logger'):
            self.sim.logger.save_csv()
        if hasattr(self, 'stop_button'):
            self.stop_button.config(state="disabled")
        print("Simulation stopped and data saved.")


if __name__ == "__main__":
    app = MarsColonyInterface()
    app.run()
