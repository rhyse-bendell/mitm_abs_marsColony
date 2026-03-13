# File: main.py
# Location: C:\Post-doc Work\MarsColonyTeamTask\ABS\main.py

from modules.environment import Environment
from modules.agent import Agent
from modules.logger import Logger
from modules.visualizer import Visualizer

# Define mission phase information directly here for now
# (Can be modularized into modules/config.py later)
MISSION_PHASES = [
    {
        "name": "Phase 1",
        "duration_minutes": 15,
        "colonist_manifest": {
            "civilians": 50,
            "VIPs": 0
        },
        "unlocks": [],
        "description": "Prepare the colony for the arrival of 50 civilians."
    },
    {
        "name": "Phase 2",
        "duration_minutes": 15,
        "colonist_manifest": {
            "civilians": 40,
            "VIPs": 20
        },
        "unlocks": ["bridge_to_zone_C"],
        "description": "Expand the colony to support 40 additional civilians and 20 VIPs."
    }
]

#bring in the agent starting locations as dictated in environment.py
from modules.environment import AGENT_START_POSITIONS

agents = [
    Agent(role="Architect", position=AGENT_START_POSITIONS["Architect"]),
    Agent(role="Botanist", position=AGENT_START_POSITIONS["Botanist"]),
    Agent(role="Engineer", position=AGENT_START_POSITIONS["Engineer"])
]

# --- Simulation Parameters ---
TIME_STEP = 0.1  # Each tick = 0.1 second
SIMULATION_DURATION = sum(phase["duration_minutes"] for phase in MISSION_PHASES) * 60  # Total time in seconds

def main():
    # -----------------------------------------------
    # Initialize the simulation environment
    # -----------------------------------------------
    environment = Environment(phases=MISSION_PHASES)

    # -----------------------------------------------
    # Create agents with their respective roles
    # -----------------------------------------------
    agents = [
        Agent(role="Architect"),
        Agent(role="Botanist"),
        Agent(role="Engineer")
    ]

    # -----------------------------------------------
    # 🧾 Initialize logging system for playback & analysis
    # -----------------------------------------------
    logger = Logger()

    # -----------------------------------------------
    # 🎥 Initialize visualizer (draw current frame or record)
    # -----------------------------------------------
    visualizer = Visualizer(environment, agents)

    # -----------------------------------------------
    # ⏱ Main simulation loop
    # -----------------------------------------------
    time = 0.0
    while time < SIMULATION_DURATION:
        # STEP 1: Agents perceive the environment
        for agent in agents:
            agent.perceive(environment)

        # STEP 2: Agents decide what to do (move, talk, act)
        for agent in agents:
            agent.decide()

        # STEP 3: Agents carry out actions
        for agent in agents:
            agent.act(environment)

        # STEP 4: Update environment state (e.g., gas levels, construction progress, phase changes)
        environment.update(time)

        # STEP 5: Log current state for later playback or analysis
        logger.log_state(time, environment, agents)

        # STEP 6: Render a visual frame (can be adjusted for live vs. post hoc playback)
        visualizer.render(time)

        # STEP 7: Advance simulation time
