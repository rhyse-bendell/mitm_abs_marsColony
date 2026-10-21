"""Headless host for a realtime human-participant session (no observer GUI). Prints the participant URL(s); runs until the mission ends, --minutes elapse, or Ctrl+C.

    python scripts/run_human_testbed.py --human Architect --seed 3
    python scripts/run_human_testbed.py --controller Architect=human_participant --controller Botanist=procedural_baseline --minutes 10

The participant server binds 127.0.0.1 only. For the observer GUI use interface.py and choose 'Human Participant' for a role.
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.controller_spec import parse_cli_assignments  # noqa: E402
from modules.realtime_testbed import HumanTestbedSession  # noqa: E402
from modules.simulation import SimulationState  # noqa: E402


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # the logger prints a check-mark; a cp1252 console would crash at shutdown
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", action="append", default=[], help="role controlled by a human participant (repeatable)")
    ap.add_argument("--controller", action="append", default=[], help="Role=controller_type[:model]")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this many simulated minutes (default: the mission length)")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--experiment-name", default="human_testbed")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    assignments = parse_cli_assignments(args.controller)
    for role in args.human:
        assignments[role] = "human_participant"
    if not any(v == "human_participant" or (isinstance(v, dict) and v.get("controller_type") == "human_participant") for v in assignments.values()):
        ap.error("at least one human participant is required (--human ROLE)")
    sim = SimulationState(speed=1.0, seed=args.seed, experiment_name=args.experiment_name, controller_assignments=assignments, execution_mode="realtime_testbed")
    session = HumanTestbedSession(sim, port=args.port, time_scale=args.time_scale).start()
    for role, url in session.urls().items():
        print(f"[{role}] participant URL: {url}")
        if not args.no_browser:
            webbrowser.open(url)
    until = None if args.minutes is None else args.minutes * 60.0
    try:
        while not session.finished and (until is None or sim.time < until):
            session.tick()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    print(f"session ended at t={sim.time:.1f}s; outputs: {sim.logger.output_session.session_folder}")


if __name__ == "__main__":
    main()
