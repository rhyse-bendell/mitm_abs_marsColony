"""Run one experimental condition headlessly: python scripts/run_experiment.py --seed 2 --teamwork 0.75 --set memory_reliability=0.3"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.experiment_runner import main

if __name__ == "__main__":
    raise SystemExit(main())