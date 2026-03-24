from copy import deepcopy

from modules.simulation import SimulationState


def run_batch_experiment(settings, progress_callback=None, run_name_builder=None):
    """Execute repeated headless simulation runs from resolved GUI settings."""
    resolved_settings = dict(settings or {})
    num_runs = max(1, int(resolved_settings.get("num_runs", 1) or 1))
    timesteps_per_run = max(1, int(resolved_settings.get("timesteps_per_run", 300) or 300))
    base_dt = float(resolved_settings.get("base_dt", 1.0) or 1.0)
    agent_configs = deepcopy(resolved_settings.get("agent_configs") or [])
    experiment_name = resolved_settings.get("experiment_name")

    for run_index in range(1, num_runs + 1):
        if callable(run_name_builder):
            run_experiment_name = run_name_builder(experiment_name, run_index, num_runs)
        else:
            base_name = (experiment_name or "experiment").strip() or "experiment"
            run_experiment_name = f"{base_name}_run{run_index:03d}" if num_runs > 1 else base_name

        if callable(progress_callback):
            progress_callback(
                {
                    "type": "batch_run_start",
                    "run_index": run_index,
                    "num_runs": num_runs,
                    "timestep": 0,
                    "timesteps_per_run": timesteps_per_run,
                    "experiment_name": run_experiment_name,
                }
            )

        sim = SimulationState(
            agent_configs=deepcopy(agent_configs),
            num_runs=num_runs,
            speed=resolved_settings.get("speed", 1.0),
            experiment_name=run_experiment_name,
            phases=resolved_settings.get("phases"),
            flash_mode=True,
            brain_backend=resolved_settings.get("brain_backend", "rule_brain"),
            brain_backend_options=deepcopy(resolved_settings.get("brain_backend_options") or {}),
            planner_config=deepcopy(resolved_settings.get("planner_config") or {}),
            construction_parameters=deepcopy(resolved_settings.get("construction_parameters") or {}),
            task_id=resolved_settings.get("task_id", "mars_colony"),
            execution_metadata={
                "batch_mode": True,
                "batch_run_index": run_index,
                "batch_total_runs": num_runs,
                "timesteps_per_run": timesteps_per_run,
                "batch_label": resolved_settings.get("batch_label"),
            },
        )

        for timestep in range(1, timesteps_per_run + 1):
            sim.update(base_dt)
            if callable(progress_callback):
                progress_callback(
                    {
                        "type": "batch_timestep",
                        "run_index": run_index,
                        "num_runs": num_runs,
                        "timestep": timestep,
                        "timesteps_per_run": timesteps_per_run,
                        "experiment_name": run_experiment_name,
                    }
                )

        sim.stop()
        if callable(progress_callback):
            progress_callback(
                {
                    "type": "batch_run_complete",
                    "run_index": run_index,
                    "num_runs": num_runs,
                    "timestep": timesteps_per_run,
                    "timesteps_per_run": timesteps_per_run,
                    "experiment_name": run_experiment_name,
                }
            )
