# File: modules/phase_definitions.py

from modules.task_model import load_task_model


def _default_phases():
    try:
        model = load_task_model("mars_colony")
        if model.phases:
            return [
                {
                    "id": p.phase_id,
                    "name": p.name,
                    "duration_minutes": p.duration_minutes,
                    "colonist_manifest": dict(p.colonist_manifest),
                    "unlocks": list(p.unlocks),
                    "required_structures": dict(p.required_structures),
                    "description": p.description,
                }
                for p in model.phases
            ]
    except Exception:
        pass

    return [
        {
            "name": "Phase 1",
            "duration_minutes": 30,
            "colonist_manifest": {"civilians": 50, "VIPs": 0},
            "unlocks": [],
            "required_structures": {"housing": {"civilians": 50, "VIPs": 0}},
            "description": "Prepare for the arrival of 50 civilians.",
        },
        {
            "name": "Phase 2",
            "duration_minutes": 20,
            "colonist_manifest": {"civilians": 40, "VIPs": 20},
            "unlocks": ["bridge_to_zone_C"],
            "required_structures": {"housing": {"civilians": 40, "VIPs": 20}},
            "description": "Expand to support 40 more civilians and 20 VIPs.",
        },
    ]


MISSION_PHASES = _default_phases()
