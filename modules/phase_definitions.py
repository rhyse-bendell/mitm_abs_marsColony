# File: modules/phase_definitions.py

MISSION_PHASES = [
    {
        "name": "Phase 1",
        "duration_minutes": 30,
        "colonist_manifest": {
            "civilians": 50,
            "VIPs": 0
        },
        "unlocks": [],
        "required_structures": {
            "housing": {"civilians": 50, "VIPs": 0}
        },
        "description": "Prepare for the arrival of 50 civilians."
    },
    {
        "name": "Phase 2",
        "duration_minutes": 20,
        "colonist_manifest": {
            "civilians": 40,
            "VIPs": 20
        },
        "unlocks": ["bridge_to_zone_C"],
        "required_structures": {
            "housing": {"civilians": 40, "VIPs": 20}
        },
        "description": "Expand to support 40 more civilians and 20 VIPs."
    }
]
