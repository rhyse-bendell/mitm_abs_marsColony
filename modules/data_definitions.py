# File: modules/data_definitions.py

# This file provides a full reference list of possible data elements for researchers.
# It is not used directly during simulation—see `knowledge.py` for actual packets.

DATA_ELEMENTS = [
    # --- Water Generator Construction ---
    {"id": "D001", "description": "Each water generator provides 60 units of water total."},
    {"id": "D002", "description": "Water generator output is split evenly between connected structures."},
    {"id": "D003", "description": "Each water generator can support only 2 structures."},
    {"id": "D004", "description": "Water generators must be placed on gray foundation."},
    {"id": "D005", "description": "Water bricks are translucent blue."},
    {"id": "D006", "description": "A water generator must be 2x2 wide and 2 bricks high (of water)."},
    {"id": "D007", "description": "Water generators must be capped with a uniform-colored brick."},

    # --- Housing Construction ---
    {"id": "D008", "description": "Houses must be completely enclosed (walls + ceiling)."},
    {"id": "D009", "description": "Houses must include at least one brick of space between floor and ceiling."},
    {"id": "D010", "description": "Houses must have an airlock: 1 brick wide and 1 brick high opening."},
    {"id": "D011", "description": "3 civilians or 2 VIPs can be accommodated by one 2x2 brick of floorspace."},
    {"id": "D012", "description": "VIPs require solid pink floors."},
    {"id": "D013", "description": "Floors must be a single solid color per house."},
    {"id": "D014", "description": "Walls and ceilings must be a different color from floors and may be multi-colored."},

    # --- Structure Placement Rules ---
    {"id": "D015", "description": "All structures must be placed at least 6 studs away from each other."},
    {"id": "D016", "description": "Each water generator, greenhouse, and house is considered a separate structure."},

    # --- Greenhouse and Food Rules ---
    {"id": "D017", "description": "Greenhouses require water to function."},
    {"id": "D018", "description": "Plants require water."},
    {"id": "D019", "description": "Colonists require food and water."},

    # --- Soil & Plant Support ---
    {"id": "D020", "description": "A correctly constructed soil brick supports 5 civilians or 2 VIPs."},

    # --- Movement & Logistics ---
    {"id": "D021", "description": "Bricks must be transported using Resource Carts."},
    {"id": "D022", "description": "Resource carts must be pulled using a rubber band."},
    {"id": "D023", "description": "Resource carts must have at least 1 wheel."},
    {"id": "D024", "description": "Carts can only dump bricks at construction sites."},
    {"id": "D025", "description": "Carts must be repaired if damaged before re-crossing bridges."}
]
