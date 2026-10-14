"""Structural support accounting: nominal vs effectively supported infrastructure (no allocation policy)."""

import copy
from types import SimpleNamespace

from modules.construction import ConstructionManager
from modules.support_accounting import evaluate_structural_support, load_support_model


def _structure(pid, ptype, valid=True):
    return {"id": pid, "type": ptype, "status": "complete" if valid else "in_progress", "validated_complete": valid, "correct": True, "connections": []}


def _connector(pid, support, pairs, valid=True, build_step_records=0):
    links = [{"connection_id": f"{pid}:build:{i}", "component": "place_connector_base", "from_project_id": None, "to_project_id": None} for i in range(build_step_records)]
    links += [{"connection_id": f"{pid}:link:{i}", "connection_type": support, "from_project_id": a, "to_project_id": b} for i, (a, b) in enumerate(pairs)]
    return {"id": pid, "type": f"{support}_connector", "status": "complete" if valid else "in_progress", "validated_complete": valid, "correct": True, "connections": links}


def _colony(*projects):
    return SimpleNamespace(projects={p["id"]: p for p in projects}, connectors=[])


def _houses(n):
    return [_structure(f"h{i:02d}", "house") for i in range(1, n + 1)]


def test_a_house_with_food_and_water_is_fully_supported():
    colony = _colony(
        _structure("h1", "house"), _structure("gh", "greenhouse"), _structure("g1", "water_generator"),
        _connector("wc", "water", [("g1", "gh"), ("h1", "g1")]), _connector("fc", "food", [("h1", "gh")]),
    )
    report = evaluate_structural_support(colony)
    assert report["housing"]["fully_supported_structures"] == 1
    assert report["housing"]["fully_supported_civilian_capacity"] == 3
    assert report["housing"]["unsupported_structures"] == []
    assert report["violations"] == []


def test_a_valid_house_without_food_is_valid_but_not_fully_supported():
    colony = _colony(_structure("h1", "house"), _structure("g1", "water_generator"), _connector("wc", "water", [("h1", "g1")]))
    housing = evaluate_structural_support(colony)["housing"]
    assert housing["validated_structures"] == 1 and housing["civilian_capacity"] == 3  # nominal
    assert housing["with_effective_water"] == 1 and housing["with_food_link"] == 0
    assert housing["fully_supported_structures"] == 0
    assert housing["unsupported_structures"][0]["missing"] == ["no_food_link"]


def test_a_valid_house_without_water_is_not_fully_supported():
    colony = _colony(
        _structure("h1", "house"), _structure("gh", "greenhouse"), _structure("g1", "water_generator"),
        _connector("wc", "water", [("g1", "gh")]), _connector("fc", "food", [("h1", "gh")]),
    )
    housing = evaluate_structural_support(colony)["housing"]
    assert housing["with_effective_food"] == 1 and housing["with_water_connection"] == 0
    assert housing["fully_supported_structures"] == 0
    assert housing["unsupported_structures"][0]["missing"] == ["no_water_link"]


def test_a_greenhouse_with_soil_but_no_water_has_only_nominal_food():
    colony = _colony(_structure("gh", "greenhouse"), _structure("h1", "house"), _connector("fc", "food", [("h1", "gh")]))
    report = evaluate_structural_support(colony)
    food = report["food"]
    assert food["soil_units"] == 1 and food["civilian_equivalent_capacity"] == 5 and food["vip_equivalent_capacity"] == 2
    assert food["watered_greenhouses"] == 0 and food["unwatered_greenhouses"] == 1
    assert food["effective_civilian_equivalent_capacity"] == 0
    assert report["housing"]["with_food_link"] == 1 and report["housing"]["with_effective_food"] == 0
    assert any(v["type"] == "greenhouse_unwatered" for v in report["violations"])


def test_a_watered_greenhouse_connected_to_housing_provides_effective_food():
    colony = _colony(
        _structure("gh", "greenhouse"), _structure("h1", "house"), _structure("g1", "water_generator"),
        _connector("wc", "water", [("g1", "gh")]), _connector("fc", "food", [("h1", "gh")]),
    )
    report = evaluate_structural_support(colony)
    assert report["food"]["watered_greenhouses"] == 1 and report["food"]["effective_civilian_equivalent_capacity"] == 5
    assert report["food"]["connected_housing"] == 1 and report["housing"]["with_effective_food"] == 1


def test_one_generator_serving_one_structure_delivers_its_full_output():
    colony = _colony(_structure("h1", "house"), _structure("g1", "water_generator"), _connector("wc", "water", [("h1", "g1")]))
    water = evaluate_structural_support(colony)["water"]
    assert water["effective_connections"] == 1 and water["unsupported_connections"] == 0
    assert water["generators"][0]["water_units_per_served_structure"] == 60 and water["overconnected_generators"] == []


def test_one_generator_serving_two_structures_splits_output_evenly():
    colony = _colony(*_houses(2), _structure("g1", "water_generator"), _connector("wc", "water", [("h01", "g1"), ("h02", "g1")]))
    report = evaluate_structural_support(colony)
    assert report["water"]["effective_connections"] == 2 and report["water"]["unsupported_connections"] == 0
    assert report["water"]["generators"][0]["water_units_per_served_structure"] == 30
    assert report["water"]["total_delivered"] == 60 and report["violations"] == []


def test_a_generator_connected_to_more_than_two_structures_is_flagged_and_not_fully_effective():
    colony = _colony(*_houses(3), _structure("g1", "water_generator"), _connector("wc", "water", [(f"h0{i}", "g1") for i in (1, 2, 3)]))
    report = evaluate_structural_support(colony)
    water = report["water"]
    assert water["overconnected_generators"] == ["g1"]
    assert water["effective_connections"] == 2 and water["unsupported_connections"] == 1
    assert water["total_delivered"] == 60  # the generator cannot deliver more than its output
    assert report["housing"]["with_water_connection"] == 3 and report["housing"]["with_effective_water"] == 2
    assert any(v["type"] == "generator_overconnected" and v["unserved"] == ["h03"] for v in report["violations"])


def test_one_generator_with_many_houses_reports_the_water_bottleneck():
    # the pattern seen in simulation: one generator effectively linked to every house
    colony = _colony(
        *_houses(14), _structure("gh", "greenhouse"), _structure("g1", "water_generator"),
        _connector("wc", "water", [(f"h{i:02d}", "g1") for i in range(1, 15)] + [("g1", "gh")], build_step_records=2),
        _connector("fc", "food", [(f"h{i:02d}", "gh") for i in range(1, 15)]),
    )
    report = evaluate_structural_support(colony)
    water = report["water"]
    assert water["connected_housing"] == 14 and water["effective_connections"] == 2 and water["unsupported_connections"] == 13
    assert water["overconnected_generators"] == ["g1"]
    assert report["housing"]["with_effective_water"] == 2  # nowhere near 14
    assert report["housing"]["fully_supported_structures"] <= 2
    reasons = report["bottleneck"]["all_reasons"]
    assert reasons["linked_but_unserved_generator_overconnected"] == 12  # 12 of 14 houses get no water
    assert reasons["food_source_unwatered"] == 14  # and the greenhouse is itself starved by the same generator
    assert report["food"]["greenhouses"][0]["water"] == "linked_but_unserved_generator_overconnected"
    assert report["bottleneck"]["limiting_factor"] in reasons
    # nominal capacity is not mistaken for effective capacity
    assert report["housing"]["civilian_capacity"] == 42 and report["housing"]["fully_supported_civilian_capacity"] <= 6


def test_unvalidated_structures_and_connectors_do_not_contribute():
    colony = _colony(
        _structure("h1", "house"), _structure("h2", "house", valid=False), _structure("g1", "water_generator", valid=False),
        _structure("g2", "water_generator"), _structure("gh", "greenhouse", valid=False),
        _connector("wc", "water", [("h1", "g1")]), _connector("wc2", "water", [("h1", "g2")], valid=False),
    )
    report = evaluate_structural_support(colony)
    assert report["housing"]["validated_structures"] == 1 and report["water"]["validated_generators"] == 1
    assert report["food"]["validated_greenhouses"] == 0
    assert report["water"]["effective_connections"] == 0 and report["housing"]["with_water_connection"] == 0
    assert report["unvalidated_structures"]["housing"] == ["h2"] and report["unvalidated_structures"]["water_generator"] == ["g1"]
    types = {v["type"] for v in report["violations"]}
    assert {"link_to_unvalidated_structure", "connector_not_validated"} <= types


def test_endpoint_less_build_step_records_and_wrong_endpoints_are_not_graph_edges():
    colony = _colony(
        *_houses(2), _structure("g1", "water_generator"),
        _connector("wc", "water", [("h01", "h02")], build_step_records=3),  # house-house water link cannot carry water
    )
    report = evaluate_structural_support(colony)
    assert report["water"]["effective_connections"] == 0
    assert [v["type"] for v in report["violations"]] == ["invalid_link_endpoints"]


def test_duplicate_links_count_once():
    colony = _colony(_structure("h1", "house"), _structure("g1", "water_generator"), _connector("wc", "water", [("h1", "g1"), ("g1", "h1")]))
    report = evaluate_structural_support(colony)
    assert report["water"]["effective_connections"] == 1
    assert [v["type"] for v in report["violations"]] == ["duplicate_link"]


def test_evaluation_is_read_only_and_deterministic():
    colony = _colony(
        *_houses(5), _structure("gh", "greenhouse"), _structure("g1", "water_generator"),
        _connector("wc", "water", [(f"h0{i}", "g1") for i in range(1, 6)]), _connector("fc", "food", [(f"h0{i}", "gh") for i in range(1, 6)]),
    )
    before = copy.deepcopy(colony.projects)
    first = evaluate_structural_support(colony)
    second = evaluate_structural_support(colony)
    assert colony.projects == before
    assert first == second


def test_real_construction_manager_is_not_mutated_by_evaluation():
    manager = ConstructionManager()
    manager.sites["site_b"].capacity = 4
    project_id, _ = manager.create_project(site_id="site_b", structure_type="house", author="system")
    before_projects = copy.deepcopy(manager.projects)
    before_connectors = copy.deepcopy(manager.connectors)
    report = evaluate_structural_support(manager)
    assert manager.projects == before_projects and manager.connectors == before_connectors
    assert report["housing"]["validated_structures"] == 0  # created but not complete/validated
    assert report["unvalidated_structures"]["housing"] == [project_id]


def test_model_is_task_package_driven_with_dik_provenance():
    model = load_support_model("mars_colony")
    assert model["housing"]["capacity_per_floorspace_unit"] == {"civilians": 3, "vips": 2}
    assert model["greenhouse"]["capacity_per_soil_unit"] == {"civilians": 5, "vips": 2}
    assert model["water"]["output_per_generator"] == 60 and model["water"]["max_supported_structures_per_generator"] == 2
    assert model["water"]["provenance"]["max_supported_structures_per_generator"] == "D_ENG_WATERGEN_MAX_2_STRUCTURES"
    # a different atom set changes the result: nothing is hard-coded in the evaluator
    custom = copy.deepcopy(model)
    custom["water"]["max_supported_structures_per_generator"] = 3
    colony = _colony(*_houses(3), _structure("g1", "water_generator"), _connector("wc", "water", [(f"h0{i}", "g1") for i in (1, 2, 3)]))
    assert evaluate_structural_support(colony, custom)["water"]["effective_connections"] == 3
