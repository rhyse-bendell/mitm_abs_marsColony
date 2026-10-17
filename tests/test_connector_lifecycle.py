"""Connector lifecycle against the task package: what is legal to start, what a connector project is, and what counts as a functional connection.

Characterization (not new legality): a connector PROJECT may be started and built speculatively (legal, consumes a site slot like every
project); only link entries with both endpoint kinds legal per support_model.json are functional edges; a validated connector project with no link
provides nothing and is reported separately from a connector that links a provider and a consumer."""
import pytest

from modules.construction import ConstructionManager
from modules.support_accounting import evaluate_structural_support, validated_structure_breakdown


def _built(cm, pid, ptype, validated=True, site="site_b"):
    cm.create_project(site, structure_type=ptype, project_id_override=pid)
    p = cm.projects[pid]
    p["started"] = True
    p["structurally_complete"] = True
    p["resource_complete"] = True
    if validated:
        p["status"], p["validated_complete"], p["correct"] = "complete", True, True
    return pid


def test_connector_start_is_legal_with_no_endpoint_structures_at_all():
    cm = ConstructionManager()
    assert not cm.projects
    legal_types = {(o["site_id"], o["structure_type"]) for o in cm.list_start_options()}
    assert ("site_b", "food_connector") in legal_types and ("site_b", "water_connector") in legal_types  # speculative construction is offered
    pid, status = cm.start_new_project("Build_Site_B", "food_connector")
    assert pid and status in {"created", "exists"}


def test_an_endpointless_connector_consumes_a_site_slot_like_any_project():
    cm = ConstructionManager()
    used = lambda: next(s["used"] for s in cm.get_site_capacity_summary() if s["site_id"] == "site_b")
    cap = next(s["capacity"] for s in cm.get_site_capacity_summary() if s["site_id"] == "site_b")
    for _ in range(cap):
        assert cm.start_new_project("Build_Site_B", "food_connector")[0]
    assert used() == cap
    legal, blockers, _ = cm.check_start("Build_Site_B", "house")
    assert not legal and "site_capacity_reached" in blockers  # no structure can be started any more at that site


def test_a_validated_endpointless_connector_project_is_not_a_functional_connection():
    cm = ConstructionManager()
    _built(cm, "site_b_food_connector_001", "food_connector")
    report = evaluate_structural_support(cm)
    assert report["housing"]["validated_structures"] == 0 and report["food"]["validated_greenhouses"] == 0 and report["water"]["physical_connections"] == 0
    b = validated_structure_breakdown(cm, report)
    assert b["validated_connectors"] == 1 and b["validated_core_structures"] == 0
    assert b["validated_connectors_without_links"] == 1 and b["validated_connectors_with_links"] == 0
    assert b["connector_projects_with_links"] == 0 and b["speculative_connector_projects"] == 1 and b["connector_links_total"] == 0


def test_wrong_kind_endpoint_combinations_are_not_offered_and_are_refused():
    cm = ConstructionManager()
    for pid, t in (("site_b_house_001", "house"), ("site_b_house_002", "house"), ("site_b_greenhouse_001", "greenhouse"), ("site_b_water_generator_001", "water_generator")):
        _built(cm, pid, t, validated=False)
    _built(cm, "site_b_food_connector_001", "food_connector", validated=False)
    _built(cm, "site_b_water_connector_001", "water_connector", validated=False)
    opts = {(o["connector_project_id"], frozenset((o["from_project_id"], o["to_project_id"]))) for o in cm.list_connection_options()}
    food = {pair for c, pair in opts if c == "site_b_food_connector_001"}
    water = {pair for c, pair in opts if c == "site_b_water_connector_001"}
    assert food == {frozenset(("site_b_greenhouse_001", "site_b_house_001")), frozenset(("site_b_greenhouse_001", "site_b_house_002"))}  # greenhouse -> housing only
    assert water == {frozenset(("site_b_water_generator_001", x)) for x in ("site_b_house_001", "site_b_house_002", "site_b_greenhouse_001")}  # generator -> housing | greenhouse
    for connector, a, b in (("site_b_food_connector_001", "site_b_house_001", "site_b_house_002"), ("site_b_food_connector_001", "site_b_water_generator_001", "site_b_house_001"),
                            ("site_b_water_connector_001", "site_b_house_001", "site_b_greenhouse_001"), ("site_b_water_connector_001", "site_b_house_001", "site_b_house_002")):
        legal, blockers = cm.check_connection(connector, a, b)
        assert not legal and "invalid_connector_endpoint_types" in blockers
    assert not cm.attach_connector("site_b_food_connector_001", from_project_id="site_b_house_001", to_project_id="site_b_house_002")[0]


def test_generator_to_greenhouse_water_link_is_representable_and_counts_only_when_validated():
    cm = ConstructionManager()
    _built(cm, "site_b_water_generator_001", "water_generator")
    _built(cm, "site_b_greenhouse_001", "greenhouse")
    _built(cm, "site_b_water_connector_001", "water_connector", validated=False)
    ok, why = cm.attach_connector("site_b_water_connector_001", from_project_id="site_b_water_generator_001", to_project_id="site_b_greenhouse_001")
    assert ok and why == "attached"
    unvalidated = evaluate_structural_support(cm)
    assert unvalidated["food"]["watered_greenhouses"] == {"min": 0, "max": 0}  # an unvalidated connector's links are ignored (connector_not_validated)
    assert any(v["type"] == "connector_not_validated" for v in unvalidated["violations"])
    p = cm.projects["site_b_water_connector_001"]
    p["status"], p["validated_complete"], p["correct"] = "complete", True, True
    report = evaluate_structural_support(cm)
    assert report["food"]["watered_greenhouses"] == {"min": 1, "max": 1}
    b = validated_structure_breakdown(cm, report)
    assert b["validated_connectors_with_links"] == 1 and b["connector_links_total"] == 1 and b["structurally_supported_greenhouses"] == {"min": 1, "max": 1}
    assert b["validated_core_structures"] == 2 and b["speculative_connector_projects"] == 0
