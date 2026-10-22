"""Connector legality (what may be physically connected) is separate from effective service."""

import pytest

from modules.construction import ConstructionManager
from modules.support_accounting import UNDETERMINED_WATER, evaluate_structural_support


@pytest.fixture
def manager():
    cm = ConstructionManager()
    cm.sites["site_b"].capacity = 20
    return cm


def _make(cm, structure_type):
    project_id, _ = cm.create_project(site_id="site_b", structure_type=structure_type, author="system")
    assert project_id, structure_type
    return project_id


def _attach(cm, connector, a, b):
    return cm.attach_connector(connector, from_project_id=a, to_project_id=b, actor="Engineer", sim_time=1.0)


def test_water_generator_to_house_is_legal_in_either_argument_order(manager):
    gen, house, wc = _make(manager, "water_generator"), _make(manager, "house"), _make(manager, "water_connector")
    assert _attach(manager, wc, gen, house) == (True, "attached")
    assert _attach(manager, wc, house, gen) == (False, "already_connected")  # the same direct connection, either order, counts once
    other = ConstructionManager()
    other.sites["site_b"].capacity = 20
    gen2, house2, wc2 = _make(other, "water_generator"), _make(other, "house"), _make(other, "water_connector")
    assert _attach(other, wc2, house2, gen2) == (True, "attached")  # and the reverse argument order is legal on its own


def test_water_generator_to_greenhouse_is_legal(manager):
    gen, gh, wc = _make(manager, "water_generator"), _make(manager, "greenhouse"), _make(manager, "water_connector")
    assert _attach(manager, wc, gen, gh) == (True, "attached")
    assert _attach(manager, wc, gh, gen) == (False, "already_connected")
    other = ConstructionManager()
    other.sites["site_b"].capacity = 20
    gen2, gh2, wc2 = _make(other, "water_generator"), _make(other, "greenhouse"), _make(other, "water_connector")
    assert _attach(other, wc2, gh2, gen2) == (True, "attached")


def test_inappropriate_endpoint_combinations_stay_illegal(manager):
    gen, gen2, house, house2, gh, gh2 = (_make(manager, t) for t in ("water_generator", "water_generator", "house", "house", "greenhouse", "greenhouse"))
    wc, fc = _make(manager, "water_connector"), _make(manager, "food_connector")
    for a, b in ((house, house2), (house, gh), (gh, gh2), (gen, gen2)):
        assert _attach(manager, wc, a, b) == (False, "invalid_connector_endpoint_types")  # water needs a generator and a consumer
    for a, b in ((gen, house), (gen, gh), (house, house2), (gh, gh2)):
        assert _attach(manager, fc, a, b) == (False, "invalid_connector_endpoint_types")  # food needs a greenhouse and housing
    assert _attach(manager, fc, gh, house)[0] is True
    assert manager.connectors == [c for c in manager.connectors if c.get("connector_project_id") == fc]


def test_an_overconnected_generator_is_representable_and_detected_not_prevented(manager):
    gen, wc = _make(manager, "water_generator"), _make(manager, "water_connector")
    houses = [_make(manager, "house") for _ in range(3)]
    for house in houses:
        assert _attach(manager, wc, house, gen)[0] is True  # a third direct connection is not hard-prevented
    for pid in (gen, wc, *houses):
        manager.projects[pid].update({"status": "complete", "validated_complete": True, "correct": True})
    report = evaluate_structural_support(manager)
    assert report["water"]["physical_connections"] == 3 and report["water"]["overconnected_generators"] == [gen]
    assert report["water"]["generators"][0]["allocation_determined"] is False
    assert {r["water"] for r in report["housing"]["structures"]} == {UNDETERMINED_WATER}
