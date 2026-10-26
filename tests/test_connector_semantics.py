"""Connector semantics (final-materials grounded): a connector is a DIRECT connection between a provider and a consumer.

Starting/building a connector project may be speculative (the simulator abstraction); a connector counts as a validated, functional connection only with
a valid endpoint link. Food: greenhouse -> housing. Water: generator -> housing OR greenhouse. Supply needs a direct connection; provider limits remain.
"""
import contextlib
import io
import tempfile

import pytest

from modules.action_gate import AgentActionGate
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction import ConstructionManager
from modules.simulation import SimulationState
from modules.support_accounting import evaluate_structural_support, validated_structure_breakdown
from tests.construction_helpers import build_step


def _built(cm, pid, ptype, *, validated=True, site="site_b"):
    cm.create_project(site, structure_type=ptype, project_id_override=pid)
    p = cm.projects[pid]
    p["started"] = True
    p["structurally_complete"] = True
    p["resource_complete"] = True
    if validated:
        p["status"], p["validated_complete"], p["correct"] = "complete", True, True
    return pid


def _physically_built(cm, pid):
    """Finish the real build path (materials, start, every build step) so structurally_complete is DERIVED, not forced, and survives update()."""
    cm.deliver_resource(pid, "bricks", int(cm.projects[pid]["required_resources"]["bricks"]))
    cm.update()
    cm.start_project(pid)
    for _ in range(40):
        build_step(cm, pid, actor="test", sim_time=1.0)
    cm.update()


def _evidence(cm, pid):
    """Give a project the claim/evidence/design-note entries the validator wants, so only the connector-link condition is under test."""
    for kind in ("claim", "evidence", "design_note"):
        cm.record_project_epistemic_externalization(pid, entry_type=kind, note=f"{kind} for {pid}", references=[], actor="test", sim_time=1.0)


def _world():
    cm = ConstructionManager()
    for pid, t in (("house1", "house"), ("gh1", "greenhouse"), ("gen1", "water_generator")):
        _built(cm, f"site_b_{pid}", t, validated=True)
    return cm, "site_b_house1", "site_b_gh1", "site_b_gen1"


def test_food_connector_greenhouse_to_house_is_a_valid_countable_connection():
    cm, house, gh, gen = _world()
    _built(cm, "site_b_food_connector_001", "food_connector", validated=False)
    legal, blockers = cm.check_connection("site_b_food_connector_001", gh, house)
    assert legal and not blockers
    assert cm.attach_connector("site_b_food_connector_001", from_project_id=gh, to_project_id=house)[0]
    assert cm.projects[house]["support_counts"]["food"] == 1
    cm.projects["site_b_food_connector_001"].update(status="complete", validated_complete=True, correct=True)
    report = evaluate_structural_support(cm)
    assert report["housing"]["with_food_link"] == 1
    assert validated_structure_breakdown(cm, report)["validated_connectors_with_links"] == 1


def test_water_connector_generator_to_house_is_valid_and_countable():
    cm, house, gh, gen = _world()
    _built(cm, "site_b_water_connector_001", "water_connector", validated=False)
    assert cm.attach_connector("site_b_water_connector_001", from_project_id=gen, to_project_id=house)[0]
    assert cm.projects[house]["support_counts"]["water"] == 1
    cm.projects["site_b_water_connector_001"].update(status="complete", validated_complete=True, correct=True)
    report = evaluate_structural_support(cm)
    assert report["housing"]["with_effective_water"] == {"min": 1, "max": 1}


def test_water_connector_generator_to_greenhouse_is_representable_attachable_and_countable():
    cm, house, gh, gen = _world()
    _built(cm, "site_b_water_connector_001", "water_connector", validated=False)
    legal, blockers = cm.check_connection("site_b_water_connector_001", gen, gh)
    assert legal, blockers  # the final Engineer brief: water connectors move water from generators to housing OR to greenhouses
    assert ("site_b_water_connector_001", frozenset((gen, gh))) in {(o["connector_project_id"], frozenset((o["from_project_id"], o["to_project_id"]))) for o in cm.list_connection_options()}
    assert cm.attach_connector("site_b_water_connector_001", from_project_id=gen, to_project_id=gh) == (True, "attached")
    cm.projects["site_b_water_connector_001"].update(status="complete", validated_complete=True, correct=True)
    report = evaluate_structural_support(cm)
    assert report["food"]["watered_greenhouses"] == {"min": 1, "max": 1}
    assert report["water"]["physical_connections"] == 1


def test_an_endpointless_connector_cannot_become_validation_ready_or_functionally_validated():
    cm, house, gh, gen = _world()
    cm.create_project("site_b", structure_type="water_connector", project_id_override="site_b_water_connector_001")
    _physically_built(cm, "site_b_water_connector_001")
    _evidence(cm, "site_b_water_connector_001")
    readiness = cm.evaluate_project_validation_readiness("site_b_water_connector_001")
    assert readiness["physical_ready"] and readiness["connector_link_ready"] is False and readiness["validation_ready"] is False
    assert "connector_has_no_valid_endpoint_link" in readiness["blockers"]
    assert any(o.get("action") == "connect_structures" for o in readiness["repair_options"])
    cm.mark_validated("site_b_water_connector_001", True, actor="test", sim_time=2.0)  # even a "pass" cannot complete a connector that connects nothing
    assert cm.projects["site_b_water_connector_001"]["validated_complete"] is False
    assert evaluate_structural_support(cm)["water"]["physical_connections"] == 0
    # with a valid endpoint link it becomes validatable
    assert cm.attach_connector("site_b_water_connector_001", from_project_id=gen, to_project_id=house)[0]
    assert cm.evaluate_project_validation_readiness("site_b_water_connector_001")["validation_ready"] is True
    cm.mark_validated("site_b_water_connector_001", True, actor="test", sim_time=3.0)
    assert cm.projects["site_b_water_connector_001"]["validated_complete"] is True


def test_the_action_gate_refuses_validating_an_endpointless_connector_with_the_reason():
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationState(speed=1.0, flash_mode=True, seed=3, outputs_root=tempfile.mkdtemp())
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", "food_connector", author="test")
    p = cm.projects[pid]
    p["started"] = True
    p["structurally_complete"] = True
    agent = sim.agents[0]
    result = AgentActionGate().evaluate(agent=agent, decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=pid), sim_state=sim)
    assert not result.legal and "connector_has_no_valid_endpoint_link" in result.blockers
    with contextlib.redirect_stdout(io.StringIO()):
        sim.stop()


def test_wrong_kind_endpoints_fail_at_attach_and_a_forged_wrong_kind_entry_is_not_a_link():
    cm, house, gh, gen = _world()
    _built(cm, "site_b_house2", "house", validated=True)
    _built(cm, "site_b_food_connector_001", "food_connector", validated=False)
    for a, b in ((house, "site_b_house2"), (gen, house), (gen, gh)):
        legal, blockers = cm.check_connection("site_b_food_connector_001", a, b)
        assert not legal and "invalid_connector_endpoint_types" in blockers
        assert cm.attach_connector("site_b_food_connector_001", from_project_id=a, to_project_id=b) == (False, "invalid_connector_endpoint_types")
    cm.projects["site_b_food_connector_001"].setdefault("connections", []).append({"connection_id": "forged", "connection_type": "food", "from_project_id": house, "to_project_id": "site_b_house2"})
    assert cm.valid_connector_links(cm.projects["site_b_food_connector_001"]) == []  # the validator re-checks kinds
    assert cm.evaluate_project_validation_readiness("site_b_food_connector_001")["connector_link_ready"] is False


def test_duplicate_links_never_double_count():
    cm, house, gh, gen = _world()
    _built(cm, "site_b_water_connector_001", "water_connector", validated=False)
    _built(cm, "site_b_water_connector_002", "water_connector", validated=False)
    assert cm.attach_connector("site_b_water_connector_001", from_project_id=gen, to_project_id=house)[0]
    assert cm.attach_connector("site_b_water_connector_001", from_project_id=house, to_project_id=gen) == (False, "already_connected")  # same pair, either order
    assert cm.attach_connector("site_b_water_connector_002", from_project_id=gen, to_project_id=house) == (False, "already_connected")  # via another connector too
    assert cm.projects[house]["support_counts"]["water"] == 1 and len(cm.connectors) == 1
    cm.projects["site_b_water_connector_001"].setdefault("connections", []).append(dict(cm.projects["site_b_water_connector_001"]["connections"][-1], connection_id="dup"))
    assert len(cm.valid_connector_links(cm.projects["site_b_water_connector_001"])) == 1
    cm.projects["site_b_water_connector_001"].update(status="complete", validated_complete=True, correct=True)
    assert evaluate_structural_support(cm)["water"]["physical_connections"] == 1


def test_direct_connection_semantics_are_intact_overconnection_is_representable_not_served():
    cm, house, gh, gen = _world()
    for i in (2, 3):
        _built(cm, f"site_b_house{i}", "house", validated=True)
    for i, target in enumerate((house, gh, "site_b_house2"), start=1):  # (site B has 8 slots: 3 houses + greenhouse + generator + 3 connectors)
        _built(cm, f"site_b_water_connector_{i:03d}", "water_connector", validated=True)
        assert cm.attach_connector(f"site_b_water_connector_{i:03d}", from_project_id=gen, to_project_id=target)[0]  # a 3rd/4th link is NOT hard-prevented
    report = evaluate_structural_support(cm)
    assert report["water"]["overconnected_generators"] == [gen]
    gen_row = report["water"]["generators"][0]
    assert gen_row["connection_count"] == 3 and gen_row["overconnected"] is True and gen_row["max_serviceable_structures"] == 2 and gen_row["service_slots"] == 2
    assert gen_row["excess_connections"] == 1 and gen_row["allocation_determined"] is False  # no endpoint is declared the winner
    assert report["water"]["physical_connections"] == 3 and report["water"]["effective_connections"] == 2 and report["water"]["unsupported_connections"] == 1
    # and supply does not pass THROUGH a structure: a house linked only to a greenhouse gets no water from the generator behind it
    cm2 = ConstructionManager()
    _built(cm2, "site_b_gen1", "water_generator")
    _built(cm2, "site_b_gh1", "greenhouse")
    _built(cm2, "site_b_house1", "house")
    _built(cm2, "site_b_water_connector_001", "water_connector")
    _built(cm2, "site_b_food_connector_001", "food_connector")
    assert cm2.attach_connector("site_b_water_connector_001", from_project_id="site_b_gen1", to_project_id="site_b_gh1")[0]
    assert cm2.attach_connector("site_b_food_connector_001", from_project_id="site_b_gh1", to_project_id="site_b_house1")[0]
    housing = evaluate_structural_support(cm2)["housing"]
    assert housing["with_effective_water"] == {"min": 0, "max": 0} and housing["fully_supported_structures"] == {"min": 0, "max": 0}
