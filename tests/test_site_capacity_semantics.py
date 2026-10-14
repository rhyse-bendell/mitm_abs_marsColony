"""Site capacity N permits the Nth project, rejects N+1, and an already-created project can start."""

from modules.construction import ConstructionManager


def _fill(cm, site_id, structure_type="house"):
    ids = []
    for i in range(cm.sites[site_id].capacity):
        pid, reason = cm.create_project(site_id, structure_type=structure_type, project_id_override=f"{site_id}_p{i + 1}")
        assert reason == "created"
        ids.append(pid)
    return ids


def test_capacity_permits_nth_project_and_rejects_n_plus_one():
    cm = ConstructionManager()
    ids = _fill(cm, "site_b")
    assert len(ids) == cm.sites["site_b"].capacity
    pid, reason = cm.create_project("site_b", structure_type="house")
    assert pid is None and reason == "site_capacity_reached"


def test_final_capacity_slot_project_can_start_and_receive_resources():
    cm = ConstructionManager()
    ids = _fill(cm, "site_b")
    last = ids[-1]
    started, reason = cm.start_project(last)
    assert (started, reason) == (True, "started")
    assert cm.projects[last]["started"] is True
    assert cm.projects[last]["id"] in cm.sites["site_b"].started_structures
    assert cm.deliver_resource(last, "bricks", 1) is True
    assert cm.projects[last]["delivered_resources"]["bricks"] == 1


def test_every_project_in_a_full_site_can_start_without_double_counting():
    cm = ConstructionManager()
    ids = _fill(cm, "site_b")
    assert all(cm.start_project(pid)[0] for pid in ids)
    assert len(cm.sites["site_b"].started_structures) == cm.sites["site_b"].capacity
    # starting again is idempotent and does not add a second slot
    assert cm.start_project(ids[0])[0] is True
    assert len(cm.sites["site_b"].started_structures) == cm.sites["site_b"].capacity
    summary = {r["site_id"]: r for r in cm.get_site_capacity_summary()}
    assert summary["site_b"]["used"] == summary["site_b"]["capacity"] and summary["site_b"]["remaining"] == 0


def test_other_sites_and_zero_capacity_are_unaffected():
    cm = ConstructionManager()
    assert cm.create_project("site_a", structure_type="house")[1] == "site_capacity_reached"
    assert cm.create_project("site_c", structure_type="house")[1] == "site_locked_by_bridge"
    assert cm._site_has_capacity("site_a") is False
    _fill(cm, "site_b")
    assert cm._site_has_capacity("site_b") is False  # a NEW project still has no room
    assert cm._site_has_capacity("site_b", exclude_project_id="site_b_p1") is True  # an existing one does
