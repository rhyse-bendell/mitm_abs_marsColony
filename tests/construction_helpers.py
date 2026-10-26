"""Test helpers for specification-aware construction: the specification choice is the Pilot's, so tests make it explicitly."""


def compliant_option(cm, project_id):
    """The pending step's first compliant option (a TEST-side stand-in for an informed Pilot; the engine never does this)."""
    project = cm.projects[project_id]
    pending = cm.pending_specification(project_id)
    if pending is None:
        return None
    return sorted(cm.specified_slot(project.get("type"), pending["component"]).compliant)[0]


def build_step(cm, project_id, *, actor=None, sim_time=None, option="compliant", **kw):
    """execute_build_step with an explicit design option (default: a compliant one; pass option=<name> or option=None for none)."""
    if option == "compliant":
        option = compliant_option(cm, project_id)
    return cm.execute_build_step(project_id, actor=actor, sim_time=sim_time, intended_option=option, **kw)


def build_all_steps(cm, project_id, *, actor=None, sim_time=None, option="compliant"):
    results = []
    while cm.projects[project_id].get("build_steps") and any(not s.get("completed") for s in cm.projects[project_id]["build_steps"]):
        ok, reason, step = build_step(cm, project_id, actor=actor, sim_time=sim_time, option=option)
        results.append((ok, reason, step))
        if not ok:
            break
    return results


def design_args_for(cm, project_id, option="compliant"):
    pending = cm.pending_specification(project_id)
    if pending is None:
        return {}
    return {"design_slot": pending["slot_id"], "design_option": compliant_option(cm, project_id) if option == "compliant" else option}


def realize_compliant(cm, project_id, actor=None, sim_time=0.0):
    """Record every slot of a (test-assembled) completed project as intended and realized compliant."""
    project = cm.projects[project_id]
    for _step, spec in cm.project_slots(project):
        good = sorted(spec.compliant)[0]
        cm.realize_specification(project_id, slot_id=spec.slot_id, intended=good, realized=good, slipped=False, fidelity=0.5, actor=actor, sim_time=sim_time)


def inject_defect(cm, project_id, slot="airlock_opening", actor="X", sim_time=0.5, update=True):
    """A realized defect: an execution slip made one slot non-compliant. Works on projects with a specification; legacy flag otherwise."""
    project = cm.projects[project_id]
    spec = next((sp for _st, sp in cm.project_slots(project) if sp.slot_id == slot), None)
    if spec is None:
        return cm.record_build_quality(project_id, correct=False, actor=actor)
    wrong = next(o for o in spec.options if o not in spec.compliant)
    cm.realize_specification(project_id, slot_id=slot, intended=sorted(spec.compliant)[0], realized=wrong, slipped=True, fidelity=0.5, actor=actor, sim_time=sim_time)
    if update:
        cm.update()
    return True
