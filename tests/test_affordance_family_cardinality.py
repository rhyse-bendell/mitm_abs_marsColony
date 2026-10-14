"""Affordance cardinality must not inflate an action family's probability (family-then-variant selection)."""

import random
import tempfile
from collections import defaultdict

import pytest

from modules.action_schema import ExecutableActionType as A
from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="cardinality_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _family_mass(probs, family):
    return sum(p for key, p in probs.items() if key.split(":", 1)[1] == family)


def test_equivalent_variants_do_not_multiply_family_probability(sim):
    brain = sim.brain_provider
    one = {"0:start_construction": 1.0, "1:wait": 0.0, "2:inspect_information_source": 0.4}
    five = {**{f"{i}:start_construction": 1.0 for i in range(5)}, "5:wait": 0.0, "6:inspect_information_source": 0.4}
    _, p1 = brain._family_then_variant_pick(one, 0.7, random.Random(3))
    _, p5 = brain._family_then_variant_pick(five, 0.7, random.Random(3))
    assert _family_mass(p5, "start_construction") == pytest.approx(_family_mass(p1, "start_construction"))
    assert _family_mass(p5, "wait") == pytest.approx(_family_mass(p1, "wait"))
    assert sum(p5.values()) == pytest.approx(1.0)
    # the flat softmax this replaces would have given START five times the weight
    flat = brain._softmax_probs(five, 0.7)
    assert _family_mass(flat, "start_construction") > 1.3 * _family_mass(p1, "start_construction")


def test_variant_preferences_survive_inside_the_chosen_family(sim):
    brain = sim.brain_provider
    scored = {"0:start_construction": 2.0, "1:start_construction": 0.2, "2:start_construction": 0.2, "3:wait": 0.0}
    _, probs = brain._family_then_variant_pick(scored, 0.7, random.Random(1))
    assert probs["0:start_construction"] > 3 * probs["1:start_construction"]
    assert probs["1:start_construction"] == pytest.approx(probs["2:start_construction"])
    # a worse extra variant neither dilutes nor inflates the family
    _, without = brain._family_then_variant_pick({"0:start_construction": 2.0, "3:wait": 0.0}, 0.7, random.Random(1))
    assert _family_mass(probs, "start_construction") == pytest.approx(_family_mass(without, "start_construction"))


def test_selection_is_deterministic_for_a_seed_and_follows_the_probabilities(sim):
    brain = sim.brain_provider
    scored = {**{f"{i}:start_construction": 0.5 for i in range(5)}, "5:wait": 0.5}
    first = [brain._family_then_variant_pick(scored, 0.7, random.Random(s))[0] for s in range(40)]
    assert first == [brain._family_then_variant_pick(scored, 0.7, random.Random(s))[0] for s in range(40)]
    starts = sum(1 for key in first if key.endswith("start_construction"))
    assert 8 <= starts <= 32  # ~half: two equally scored families, not 5 of 6 entries
    chosen = {key for key in first if key.endswith("start_construction")}
    assert len(chosen) > 1  # variants still compete inside the family


def test_real_start_affordances_one_type_versus_five_types_have_equal_family_mass(sim):
    """Same policy state, one legal START target versus five equivalent ones."""
    cm = sim.environment.construction
    agent = sim.agents[0]
    brain = sim.brain_provider
    features = defaultdict(float)
    features["productive_taskwork_available"] = 1.0

    def start_family_mass():
        affordances = sorted(sim.brain_context_builder.build(sim, agent).action_affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
        scores = brain._score_actions_for_mode(sorted_affordances=affordances, selected_mode="CONSTRUCT", features=features, mechanism_profile={})
        _, probs = brain._family_then_variant_pick(scores, brain.policy_config.action_selection_temperature, random.Random(5))
        n = sum(1 for a in affordances if a["action_type"] == A.START_CONSTRUCTION.value)
        return _family_mass(probs, A.START_CONSTRUCTION.value), n

    all_types = cm.startable_structure_types()
    mass_five, n_five = start_family_mass()
    cm.startable_structure_types = lambda: all_types[:1]  # a task package with a single startable type
    mass_one, n_one = start_family_mass()
    assert (n_one, n_five) == (1, 5)
    assert mass_five == pytest.approx(mass_one, rel=0.02)  # the tiny idx-based floor term is the only difference