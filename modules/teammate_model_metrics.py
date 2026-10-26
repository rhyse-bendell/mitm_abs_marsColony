"""Analysis-only measures of teammate-model quality against separately held ground truth.

This module is for measurement and tests. It deliberately reads every agent's actual mental model, so it must never be
imported by agent, Pilot, or middleware code: an agent's belief about a teammate (``Agent.theory_of_mind``) is built only
from evidence available to that agent, and this module is how that belief is compared with what the teammate really holds.
"""

from modules.task_model import normalize_rule_token


def held_tokens(agent):
    """Ground truth: the canonical ``data:``/``information:``/``rule:`` tokens an agent actually holds right now."""
    tokens = {f"data:{d.id}" for d in agent.mental_model["data"]}
    tokens |= {f"information:{i.id}" for i in agent.mental_model["information"]}
    tokens |= {f"rule:{normalize_rule_token(r)}" for r in agent.mental_model["knowledge"].rules if normalize_rule_token(r)}
    return tokens


def believed_tokens(observer, target_name):
    """What ``observer`` believes ``target_name`` holds (its evidence-grounded teammate model)."""
    entry = (observer.theory_of_mind or {}).get(target_name) or {}
    return set(entry.get("believed_ids") or set())


def model_accuracy(observer, target):
    """Precision/recall of the observer's belief about one teammate against that teammate's actual holdings."""
    believed, truth = believed_tokens(observer, target.name), held_tokens(target)
    tp = len(believed & truth)
    return {
        "observer": observer.name,
        "target": target.name,
        "believed": len(believed),
        "held": len(truth),
        "true_positive": tp,
        "precision": (tp / len(believed)) if believed else None,
        "recall": (tp / len(truth)) if truth else None,
    }


def team_model_accuracy(agents):
    """Per observer->target accuracy rows plus micro-averaged precision/recall over all pairs."""
    rows = [model_accuracy(o, t) for o in agents for t in agents if o is not t]
    believed, held, tp = (sum(r[k] for r in rows) for k in ("believed", "held", "true_positive"))
    return {"pairs": rows, "precision": (tp / believed) if believed else None, "recall": (tp / held) if held else None}


def routing_quality(agents, kinds=("data", "information")):
    """Expected share of requests routed to an actual holder, using the agent's own routing score (ties split evenly).

    For every element an observer lacks that at least one teammate holds, the observer's candidate scores come from
    ``Agent._request_route_score`` (belief + public role responsibility + past provision, never actual holdings); the metric
    then checks the top-scoring candidate(s) against ground truth. A uniformly random choice among teammates is the reference.
    """
    outcomes, chance = [], []
    for observer in agents:
        mine = held_tokens(observer)
        others = [a for a in agents if a is not observer]
        universe = {t for a in others for t in held_tokens(a) if t.split(":")[0] in kinds} - mine
        for token in sorted(universe):
            scores = {a.name: observer._request_route_score(a, 0.0, [token]) for a in others}
            top = max(scores.values())
            tied = [a for a in others if scores[a.name] == top]
            holders = {a.name for a in others if token in held_tokens(a)}
            outcomes.append(sum(1 for a in tied if a.name in holders) / len(tied))
            chance.append(len(holders) / len(others))
    n = len(outcomes)
    return {"requests_scored": n, "routing_quality": (sum(outcomes) / n) if n else None, "random_routing_reference": (sum(chance) / n) if n else None}
