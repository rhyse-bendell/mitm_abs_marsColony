from modules.goal_state import GoalRecord, coerce_goal_source, coerce_goal_status, goal_priority


class GoalManager:
    """Owns authoritative simulator-side goal registry/order/transition bookkeeping."""

    def __init__(self, agent):
        self.agent = agent

    def next_goal_key(self):
        self.agent.goal_transition_counter += 1
        return f"{self.agent.agent_id}:goal:{self.agent.goal_transition_counter}"

    def log_goal_transition(self, sim_state, goal, reason, extra=None):
        payload = {
            "agent": self.agent.name,
            "goal_key": goal.goal_key,
            "goal_id": goal.goal_id,
            "label": goal.label,
            "status": goal.status,
            "priority": round(goal.priority, 3),
            "source": goal.source,
            "goal_level": goal.goal_level,
            "goal_type": goal.goal_type,
            "trust_tier": goal.trust_tier,
            "parent_goal_key": goal.parent_goal_key,
            "reason": reason,
        }
        if extra:
            payload.update(extra)
        self.agent.goal_status_history.append(payload)
        if sim_state is not None:
            sim_state.logger.log_event(sim_state.time, "goal_state_transition", payload)
            sim_state.logger.log_event(sim_state.time, f"goal_{goal.status}", payload)

    def upsert_goal_record(self, **kwargs):
        label = kwargs.get("label")
        goal_id = kwargs.get("goal_id")
        source = kwargs.get("source", "planner_proposed")
        status = kwargs.get("status", "candidate")
        priority = kwargs.get("priority", 0.5)
        target = kwargs.get("target")
        parent_goal_key = kwargs.get("parent_goal_key")
        evidence = kwargs.get("evidence")
        activation_conditions = kwargs.get("activation_conditions")
        completion_conditions = kwargs.get("completion_conditions")
        invalidation_reason = kwargs.get("invalidation_reason")
        blocking_reason = kwargs.get("blocking_reason")
        sim_state = kwargs.get("sim_state")
        reason = kwargs.get("reason", "goal_upsert")
        goal_level = kwargs.get("goal_level")
        goal_type = kwargs.get("goal_type")
        trust_tier = kwargs.get("trust_tier", "normal")

        if goal_id and goal_id in self.agent.goal_registry:
            goal = self.agent.goal_registry[goal_id]
            changed = False
            next_status = coerce_goal_status(status)
            if goal.status != next_status:
                goal.status = next_status
                changed = True
            next_priority = goal_priority(next_status, priority)
            if abs(goal.priority - next_priority) > 1e-6:
                goal.priority = next_priority
                changed = True
            if target is not None:
                goal.target = target
                changed = True
            if evidence:
                goal.evidence = sorted(set((goal.evidence or []) + list(evidence)))
                changed = True
            if activation_conditions:
                goal.activation_conditions = sorted(set((goal.activation_conditions or []) + list(activation_conditions)))
                changed = True
            if completion_conditions:
                goal.completion_conditions = sorted(set((goal.completion_conditions or []) + list(completion_conditions)))
                changed = True
            if blocking_reason:
                goal.blocking_reasons = sorted(set((goal.blocking_reasons or []) + [blocking_reason]))
                changed = True
            if invalidation_reason:
                goal.invalidation_reasons = sorted(set((goal.invalidation_reasons or []) + [invalidation_reason]))
                changed = True
            if parent_goal_key and goal.parent_goal_key != parent_goal_key:
                goal.parent_goal_key = parent_goal_key
                changed = True
                if sim_state is not None:
                    sim_state.logger.log_event(sim_state.time, "goal_linked_to_parent", {"agent": self.agent.name, "goal_id": goal.goal_id, "parent_goal_key": parent_goal_key})
            if goal_level and goal.goal_level != goal_level:
                goal.goal_level = goal_level
                changed = True
            if goal_type and goal.goal_type != goal_type:
                goal.goal_type = goal_type
                changed = True
            if trust_tier and goal.trust_tier != trust_tier:
                goal.trust_tier = trust_tier
                changed = True
            if changed:
                goal.last_transition_reason = reason
                self.log_goal_transition(sim_state, goal, reason)
            return goal

        key = goal_id or self.next_goal_key()
        goal = GoalRecord(
            goal_key=key,
            goal_id=goal_id,
            label=label,
            source=coerce_goal_source(source),
            status=coerce_goal_status(status),
            priority=goal_priority(status, priority),
            target=target,
            parent_goal_key=parent_goal_key,
            evidence=list(evidence or []),
            activation_conditions=list(activation_conditions or []),
            completion_conditions=list(completion_conditions or []),
            invalidation_reasons=[invalidation_reason] if invalidation_reason else [],
            blocking_reasons=[blocking_reason] if blocking_reason else [],
            goal_level=goal_level,
            goal_type=goal_type,
            trust_tier=trust_tier,
            last_transition_reason=reason,
        )
        self.agent.goal_registry[key] = goal
        self.agent.goal_order.append(key)
        self.log_goal_transition(sim_state, goal, reason, extra={"event": "created", "trust_tier": trust_tier})
        if sim_state is not None:
            sim_state.logger.log_event(sim_state.time, "goal_created", {"agent": self.agent.name, "goal_id": goal.goal_id, "goal_key": key, "goal_type": goal.goal_type, "source": goal.source})
        return goal

    def refresh_goal_stack_view(self):
        live = [g for g in self.agent.goal_registry.values() if g.status in {"active", "queued", "candidate", "blocked"}]
        live.sort(key=lambda g: (g.priority, -self.agent.goal_order.index(g.goal_key)), reverse=True)
        self.agent.goal_stack = [{"goal": g.label, "target": g.target, "goal_id": g.goal_id or g.goal_key, "status": g.status, "source": g.source, "parent_goal_key": g.parent_goal_key, "goal_level": g.goal_level, "goal_type": g.goal_type, "trust_tier": g.trust_tier} for g in live]
        self.agent.goal = self.agent.goal_stack[0]["goal"] if self.agent.goal_stack else None

    def current_goal(self):
        self.refresh_goal_stack_view()
        return self.agent.goal_stack[0] if self.agent.goal_stack else None

    def push_goal(self, goal, target=None):
        rec = self.upsert_goal_record(
            label=str(goal),
            goal_id=str(goal) if isinstance(goal, str) and goal.startswith("G_") else None,
            source="legacy_seed",
            status="active",
            target=target,
            reason="legacy_push_goal",
        )
        self.agent.activity_log.append(f"Pushed goal: {rec.label}")
        self.refresh_goal_stack_view()

    def pop_goal(self):
        self.refresh_goal_stack_view()
        if not self.agent.goal_stack:
            return
        top = self.agent.goal_stack[0]
        key = top.get("goal_id")
        goal = self.agent.goal_registry.get(key)
        if goal is None:
            for candidate in self.agent.goal_registry.values():
                if candidate.label == top.get("goal") and candidate.status in {"active", "queued", "candidate", "blocked"}:
                    goal = candidate
                    break
        if goal is None:
            return
        goal.status = "satisfied"
        self.agent.activity_log.append(f"Completed goal: {goal.label}")
        self.refresh_goal_stack_view()
