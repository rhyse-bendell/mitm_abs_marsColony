"""Local Ollama controller: loopback-only, per-agent, canonical-context-only prompts, deterministic budgeting, structured choice by affordance_id,
bounded retries with NO fallback to another controller, durable decision traces, and a single controller-selection path (ControllerSpec).

Every test uses a fake client injected through `modules.ollama_pilot.client_factory` or a stub HTTP server bound to 127.0.0.1; nothing here talks to a real
model, and nothing leaves the machine."""

import contextlib
import http.server
import io
import json
import tempfile
import threading
from pathlib import Path

import pytest

from modules import controller_spec as cs
from modules import ollama_pilot
from modules.controller_context import build_canonical_context
from modules.ollama_client import (NonLoopbackEndpointError, OllamaClient, OllamaRequestError, validate_loopback_endpoint)
from modules.ollama_prompt import build_prompt
from modules.simulation import SimulationState

ROLES = ["Architect", "Botanist", "Engineer"]


# ------------------------------------------------------------------ fakes
def first_inspect_reply(messages, options, model):
    ctx = json.loads(messages[1]["content"].split("CONTEXT:\n", 1)[1].split("\n\nChoose", 1)[0])
    legal = ctx["executable_actions"]
    pick = next((a for a in legal if a["action_type"] == "inspect_information_source"), legal[0])
    return json.dumps({"affordance_id": pick["affordance_id"], "arguments": {}, "decision_note": "test"})


class FakeOllama:
    def __init__(self, base_url="http://127.0.0.1:11434", timeout_s=120.0, *, models=("model-a", "model-b"), available=True, reply=first_inspect_reply, registry=None):
        self.base_url, self.timeout_s, self.models, self.available, self.reply = base_url, timeout_s, list(models), available, reply
        self.chat_calls = []
        if registry is not None:
            registry.append(self)

    def status(self):
        return {"available": self.available, "base_url": self.base_url, "version": "0.0-fake" if self.available else None, "error": None if self.available else "connection refused"}

    def installed_model_names(self):
        return list(self.models)

    def list_models(self):
        return [{"name": m} for m in self.models]

    def chat(self, *, model, messages, options, response_format=None, think=False, timeout_s=None, keep_alive=None):
        self.chat_calls.append({"model": model, "messages": messages, "options": options, "format": response_format, "think": think})
        content = self.reply(messages, options, model)
        return {"content": content, "latency_s": 0.01, "done_reason": "stop", "token_metadata": {"prompt_eval_count": 10, "eval_count": 5}}


@pytest.fixture(autouse=True)
def _never_a_real_service(monkeypatch):
    """Default for every test here: a fake local service (individual tests replace it). No test depends on a real Ollama installation."""
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: FakeOllama(base_url, timeout_s))


@pytest.fixture
def fake_factory(monkeypatch):
    registry = []

    def install(**kw):
        monkeypatch.setattr(ollama_pilot, "client_factory", lambda base_url, timeout_s: FakeOllama(base_url, timeout_s, registry=registry, **kw))
        return registry

    install()
    return install, registry


def _sim(assignments, seed=3, tmp=None, **kw):
    root = tmp or tempfile.mkdtemp(prefix="olt_")
    return SimulationState(phases=None, project_root=root, flash_mode=True, experiment_name="olt", seed=seed, controller_assignments=assignments,
                           experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "profile_mode": "compact_panel"}, **kw)


def _run(st, seconds):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seconds):
            st.update(1.0)


def _stop(st):
    with contextlib.redirect_stdout(io.StringIO()):
        st.stop()


def _trace_rows(st):
    path = Path(st.logger.output_session.session_folder) / "logs" / "controller_decisions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


OLLAMA_A = {"controller_type": "local_ollama", "settings": {"model": "model-a", "temperature": 0.1, "controller_generation_seed": 11}}
OLLAMA_B = {"controller_type": "local_ollama", "settings": {"model": "model-b", "temperature": 0.9, "controller_generation_seed": 22, "max_output_tokens": 128}}


# ------------------------------------------------------------------ loopback / transport
@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.2:5000"])
def test_loopback_endpoints_are_accepted(url):
    assert validate_loopback_endpoint(url).startswith("http://")


@pytest.mark.parametrize("url", ["http://192.168.1.5:11434", "http://10.0.0.2:11434", "http://example.com", "https://127.0.0.1:11434", "http://localhost.evil.com:11434",
                                 "http://0.0.0.0:11434", "http://user:pw@127.0.0.1:11434", "ftp://127.0.0.1", "127.0.0.1:11434", ""])
def test_non_loopback_or_odd_endpoints_are_rejected(url):
    with pytest.raises(NonLoopbackEndpointError):
        validate_loopback_endpoint(url)
    with pytest.raises(NonLoopbackEndpointError):
        OllamaClient(base_url=url)


def test_spec_rejects_a_remote_endpoint_before_anything_runs():
    with pytest.raises(cs.ControllerConfigError):
        cs.ControllerSpec.from_value({"controller_type": "local_ollama", "settings": {"model": "m", "base_url": "http://203.0.113.5:11434"}})
    with pytest.raises(cs.ControllerConfigError):
        cs.ControllerSpec.from_value({"controller_type": "local_ollama", "settings": {"model": "m", "inference_parameters": {"api_key": "x"}}})


class _StubHandler(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _StubHandler.seen.append(("GET", self.path, dict(self.headers)))
        if self.path == "/api/version":
            self._send({"version": "9.9"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "stub:7b", "size": 1, "digest": "abcdef123456789", "details": {"parameter_size": "7B", "quantization_level": "Q4", "context_length": 4096, "family": "x"}}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        _StubHandler.seen.append(("POST", self.path, payload))
        self._send({"message": {"content": "{\"affordance_id\":\"a0\"}"}, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2})


@pytest.fixture
def stub_server():
    _StubHandler.seen = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _StubHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def test_real_client_discovers_models_chats_and_never_uses_other_endpoints(stub_server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.9:3128")  # an environment proxy must be ignored for loopback traffic
    c = OllamaClient(stub_server, timeout_s=5)
    assert c.status()["available"] and c.status()["version"] == "9.9"
    assert c.installed_model_names() == ["stub:7b"] and c.list_models()[0]["parameter_size"] == "7B"
    assert OllamaClient.model_matches("stub:7b", ["stub:7b"]) == "stub:7b" and OllamaClient.model_matches("stub", ["stub:latest"]) == "stub:latest" and OllamaClient.model_matches("stu", ["stub:7b"]) is None
    reply = c.chat(model="stub:7b", messages=[{"role": "user", "content": "hi"}], options={"temperature": 0.1, "seed": 5, "num_predict": 9}, response_format={"type": "object"}, think=False)
    assert reply["content"].startswith("{") and reply["token_metadata"]["eval_count"] == 2
    post = next(p for m, path, p in _StubHandler.seen if m == "POST")
    assert post["options"]["seed"] == 5 and post["think"] is False and post["stream"] is False and post["format"] == {"type": "object"}
    assert {path for _m, path, _p in _StubHandler.seen} <= {"/api/version", "/api/tags", "/api/chat"}
    assert all("authorization" not in {k.lower() for k in (p if isinstance(p, dict) and m == "GET" else {})} for m, _path, p in _StubHandler.seen)
    with pytest.raises(OllamaRequestError):
        c._request("POST", "/api/pull", {"name": "x"})  # there is no download path
    with pytest.raises(OllamaRequestError):
        c._request("POST", "/api/delete", {"name": "x"})


def test_unreachable_service_reports_unavailable_without_starting_anything():
    st = OllamaClient("http://127.0.0.1:9", timeout_s=1).status()  # nothing listens on the discard port
    assert st["available"] is False and st["error"]


# ------------------------------------------------------------------ preflight (fail before the simulation starts)
def test_unavailable_ollama_fails_before_the_simulation_starts(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, available=False))
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(ollama_pilot.ControllerUnavailableError, match="not running"):
        _sim({"Engineer": OLLAMA_A}, tmp=tmp)


def test_missing_model_fails_clearly_and_never_pulls(monkeypatch):
    clients = []
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, models=("other:1b",), registry=clients))
    with pytest.raises(ollama_pilot.ControllerUnavailableError, match="not installed.*never pulled|never pulled"):
        _sim({"Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-a"}}})
    assert all(not c.chat_calls for c in clients)


def test_headless_condition_fails_clearly_before_stepping(monkeypatch):
    from modules.experiment_runner import run_condition

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, available=False))
    with pytest.raises(ollama_pilot.ControllerUnavailableError):
        run_condition(seed=1, duration_s=5, controller_assignments={"Engineer": OLLAMA_A})


# ------------------------------------------------------------------ single controller-selection path
def test_a_backend_string_cannot_instantiate_the_ollama_provider(monkeypatch, fake_factory):
    from modules import brain_provider as bp

    for alias in cs.LEGACY_OLLAMA_BACKENDS:
        with pytest.raises(bp.LegacyOllamaBackendRetiredError):
            bp.create_brain_provider(bp.BrainBackendConfig(backend=alias))
    monkeypatch.setattr(bp.OllamaLocalBrainProvider, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy Ollama provider constructed")))
    install, registry = fake_factory
    st = _sim({"Engineer": OLLAMA_A, "Botanist": OLLAMA_B})  # the new controller never builds the legacy provider
    assert type(st.get_agent_brain_runtime(next(a for a in st.agents if a.role == "Engineer"))["adapter"]).__name__ == "OllamaPilotAdapter"
    _stop(st)


def test_a_legacy_global_ollama_backend_is_routed_through_controller_specs_not_instantiated(fake_factory):
    install, registry = fake_factory
    install(models=("qwen-test",))
    st = _sim(None, brain_backend="ollama", brain_backend_options={"local_model": "qwen-test"})
    assert all(a.controller_spec.controller_type == "local_ollama" and a.controller_spec.settings["model"] == "qwen-test" for a in st.agents)
    assert all(type(st.get_agent_brain_runtime(a)["provider"]).__name__ == "OllamaControllerProvider" for a in st.agents)
    _stop(st)
    with pytest.raises(cs.ControllerConfigError):  # a non-loopback legacy endpoint is rejected, not honoured
        _sim(None, brain_backend="ollama", brain_backend_options={"local_model": "qwen-test", "local_base_url": "http://203.0.113.5:11434"})


def test_procedural_only_runs_never_contact_ollama(monkeypatch):
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Ollama client created in a procedural-only run")))
    st = _sim({r: "procedural_baseline" for r in ROLES})
    _run(st, 15)
    _stop(st)


# ------------------------------------------------------------------ per-agent independence + mixed composition
def test_each_agent_has_its_own_model_and_settings_and_only_its_own_context(fake_factory):
    install, registry = fake_factory
    st = _sim({"Botanist": OLLAMA_A, "Engineer": OLLAMA_B})
    _run(st, 20)
    by_model = {}
    for c in registry:
        for call in c.chat_calls:
            by_model.setdefault(call["model"], []).append(call)
    assert set(by_model) == {"model-a", "model-b"}
    a, b = by_model["model-a"][0], by_model["model-b"][0]
    assert a["options"]["temperature"] == 0.1 and a["options"]["seed"] == 11 and b["options"]["temperature"] == 0.9 and b["options"]["seed"] == 22 and b["options"]["num_predict"] == 128
    assert all('"role":"Botanist"' in c["messages"][1]["content"] for c in by_model["model-a"]) and all('"role":"Engineer"' in c["messages"][1]["content"] for c in by_model["model-b"])
    arch = next(x for x in st.agents if x.role == "Architect")
    assert type(st.get_agent_brain_runtime(arch)["adapter"]).__name__ == "ProceduralBaselinePilotAdapter"
    man = st.controller_manifest()
    assert man["composition_id"] == "Architect=procedural_baseline|Botanist=local_ollama[controller_generation_seed=11,model=model-a,temperature=0.1]|Engineer=local_ollama[controller_generation_seed=22,max_output_tokens=128,model=model-b,temperature=0.9]"
    assert man["per_agent"]["Botanist"]["ollama"]["model"] == "model-a" and man["per_agent"]["Engineer"]["ollama"]["temperature"] == 0.9
    assert man["per_agent"]["Architect"]["controller"]["controller_type"] == "procedural_baseline" and "ollama" not in man["per_agent"]["Architect"]
    _stop(st)


def test_mixed_team_decisions_are_attributed_to_the_right_controller_and_sim_time_is_independent_of_inference(fake_factory):
    install, registry = fake_factory
    st = _sim({"Engineer": OLLAMA_A})
    seen = {}
    st.logger.register_event_listener(lambda e: seen.setdefault((e["payload_data"].get("agent"), e["payload_data"].get("controller_type")), 0) if e["event_type"] == "pilot_selection_via_adapter" else None)
    _run(st, 25)
    assert st.time == pytest.approx(25.0)  # 25 updates of 1 s: inference wall-clock never adds simulated time
    assert {k[1] for k in seen if k[0] == "Engineer"} == {"local_ollama"} and {k[1] for k in seen if k[0] in {"Architect", "Botanist"}} == {"procedural_baseline"}
    rows = _trace_rows(st)
    decisions = [r for r in rows if r["kind"] == "ollama_decision"]
    assert decisions and {r["role"] for r in decisions} == {"Engineer"} and all(r["controller_type"] == "local_ollama" for r in decisions)
    d = decisions[0]
    assert d["model"] == "model-a" and d["prompt_hash"] and d["system_prompt"] and d["user_prompt"] and d["validation_outcome"] == "valid" and d["retry_count"] == 0
    assert d["wall_clock_inference_s"] >= 0 and d["attempts"][0]["raw_response"] and d["attempts"][0]["token_metadata"]["eval_count"] == 5 and d["budget"]["estimated_prompt_tokens"] > 0
    assert any(r["kind"] == "translation_outcome" and r["role"] == "Engineer" for r in rows)
    _stop(st)


def test_three_ollama_agents_may_share_one_model_with_distinct_settings(fake_factory):
    install, registry = fake_factory
    same = lambda seed, temp: {"controller_type": "local_ollama", "settings": {"model": "model-a", "controller_generation_seed": seed, "temperature": temp}}  # noqa: E731
    st = _sim({"Architect": same(1, 0.1), "Botanist": same(2, 0.5), "Engineer": same(3, 0.9)})
    _run(st, 6)
    seeds = {call["options"]["seed"] for c in registry for call in c.chat_calls}
    assert seeds == {1, 2, 3}
    _stop(st)


# ------------------------------------------------------------------ prompt: canonical only, no hidden/private state
def test_exact_prompt_contains_no_hidden_or_private_state(fake_factory):
    install, registry = fake_factory
    st = _sim({"Botanist": OLLAMA_A})
    arch = next(a for a in st.agents if a.role == "Architect")
    bot = next(a for a in st.agents if a.role == "Botanist")
    held_by_bot = {getattr(i, "id", None) for i in bot.mental_model["information"]}
    private = next(e for e in st.task_model.dik_elements.values() if e.element_type == "information" and str(e.role_scope).lower() == "architect" and e.element_id not in held_by_bot)
    arch.mental_model["information"].add(arch._create_dik_object_from_element(private, source_id="private"))
    pid, _status = st.environment.construction.create_project("site_b", "house")
    st.environment.construction.projects[pid]["correct"] = False  # hidden construction correctness
    m = st.team_knowledge_manager
    m.propose_team_plan(plan_id="px", proposed_by="Architect", sim_time=st.time, goal_ids=["g"], goal_summary="g", project_targets=["SECRET_PLAN_TARGET"],
                        assignments_by_role={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "SECRET_PLAN_TARGET"}}, supporters=["Architect"], expires_at=500.0, review_at=60.0)
    _run(st, 3)
    prompts = [c["messages"][0]["content"] + c["messages"][1]["content"] for cl in registry for c in cl.chat_calls]
    assert prompts
    for text in prompts:
        for forbidden in ('"correct"', "mental_model", "TeamKnowledgeManager", "team_knowledge_manager", "SECRET_PLAN_TARGET", "Outputs", "session_folder", "C:\\\\", "logs/",
                          '"environment"', "sim_state", "<modules", "object at 0x", private.element_id):
            assert forbidden not in text, forbidden
    ctx_bot = build_canonical_context(st.brain_context_builder.build(st, bot), agent_id=bot.agent_id, role=bot.role, controller_type="local_ollama")
    assert private.element_id not in json.dumps(ctx_bot, default=str)
    _stop(st)


def test_plan_context_in_the_prompt_follows_legitimate_acquisition(fake_factory):
    install, registry = fake_factory
    st = _sim({"Botanist": OLLAMA_A})
    bot = next(a for a in st.agents if a.role == "Botanist")
    m = st.team_knowledge_manager
    m.propose_team_plan(plan_id="px", proposed_by="Architect", sim_time=st.time, goal_ids=["g"], goal_summary="the-plan-goal-text", project_targets=["t"],
                        assignments_by_role={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "t"}}, supporters=["Architect"], expires_at=500.0, review_at=60.0)
    canon = build_canonical_context(st.brain_context_builder.build(st, bot), agent_id=bot.agent_id, role="Botanist", controller_type="local_ollama")
    before = build_prompt(canon, budget_tokens=6000)["user"]
    assert "the-plan-goal-text" not in before and '"unread_plan_artifact_on_workspace":true' in before
    bot._note_known_plan(st, m._team_plan_summary(dict(m.artifacts["team_plan:px"].content)), via="message")
    canon = build_canonical_context(st.brain_context_builder.build(st, bot), agent_id=bot.agent_id, role="Botanist", controller_type="local_ollama")
    after = build_prompt(canon, budget_tokens=6000)["user"]
    assert "the-plan-goal-text" in after and '"read_via":"message"' in after
    _stop(st)


# ------------------------------------------------------------------ prompt budgeting
def _canon(st, role="Engineer"):
    ag = next(a for a in st.agents if a.role == role)
    return build_canonical_context(st.brain_context_builder.build(st, ag), agent_id=ag.agent_id, role=role, controller_type="local_ollama")


def test_prompt_budget_is_deterministic_tiered_and_never_drops_affordances(fake_factory):
    st = _sim({"Engineer": OLLAMA_A})
    _run(st, 40)
    canon = _canon(st)
    # make the history large so a small budget must omit lower tiers
    canon["recent_outcomes"]["recent_action_outcomes"] = [{"action": "x", "i": i, "note": "n" * 80} for i in range(80)]
    canon["recent_outcomes"]["near_preceding_events"] = [f"event number {i} " + "e" * 60 for i in range(80)]
    canon["held_dik"]["information"] = [f"I_FAKE_{i}" for i in range(200)]
    small, again, large = build_prompt(canon, budget_tokens=1500), build_prompt(canon, budget_tokens=1500), build_prompt(canon, budget_tokens=40000)
    assert small["prompt_hash"] == again["prompt_hash"] and small["metadata"] == again["metadata"]
    assert len(small["affordance_map"]) == len(canon["executable_actions"]) == small["metadata"]["sections"]["executable_actions"]["included_items"]  # affordances are never traded for history
    assert small["metadata"]["omitted_item_counts"] and small["metadata"]["estimated_prompt_tokens"] < large["metadata"]["estimated_prompt_tokens"]
    om = small["metadata"]["sections"]
    assert om["recent_outcomes.older_events"]["omitted_items"] >= om["recent_outcomes.actions"]["omitted_items"] or om["recent_outcomes.older_events"]["included_items"] <= om["recent_outcomes.actions"]["included_items"]
    assert not large["metadata"]["omitted_item_counts"]
    body = json.loads(small["user"].split("CONTEXT:\n", 1)[1].split("\n\nChoose", 1)[0])  # complete, valid JSON: never cut off at the end
    assert {"identity", "goals", "observations", "known_gaps", "team_plan", "executable_actions"} <= set(body)
    tiny = build_prompt(canon, budget_tokens=200)
    assert tiny["metadata"]["tier1_over_budget"] is True and len(tiny["affordance_map"]) == len(canon["executable_actions"])
    _stop(st)


# ------------------------------------------------------------------ structured choice, validation, retries, failure
def _adapter(reply, **settings):
    s = cs.ControllerSpec.from_value({"controller_type": "local_ollama", "settings": {"model": "model-a", **settings}}).full_settings()
    return ollama_pilot.OllamaPilotAdapter(s, agent_id="engineer_1", role="Engineer")


@pytest.fixture
def request_obj():
    st = _sim({"Engineer": OLLAMA_A})
    ag = next(a for a in st.agents if a.role == "Engineer")
    ctx = st.brain_context_builder.build(st, ag)
    req = ag._build_brain_request(st, ctx, False, "t", request_id="r1")
    yield st, req, ctx
    _stop(st)


def test_valid_affordance_choice_becomes_the_exact_offered_action(monkeypatch, request_obj):
    st, req, ctx = request_obj
    typed = next(a for a in ctx.action_affordances if a.get("action_args"))
    idx = ctx.action_affordances.index(typed)
    ad = _adapter(None)
    monkeypatch.setattr(ad, "client", FakeOllama(reply=lambda m, o, mod: json.dumps({"affordance_id": f"a{idx}", "arguments": dict(typed["action_args"]), "decision_note": "typed start"})))
    resp = ad.choose_action(req)
    step = resp.plan.next_action
    assert step.action_type.value == typed["action_type"] and step.target_id == typed.get("target_id") and dict(step.action_args) == dict(typed["action_args"])
    assert ad.last_trace["validation_outcome"] == "valid" and ad.last_trace["retry_count"] == 0


@pytest.mark.parametrize("bad", ['{"affordance_id": "a999"}', '{"affordance_id": "start_construction"}', '{"affordance_id": "a0", "arguments": {"structure_type": "castle"}}',
                                 'not json at all', '[1,2]', '{"arguments": {}}', '{"affordance_id": "a0", "arguments": 5}'])
def test_invented_illegal_or_malformed_choices_are_rejected_and_retried_within_the_bound(monkeypatch, request_obj, bad):
    st, req, ctx = request_obj
    ad = _adapter(None, max_retries=2)
    calls = []
    good = json.dumps({"affordance_id": "a0"})
    monkeypatch.setattr(ad, "client", FakeOllama(reply=lambda m, o, mod: (calls.append(m), bad if len(calls) == 1 else good)[1]))
    resp = ad.choose_action(req)
    assert ad.last_trace["retry_count"] == 1 and ad.last_trace["attempts"][0]["error"] and resp.plan.next_action.action_type.value == ctx.action_affordances[0]["action_type"]
    assert "rejected" in calls[1][-1]["content"] and "currently executable ids" in calls[1][-1]["content"]  # a concise correction naming the legal affordances


def test_retry_exhaustion_raises_a_controller_failure_and_never_consults_the_baseline(monkeypatch, request_obj):
    from modules import brain_provider as bp

    st, req, ctx = request_obj
    monkeypatch.setattr(bp.ProceduralBaselinePilot, "generate_plan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("baseline consulted")))
    ad = _adapter(None, max_retries=2)
    fake = FakeOllama(reply=lambda m, o, mod: "garbage")
    monkeypatch.setattr(ad, "client", fake)
    with pytest.raises(ollama_pilot.ControllerFailure) as err:
        ad.choose_action(req)
    assert len(fake.chat_calls) == 3 and err.value.trace["retry_count"] == 2 and err.value.trace["attempts"][-1]["raw_response"] == "garbage"
    assert ad.choose_fallback_action(agent=None, context_packet=None, reason="x") is None and ad.handle_blocked_action() is None


def test_inference_errors_are_bounded_too_and_not_replaced(monkeypatch, request_obj):
    from modules.ollama_client import OllamaUnavailableError

    st, req, ctx = request_obj
    ad = _adapter(None, max_retries=1)
    n = {"c": 0}

    def boom(**kw):
        n["c"] += 1
        raise OllamaUnavailableError("down")

    ad.client = type("C", (), {"chat": staticmethod(boom)})()
    with pytest.raises(ollama_pilot.ControllerFailure):
        ad.choose_action(req)
    assert n["c"] == 2 and ad.last_trace["validation_outcome"] == "inference_error"


def test_run_halts_with_an_explicit_controller_error_and_no_silent_fallback(monkeypatch):
    from modules import brain_provider as bp

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, models=("model-a",), reply=lambda m, o, mod: "garbage"))
    st = _sim({"Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-a", "max_retries": 1}}})
    eng = next(a for a in st.agents if a.role == "Engineer")
    real = bp.ProceduralBaselinePilot.generate_plan
    consulted_for = []

    def spy(self, request_packet):
        consulted_for.append(request_packet.agent_id)
        return real(self, request_packet)

    monkeypatch.setattr(bp.ProceduralBaselinePilot, "generate_plan", spy)
    _run(st, 10)
    assert st.controller_error and st.controller_error["role"] == "Engineer" and st.terminal_state == "controller_error"
    t_halt = st.time
    _run(st, 5)
    assert st.time == t_halt  # the run stopped advancing
    assert eng.agent_id not in consulted_for  # the Procedural Baseline never decided for the failed agent
    rows = _trace_rows(st)
    assert any(r.get("controller_failure") and r["role"] == "Engineer" and r["retry_count"] == 1 for r in rows)
    assert st.controller_manifest()["controller_error"]["agent"] == eng.name
    _stop(st)


def test_idle_and_log_policy_is_explicit_and_still_never_uses_another_controller(monkeypatch):
    from modules import brain_provider as bp

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, models=("model-a",), reply=lambda m, o, mod: "garbage"))
    st = _sim({"Engineer": {"controller_type": "local_ollama", "settings": {"model": "model-a", "max_retries": 0, "failure_policy": "idle_and_log"}}})
    failures = []
    st.logger.register_event_listener(lambda e: failures.append(e) if e["event_type"] == "controller_failure" else None)
    _run(st, 6)
    assert st.controller_error is None and st.time == pytest.approx(6.0) and failures and failures[0]["payload_data"]["failure_policy"] == "idle_and_log"
    _stop(st)


# ------------------------------------------------------------------ GUI
@pytest.fixture
def app(monkeypatch):
    pytest.importorskip("tkinter")
    import interface

    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, models=("model-a", "model-b", "model-c")))
    try:
        a = interface.MarsColonyInterface()
    except Exception as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass


def _set(app, role, label, **fields):
    app.agent_controller_vars[role].set(label)
    for k, v in fields.items():
        app.agent_ollama_vars[role][k].set(str(v))


def test_gui_per_agent_ollama_controls_are_independent_and_default_is_all_procedural(app):
    roles = app.agent_card_order[:3]
    assert {app.agent_controller_vars[r].get() for r in roles} == {"Procedural Baseline"}
    _set(app, roles[1], "Local Ollama", model="model-a", temperature="0.3", controller_generation_seed="7")
    assert "installed" in app.agent_ollama_vars[roles[1]] and app.agent_ollama_vars[roles[1]]["installed"] == ["model-a", "model-b", "model-c"]
    assert "connected" in app.agent_ollama_vars[roles[1]]["status"].get()
    a = app.controller_assignments()
    assert a[roles[0]] == {"controller_type": "procedural_baseline", "settings": {}} and a[roles[2]] == {"controller_type": "procedural_baseline", "settings": {}}
    assert a[roles[1]]["controller_type"] == "local_ollama" and a[roles[1]]["settings"]["model"] == "model-a" and a[roles[1]]["settings"]["controller_generation_seed"] == 7
    assert "Local Ollama[model-a]" in app.test_drive_summary_var.get()
    app.reset_all_controllers()
    assert {app.controller_assignments()[r]["controller_type"] for r in roles} == {"procedural_baseline"}


@pytest.mark.parametrize("plan", [
    ("Procedural Baseline", "Procedural Baseline", "model-a"),
    ("model-a", "Procedural Baseline", "model-b"),
    ("model-a", "model-a", "model-a"),
    ("model-a", "model-b", "model-c"),
])
def test_gui_supports_the_required_team_compositions(app, plan):
    roles = app.agent_card_order[:3]
    for role, choice in zip(roles, plan):
        if choice == "Procedural Baseline":
            _set(app, role, "Procedural Baseline")
        else:
            _set(app, role, "Local Ollama", model=choice)
    st = app._collect_experiment_settings()
    got = [(st["controller_assignments"][r]["controller_type"], st["controller_assignments"][r]["settings"].get("model")) for r in roles]
    want = [("procedural_baseline", None) if c == "Procedural Baseline" else ("local_ollama", c) for c in plan]
    assert got == want
    assert [cfg["controller"]["controller_type"] for cfg in st["agent_configs"]] == [w[0] for w in want]


def test_gui_start_is_blocked_when_ollama_or_the_model_is_unusable(app, monkeypatch):
    roles = app.agent_card_order[:3]
    _set(app, roles[2], "Local Ollama", model="not-installed")
    assert "not installed" in app._controller_preflight_error()
    monkeypatch.setattr(ollama_pilot, "client_factory", lambda b, t: FakeOllama(b, t, available=False))
    _set(app, roles[2], "Local Ollama", model="model-a")
    assert "not running" in app._controller_preflight_error()
    shown = []
    import interface

    # patch the module object the GUI actually calls (another test may have re-imported interface, leaving tkinter.messagebox a different object)
    monkeypatch.setattr(interface.messagebox, "showerror", lambda *a, **k: shown.append(a))
    app.start_experiment()
    assert shown and app.sim is None and app.run_state != app.STATE_STARTING
    _set(app, roles[2], "Procedural Baseline")
    assert app._controller_preflight_error() is None


def test_gui_offers_no_cloud_or_legacy_provider_controls(app):
    labels = set()

    def walk(w):
        for c in w.winfo_children():
            try:
                labels.add(str(c.cget("text")))
            except Exception:
                pass
            try:
                labels.update(str(v) for v in c.cget("values"))
            except Exception:
                pass
            walk(c)

    walk(app.root)
    blob = " ".join(labels).lower()
    for banned in ("openai", "anthropic", "claude", "gpt", "api key", "local_http", "ollama_local"):
        assert banned not in blob
    assert "local ollama" in blob


def test_a_translation_time_legality_failure_is_reported_back_to_the_generative_pilot_only(fake_factory):
    install, registry = fake_factory
    st = _sim({"Engineer": OLLAMA_A})
    eng = next(a for a in st.agents if a.role == "Engineer")
    arch = next(a for a in st.agents if a.role == "Architect")
    payload = {"planner_action_type": "repair_or_correct_construction", "failure_category": "readiness_not_unlocked", "project_id": "p1", "blockers": ["repair_project_rules_not_held"]}
    eng._emit_event(st, "action_legality_failed", payload)
    arch._emit_event(st, "action_legality_failed", payload)
    seen = [o for o in eng.recent_action_outcomes if o["blocker"] == "readiness_not_unlocked"]
    assert seen and seen[0]["status"] == "blocked" and seen[0]["action"] == "repair_or_correct_construction" and seen[0]["target"] == "p1"
    assert not getattr(arch, "recent_action_outcomes", [])  # the Procedural Baseline's feedback is unchanged
    _stop(st)
