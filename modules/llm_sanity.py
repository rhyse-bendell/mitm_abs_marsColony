from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request

LOCAL_BACKEND_ALIASES = {"local_http", "openai_compatible_local", "ollama_local", "ollama"}

SANITY_RESPONSE_FIELDS = [
    "agent_name",
    "role_or_focus",
    "understood_mission",
    "relevant_data_ids",
    "relevant_information_ids",
    "relevant_knowledge_or_rule_ids",
    "first_information_priority",
    "first_coordination_need",
    "confidence",
]


@dataclass(frozen=True)
class StartupLLMSanityConfig:
    enabled: bool = False
    timeout_s: float = 8.0
    max_sources: int = 2
    max_items_per_type: int = 3
    raw_response_max_chars: int = 4000
    artifact_name: str = "startup_llm_sanity.json"


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response")
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_sanity_response_schema(payload: Dict[str, Any], *, expected_agent_name: str | None = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["response is not a JSON object"]

    missing = [k for k in SANITY_RESPONSE_FIELDS if k not in payload]
    if missing:
        errors.append(f"missing fields: {missing}")

    for key in ["relevant_data_ids", "relevant_information_ids", "relevant_knowledge_or_rule_ids"]:
        value = payload.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{key} must be a list")

    for key in ["agent_name", "role_or_focus", "understood_mission", "first_information_priority", "first_coordination_need"]:
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"{key} must be a string")

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            conf = float(confidence)
            if conf < 0.0 or conf > 1.0:
                errors.append("confidence must be in [0.0, 1.0]")
        except (TypeError, ValueError):
            errors.append("confidence must be numeric")

    if expected_agent_name and isinstance(payload.get("agent_name"), str):
        if payload.get("agent_name").strip().lower() != expected_agent_name.strip().lower():
            errors.append("agent_name does not match probe target")

    return len(errors) == 0, errors


def build_startup_sanity_prompt(agent, task_model, *, max_sources: int = 2, max_items_per_type: int = 3) -> Dict[str, Any]:
    role = getattr(agent, "role", "Agent")
    display_name = getattr(agent, "display_name", getattr(agent, "name", role))
    source_ids = task_model.source_ids_for_role(role)[: max(1, int(max_sources))]

    data_items: List[Dict[str, str]] = []
    info_items: List[Dict[str, str]] = []
    knowledge_items: List[Dict[str, str]] = []

    for source_id in source_ids:
        for element_type, bucket in (("data", data_items), ("information", info_items), ("knowledge", knowledge_items)):
            elements = task_model.elements_for_source(source_id, element_type=element_type)
            for element in elements:
                if len(bucket) >= max_items_per_type:
                    break
                bucket.append({"id": element.element_id, "label": element.label, "source_id": source_id})

    role_rules = [
        {
            "id": r.rule_id,
            "label": r.label,
        }
        for r in task_model.rules.values()
        if r.enabled and ((not r.role_scope) or r.role_scope.strip().lower() in {"", "all", role.lower()})
    ][: max_items_per_type]

    mission = f"{task_model.manifest.get('name', task_model.task_id)}: {task_model.manifest.get('description', '')}".strip()

    prompt_contract = {
        "agent_identity": {
            "agent_name": getattr(agent, "name", display_name),
            "display_name": display_name,
            "role": role,
            "label": getattr(agent, "agent_label", role),
        },
        "mission_context": {
            "task_id": task_model.task_id,
            "mission_summary": mission,
            "objective": "Provide a startup sanity check response only; do not provide a full execution plan.",
        },
        "dik_framing": {
            "data": "raw factual units",
            "information": "contextualized/combined data",
            "knowledge": "structured understanding and actionable rules",
        },
        "bounded_context": {
            "source_ids": source_ids,
            "data_examples": data_items[:max_items_per_type],
            "information_examples": info_items[:max_items_per_type],
            "knowledge_examples": knowledge_items[:max_items_per_type],
            "role_rule_examples": role_rules,
        },
        "response_schema": {field: "required" for field in SANITY_RESPONSE_FIELDS},
        "constraints": [
            "Return only a single JSON object.",
            "Keep understood_mission to <= 24 words.",
            "Keep first_information_priority and first_coordination_need concise.",
            "Do not include a full construction solution.",
        ],
    }

    return {
        "prompt_contract": prompt_contract,
        "prompt_text": json.dumps(prompt_contract, indent=2),
    }


def _post_chat_completion(*, endpoint: str, model: str, prompt_text: str, timeout_s: float) -> Tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON matching the requested schema."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 360,
    }
    started = time.perf_counter()
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return raw, latency_ms


def run_startup_llm_sanity_check(simulation, *, config: StartupLLMSanityConfig) -> Dict[str, Any]:
    timestamp = int(time.time() * 1000)
    results: List[Dict[str, Any]] = []
    simulation.logger.log_event(simulation.time, "startup_llm_sanity_started", {"agent_count": len(simulation.agents), "enabled": True})

    for agent in simulation.agents:
        runtime = simulation.get_agent_brain_runtime(agent)
        runtime_config = runtime["config"]
        backend = str(runtime.get("configured_backend", runtime_config.backend)).lower()
        prompt = build_startup_sanity_prompt(
            agent,
            simulation.task_model,
            max_sources=config.max_sources,
            max_items_per_type=config.max_items_per_type,
        )
        endpoint = f"{runtime_config.local_base_url.rstrip('/')}{runtime_config.local_endpoint}"
        row: Dict[str, Any] = {
            "agent_id": getattr(agent, "agent_id", agent.name),
            "agent_name": agent.name,
            "display_name": agent.display_name,
            "role": agent.role,
            "configured_backend": runtime.get("configured_backend"),
            "effective_backend": runtime.get("effective_backend"),
            "provider_class": runtime.get("provider").__class__.__name__,
            "model": runtime_config.local_model,
            "endpoint": endpoint if backend in LOCAL_BACKEND_ALIASES else None,
            "request_sent": False,
            "response_received": False,
            "parsed_success": False,
            "validation_success": False,
            "timeout": False,
            "fallback_used": False,
            "latency_ms": None,
            "prompt_size_chars": len(prompt["prompt_text"]),
            "prompt_summary": {
                "source_count": len(prompt["prompt_contract"]["bounded_context"]["source_ids"]),
                "data_count": len(prompt["prompt_contract"]["bounded_context"]["data_examples"]),
                "information_count": len(prompt["prompt_contract"]["bounded_context"]["information_examples"]),
                "knowledge_count": len(prompt["prompt_contract"]["bounded_context"]["knowledge_examples"]),
                "rule_count": len(prompt["prompt_contract"]["bounded_context"]["role_rule_examples"]),
            },
            "raw_response_text": None,
            "parsed_response": None,
            "validation_errors": [],
            "error": None,
        }

        if backend not in LOCAL_BACKEND_ALIASES:
            row["error"] = f"backend_not_local:{backend}"
            results.append(row)
            continue

        try:
            row["request_sent"] = True
            raw_text, latency_ms = _post_chat_completion(
                endpoint=endpoint,
                model=runtime_config.local_model,
                prompt_text=prompt["prompt_text"],
                timeout_s=min(float(config.timeout_s), float(runtime_config.timeout_s)),
            )
            row["latency_ms"] = latency_ms
            row["response_received"] = bool(raw_text)
            row["raw_response_text"] = raw_text[: config.raw_response_max_chars]

            parsed_http = json.loads(raw_text)
            choices = parsed_http.get("choices") or []
            message = (choices[0] if choices else {}).get("message", {})
            content = message.get("content")
            if isinstance(content, dict):
                parsed_payload = content
            else:
                parsed_payload = _extract_json_object(str(content or ""))
            row["parsed_success"] = True
            row["parsed_response"] = parsed_payload

            valid, errors = validate_sanity_response_schema(parsed_payload, expected_agent_name=agent.name)
            row["validation_success"] = valid
            row["validation_errors"] = errors
        except TimeoutError as exc:
            row["timeout"] = True
            row["error"] = f"TimeoutError: {exc}"
        except (error.URLError, error.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"

        provider = runtime.get("provider")
        if hasattr(provider, "last_outcome") and isinstance(provider.last_outcome, dict):
            row["fallback_used"] = bool(provider.last_outcome.get("fallback", False))

        results.append(row)

    summary = {
        "startup_llm_sanity_enabled": True,
        "startup_llm_sanity_started_epoch_ms": timestamp,
        "startup_llm_sanity_agent_count": len(results),
        "startup_llm_sanity_success_count": sum(1 for r in results if r.get("validation_success")),
        "startup_llm_sanity_failure_count": sum(1 for r in results if not r.get("validation_success")),
        "startup_llm_sanity_timeout_count": sum(1 for r in results if r.get("timeout")),
        "startup_llm_sanity_parse_failure_count": sum(1 for r in results if r.get("response_received") and not r.get("parsed_success")),
    }

    artifact_payload = {
        "summary": summary,
        "schema_fields": SANITY_RESPONSE_FIELDS,
        "results": results,
    }

    artifact_path = Path(simulation.logger.output_session.build_log_path(config.artifact_name))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    simulation.logger.log_event(
        simulation.time,
        "startup_llm_sanity_completed",
        {**summary, "artifact": str(artifact_path)},
    )

    return {
        **summary,
        "startup_llm_sanity_artifact": f"logs/{config.artifact_name}",
        "startup_llm_sanity_results": results,
    }
