from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
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
    timeout_s: float = 45.0
    max_sources: int = 2
    max_items_per_type: int = 3
    completion_max_tokens: int = 1024
    json_only_mode: bool = True
    reasoning_suppression: bool = True
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


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block.get("text", ""))
                elif isinstance(block.get("content"), str):
                    parts.append(block.get("content", ""))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content.get("text", "")
        if isinstance(content.get("content"), str):
            return content.get("content", "")
    return ""


def _extract_payload_from_wrapper(response_json: Dict[str, Any]) -> Dict[str, Any]:
    choices = response_json.get("choices") or []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    finish_reason = choice0.get("finish_reason")
    reasoning = message.get("reasoning") or message.get("thinking") or choice0.get("reasoning")
    reasoning_text = _content_text(reasoning)

    wrapper_summary = {
        "has_choices": bool(choices),
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "content_type": type(message.get("content")).__name__ if message else None,
        "has_reasoning": bool(reasoning_text.strip()),
    }

    content = message.get("content")
    if isinstance(content, dict):
        return {
            "payload": content,
            "candidate_text": json.dumps(content),
            "candidate_source": "choices[0].message.content(dict)",
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "wrapper_summary": wrapper_summary,
            "reasoning_text": reasoning_text,
            "json_found": True,
        }

    candidate_texts: List[Tuple[str, str]] = []
    for source, value in [
        ("choices[0].message.content", content),
        ("choices[0].message.output_text", message.get("output_text")),
        ("choices[0].text", choice0.get("text")),
        ("response", response_json.get("response")),
    ]:
        extracted = _content_text(value).strip()
        if extracted:
            candidate_texts.append((source, extracted))

    candidate_source = candidate_texts[0][0] if candidate_texts else None
    candidate_text = candidate_texts[0][1] if candidate_texts else ""
    json_found = bool(re.search(r"\{.*\}", candidate_text, flags=re.DOTALL)) if candidate_text else False

    if not candidate_text and reasoning_text.strip():
        try:
            recovered_payload = _extract_json_object(reasoning_text)
            return {
                "payload": recovered_payload,
                "candidate_text": reasoning_text,
                "candidate_source": "choices[0].message.reasoning(recovered_json)",
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "wrapper_summary": wrapper_summary,
                "reasoning_text": reasoning_text,
                "json_found": True,
                "recovered_from_reasoning": True,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    failure_category = None
    if not candidate_text:
        failure_category = "reasoning_only_response" if reasoning_text.strip() else "empty_content_wrapper"
    elif not json_found:
        failure_category = "truncated_response" if finish_reason == "length" else "json_not_found_in_response"

    return {
        "payload": None,
        "candidate_text": candidate_text,
        "candidate_source": candidate_source,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "wrapper_summary": wrapper_summary,
        "reasoning_text": reasoning_text,
        "json_found": json_found,
        "failure_category": failure_category,
    }

def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # tolerate comma/newline/semicolon separated responses
        parts = re.split(r"[,;\n]+", text)
        return [p.strip() for p in parts if p and p.strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        conf = float(value)
        if conf > 1.0 and conf <= 100.0:
            conf = conf / 100.0
        return min(1.0, max(0.0, conf))

    text = str(value).strip().lower()
    if not text:
        return 0.5

    aliases = {
        "high": 0.85,
        "medium": 0.55,
        "med": 0.55,
        "moderate": 0.55,
        "low": 0.25,
        "very high": 0.95,
        "very low": 0.10,
    }
    if text in aliases:
        return aliases[text]

    if text.endswith("%"):
        try:
            pct = float(text[:-1].strip())
            return min(1.0, max(0.0, pct / 100.0))
        except ValueError:
            return 0.5

    try:
        conf = float(text)
        if conf > 1.0 and conf <= 100.0:
            conf = conf / 100.0
        return min(1.0, max(0.0, conf))
    except ValueError:
        return 0.5


def normalize_sanity_response_payload(
    payload: Dict[str, Any],
    *,
    expected_agent_name: str | None = None,
    fallback_role: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    normalized = dict(payload)

    if not isinstance(normalized.get("agent_name"), str) or not normalized.get("agent_name", "").strip():
        normalized["agent_name"] = expected_agent_name or ""

    if not isinstance(normalized.get("role_or_focus"), str):
        normalized["role_or_focus"] = str(normalized.get("role_or_focus", "") or "")
    normalized["role_or_focus"] = normalized["role_or_focus"].strip() or (fallback_role or "")

    for key in ["understood_mission", "first_information_priority", "first_coordination_need"]:
        if not isinstance(normalized.get(key), str):
            normalized[key] = str(normalized.get(key, "") or "")
        normalized[key] = normalized[key].strip()

    normalized["relevant_data_ids"] = _coerce_string_list(normalized.get("relevant_data_ids"))
    normalized["relevant_information_ids"] = _coerce_string_list(normalized.get("relevant_information_ids"))
    normalized["relevant_knowledge_or_rule_ids"] = _coerce_string_list(normalized.get("relevant_knowledge_or_rule_ids"))
    normalized["confidence"] = _coerce_confidence(normalized.get("confidence"))

    return normalized

def validate_sanity_response_schema(payload: Dict[str, Any], *, expected_agent_name: str | None = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["response is not a JSON object"]

    # After normalization, these should all exist. We only fail on fields that
    # are still unusable for a basic startup sanity pass.
    required_string_fields = [
        "role_or_focus",
        "understood_mission",
        "first_information_priority",
        "first_coordination_need",
    ]
    for key in required_string_fields:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")

    for key in ["relevant_data_ids", "relevant_information_ids", "relevant_knowledge_or_rule_ids"]:
        value = payload.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{key} must be a list")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        errors.append("confidence must be numeric")
    else:
        conf = float(confidence)
        if conf < 0.0 or conf > 1.0:
            errors.append("confidence must be in [0.0, 1.0]")

    # Do not fail startup sanity just because the model omitted or slightly altered
    # agent_name. That field is useful metadata, not mission-critical comprehension.
    agent_name = payload.get("agent_name")
    if agent_name is not None and not isinstance(agent_name, str):
        errors.append("agent_name must be a string if present")

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
            "mission_summary": mission[:220],
            "objective": "Startup sanity probe only. Provide quick role-grounded comprehension.",
        },
        "bounded_context": {
            "source_ids": source_ids[: max(1, int(max_sources))],
            "example_data_ids": [item["id"] for item in data_items[:max_items_per_type]],
            "example_information_ids": [item["id"] for item in info_items[:max_items_per_type]],
            "example_knowledge_or_rule_ids": (
                [item["id"] for item in knowledge_items[:max_items_per_type]]
                + [item["id"] for item in role_rules[:max_items_per_type]]
            )[: max_items_per_type],
        },
        "response_schema": {field: "required" for field in SANITY_RESPONSE_FIELDS},
        "constraints": [
            "Return exactly one JSON object with only the schema keys.",
            "No markdown, no analysis, no chain-of-thought, no hidden/visible reasoning.",
            "Use short strings; keep understood_mission <= 20 words.",
            "Be action-oriented: set first_information_priority and first_coordination_need as immediate next focus.",
        ],
    }

    compact_prompt = {
        "agent_name": prompt_contract["agent_identity"]["agent_name"],
        "role": prompt_contract["agent_identity"]["role"],
        "mission_summary": prompt_contract["mission_context"]["mission_summary"],
        "source_ids": prompt_contract["bounded_context"]["source_ids"],
        "example_data_ids": prompt_contract["bounded_context"]["example_data_ids"],
        "example_information_ids": prompt_contract["bounded_context"]["example_information_ids"],
        "example_knowledge_or_rule_ids": prompt_contract["bounded_context"]["example_knowledge_or_rule_ids"],
        "required_response_keys": SANITY_RESPONSE_FIELDS,
        "instructions": prompt_contract["constraints"],
    }

    return {
        "prompt_contract": prompt_contract,
        "prompt_text": json.dumps(compact_prompt, separators=(",", ":")),
    }


def _post_chat_completion(
    *,
    endpoint: str,
    model: str,
    prompt_text: str,
    timeout_s: float,
    completion_max_tokens: int,
    json_only_mode: bool,
    reasoning_suppression: bool,
) -> Tuple[str, float]:
    system_instruction = "Return exactly one JSON object matching the requested schema."
    if reasoning_suppression:
        system_instruction += " Do not include analysis, reasoning, or chain-of-thought."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0,
        "max_tokens": int(completion_max_tokens),
    }
    if json_only_mode:
        payload["response_format"] = {"type": "json_object"}
    # Keep startup call standards-compliant for OpenAI-compatible local endpoints.
    # We intentionally rely on strict prompt constraints for reasoning suppression,
    # because cross-endpoint "disable thinking" controls are not reliably portable.
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


def _build_bootstrap_summary(parsed_payload: Dict[str, Any], *, max_chars: int) -> Tuple[str, Dict[str, Any]]:
    summary_structured = {
        "role_or_focus": str(parsed_payload.get("role_or_focus", "")).strip(),
        "understood_mission": str(parsed_payload.get("understood_mission", "")).strip(),
        "first_information_priority": str(parsed_payload.get("first_information_priority", "")).strip(),
        "first_coordination_need": str(parsed_payload.get("first_coordination_need", "")).strip(),
        "confidence": parsed_payload.get("confidence"),
        "relevant_data_ids": list(parsed_payload.get("relevant_data_ids", []))[:3],
        "relevant_information_ids": list(parsed_payload.get("relevant_information_ids", []))[:3],
        "relevant_knowledge_or_rule_ids": list(parsed_payload.get("relevant_knowledge_or_rule_ids", []))[:3],
    }
    summary_text = (
        f"focus={summary_structured['role_or_focus']}; "
        f"mission={summary_structured['understood_mission']}; "
        f"priority={summary_structured['first_information_priority']}; "
        f"coordination={summary_structured['first_coordination_need']}; "
        f"confidence={summary_structured['confidence']}"
    ).strip()
    if len(summary_text) > max_chars:
        summary_text = summary_text[:max_chars].rstrip()
    return summary_text, summary_structured


def _emit_progress(
    progress_callback: Callable[[str, int, int, str | None, str | None], None] | None,
    *,
    stage: str,
    current: int,
    total: int,
    agent_name: str | None = None,
    detail: str | None = None,
) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback(stage, current, total, agent_name, detail)
    except Exception:
        # Keep startup semantics unchanged if callback consumer fails.
        return


def run_startup_llm_sanity_check(
    simulation,
    *,
    config: StartupLLMSanityConfig,
    progress_callback: Callable[[str, int, int, str | None, str | None], None] | None = None,
) -> Dict[str, Any]:
    timestamp = int(time.time() * 1000)
    results: List[Dict[str, Any]] = []
    total_agents = len(simulation.agents)
    bootstrap_reuse_enabled = bool(getattr(simulation, "bootstrap_reuse_enabled", True))
    bootstrap_summary_max_chars = int(getattr(simulation, "bootstrap_summary_max_chars", 280))
    simulation.logger.log_event(simulation.time, "startup_llm_sanity_started", {"agent_count": len(simulation.agents), "enabled": True, "bootstrap_reuse_enabled": bootstrap_reuse_enabled})
    _emit_progress(
        progress_callback,
        stage="startup_begin",
        current=0,
        total=total_agents,
        detail="Startup LLM sanity check started.",
    )

    for idx, agent in enumerate(simulation.agents, start=1):
        _emit_progress(
            progress_callback,
            stage="agent_begin",
            current=idx - 1,
            total=total_agents,
            agent_name=getattr(agent, "name", None),
            detail=f"Running startup sanity for {getattr(agent, 'name', 'agent')}.",
        )
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
        runtime_bootstrap = runtime.get("bootstrap") if isinstance(runtime, dict) else None
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
                "data_count": len(prompt["prompt_contract"]["bounded_context"]["example_data_ids"]),
                "information_count": len(prompt["prompt_contract"]["bounded_context"]["example_information_ids"]),
                "knowledge_count": len(prompt["prompt_contract"]["bounded_context"]["example_knowledge_or_rule_ids"]),
                "rule_count": 0,
            },
            "startup_request_config": {
                "timeout_s": float(config.timeout_s),
                "completion_max_tokens": int(config.completion_max_tokens),
                "json_only_mode_requested": bool(config.json_only_mode),
                "reasoning_suppression_requested": bool(config.reasoning_suppression),
                "unrestricted_local_qwen_mode": bool(getattr(runtime_config, "unrestricted_local_qwen_mode", False)),
            },
            "raw_response_text": None,
            "parsed_response": None,
            "response_wrapper_summary": None,
            "extracted_candidate_text": None,
            "extracted_candidate_source": None,
            "json_found_in_response": False,
            "json_parsed": False,
            "schema_validated": False,
            "failure_category": None,
            "finish_reason": None,
            "truncated": False,
            "validation_errors": [],
            "error": None,
        }

        if backend not in LOCAL_BACKEND_ALIASES:
            row["error"] = f"backend_not_local:{backend}"
            if isinstance(runtime_bootstrap, dict):
                runtime_bootstrap.update({"status": "failed", "latency_ms": None, "validated_response": None, "summary_text": None, "summary_structured": None})
            results.append(row)
            continue

        try:
            row["request_sent"] = True
            raw_text, latency_ms = _post_chat_completion(
                endpoint=endpoint,
                model=runtime_config.local_model,
                prompt_text=prompt["prompt_text"],
                timeout_s=min(float(config.timeout_s), float(runtime_config.timeout_s)),
                completion_max_tokens=config.completion_max_tokens,
                json_only_mode=config.json_only_mode,
                reasoning_suppression=config.reasoning_suppression,
            )
            row["latency_ms"] = latency_ms
            row["response_received"] = bool(raw_text)
            row["raw_response_text"] = raw_text[: config.raw_response_max_chars]

            parsed_http = json.loads(raw_text)
            extraction = _extract_payload_from_wrapper(parsed_http)
            row["response_wrapper_summary"] = extraction.get("wrapper_summary")
            row["extracted_candidate_text"] = (extraction.get("candidate_text") or "")[: config.raw_response_max_chars]
            row["extracted_candidate_source"] = extraction.get("candidate_source")
            row["finish_reason"] = extraction.get("finish_reason")
            row["truncated"] = bool(extraction.get("truncated"))
            row["json_found_in_response"] = bool(extraction.get("json_found", False))

            if extraction.get("payload") is not None:
                parsed_payload = extraction["payload"]
            else:
                if extraction.get("failure_category"):
                    row["failure_category"] = extraction["failure_category"]
                    raise ValueError(extraction["failure_category"])
                try:
                    parsed_payload = _extract_json_object(extraction.get("candidate_text") or "")
                except json.JSONDecodeError as exc:
                    row["failure_category"] = "truncated_response" if row["truncated"] else "json_parse_failure"
                    raise ValueError(row["failure_category"]) from exc

            normalized_payload = normalize_sanity_response_payload(
                parsed_payload,
                expected_agent_name=agent.name,
                fallback_role=agent.role,
            )

            row["parsed_success"] = True
            row["json_parsed"] = True
            row["parsed_response"] = normalized_payload

            valid, errors = validate_sanity_response_schema(
                normalized_payload,
                expected_agent_name=agent.name,
            )
            row["validation_success"] = valid
            row["schema_validated"] = valid
            row["validation_errors"] = errors

            if isinstance(runtime_bootstrap, dict):
                if valid:
                    summary_text, summary_structured = _build_bootstrap_summary(
                        normalized_payload,
                        max_chars=bootstrap_summary_max_chars,
                    )
                    runtime_bootstrap.update(
                        {
                            "status": "success",
                            "latency_ms": latency_ms,
                            "validated_response": normalized_payload,
                            "summary_text": summary_text,
                            "summary_structured": summary_structured,
                        }
                    )
                    row["bootstrap_summary_text"] = summary_text
                else:
                    row["failure_category"] = "schema_validation_failure"
                    runtime_bootstrap.update(
                        {
                            "status": "failed",
                            "latency_ms": latency_ms,
                            "validated_response": None,
                            "summary_text": None,
                            "summary_structured": None,
                        }
                    )
        except TimeoutError as exc:
            row["timeout"] = True
            row["error"] = f"TimeoutError: {exc}"
            if isinstance(runtime_bootstrap, dict):
                runtime_bootstrap.update({"status": "timeout", "latency_ms": None, "validated_response": None, "summary_text": None, "summary_structured": None})
        except (error.URLError, error.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
            if row.get("failure_category") is None:
                if isinstance(exc, json.JSONDecodeError):
                    row["failure_category"] = "json_parse_failure"
                elif isinstance(exc, ValueError) and str(exc) in {
                    "empty_content_wrapper",
                    "reasoning_only_response",
                    "truncated_response",
                    "json_not_found_in_response",
                    "json_parse_failure",
                }:
                    row["failure_category"] = str(exc)
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(runtime_bootstrap, dict):
                runtime_bootstrap.update({"status": "failed", "latency_ms": None, "validated_response": None, "summary_text": None, "summary_structured": None})
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(runtime_bootstrap, dict):
                runtime_bootstrap.update({"status": "failed", "latency_ms": None, "validated_response": None, "summary_text": None, "summary_structured": None})

        provider = runtime.get("provider")
        if hasattr(provider, "last_outcome") and isinstance(provider.last_outcome, dict):
            row["fallback_used"] = bool(provider.last_outcome.get("fallback", False))

        results.append(row)
        _emit_progress(
            progress_callback,
            stage="agent_result",
            current=idx,
            total=total_agents,
            agent_name=getattr(agent, "name", None),
            detail="success" if row.get("validation_success") else (row.get("error") or row.get("failure_category") or "failed"),
        )

    summary = {
        "startup_llm_sanity_enabled": True,
        "startup_llm_sanity_started_epoch_ms": timestamp,
        "startup_llm_sanity_agent_count": len(results),
        "startup_llm_sanity_success_count": sum(1 for r in results if r.get("validation_success")),
        "startup_llm_sanity_failure_count": sum(1 for r in results if not r.get("validation_success")),
        "startup_llm_sanity_timeout_count": sum(1 for r in results if r.get("timeout")),
        "startup_llm_sanity_parse_failure_count": sum(1 for r in results if r.get("response_received") and not r.get("parsed_success")),
        "bootstrap_reuse_enabled": bootstrap_reuse_enabled,
        "bootstrap_reuse_agent_count": len(simulation.agents) if bootstrap_reuse_enabled else 0,
    }

    artifact_payload = {
        "summary": summary,
        "startup_sanity_request_config": {
            "timeout_s": float(config.timeout_s),
            "completion_max_tokens": int(config.completion_max_tokens),
            "json_only_mode_requested": bool(config.json_only_mode),
            "reasoning_suppression_requested": bool(config.reasoning_suppression),
            "unrestricted_local_qwen_mode": bool(getattr(getattr(simulation, "brain_backend_config", None), "unrestricted_local_qwen_mode", False)),
            "effective_startup_llm_sanity_timeout_seconds": float(getattr(config, "timeout_s", 0.0) or 0.0),
            "effective_startup_llm_sanity_completion_max_tokens": int(getattr(config, "completion_max_tokens", 0) or 0),
            "effective_planner_timeout_seconds": float(getattr(getattr(simulation, "brain_backend_config", None), "timeout_s", 0.0) or 0.0),
            "effective_warmup_timeout_seconds": float(getattr(getattr(simulation, "brain_backend_config", None), "warmup_timeout_s", 0.0) or 0.0),
            "effective_planner_completion_max_tokens": int(getattr(getattr(simulation, "brain_backend_config", None), "completion_max_tokens", 0) or 0),
            "permissive_timeout_ceiling_s": float(getattr(getattr(simulation, "brain_backend_config", None), "permissive_timeout_ceiling_s", 0.0) or 0.0),
            "permissive_completion_ceiling_tokens": int(getattr(getattr(simulation, "brain_backend_config", None), "permissive_completion_ceiling_tokens", 0) or 0),
            "stale_result_relaxation_enabled": bool(getattr(getattr(simulation, "brain_backend_config", None), "unrestricted_local_qwen_mode", False)) or float(getattr(getattr(simulation, "planner_defaults", {}), "get", lambda *_: 0.0)("high_latency_stale_result_grace_s", 0.0) or 0.0) > 0.0,
        },
        "schema_fields": SANITY_RESPONSE_FIELDS,
        "session_semantics": {
            "simulator_agents_persistent": True,
            "bootstrap_is_explicit_and_session_scoped": True,
            "model_calls_are_stateless_unless_bootstrap_summary_is_explicitly_included": True,
            "hidden_model_side_session_memory_relied_on": False,
        },
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
    _emit_progress(
        progress_callback,
        stage="startup_complete",
        current=total_agents,
        total=total_agents,
        detail="Startup LLM sanity check complete.",
    )

    return {
        **summary,
        "startup_llm_sanity_artifact": f"logs/{config.artifact_name}",
        "startup_llm_sanity_results": results,
    }
