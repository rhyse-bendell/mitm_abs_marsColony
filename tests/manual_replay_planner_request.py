"""Manual diagnostic replay for one planner request against local Ollama.

Example:
    py -3.11 tests/manual_replay_planner_request.py --input planner_request_bundle.json --model qwen2.5:3b-instruct --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import request

from modules.brain_contract import AgentBrainRequest
from modules.brain_provider import BrainBackendConfig, OllamaLocalBrainProvider, RuleBrain


REQUEST_REQUIRED_KEYS = {
    "request_id",
    "tick",
    "sim_time",
    "agent_id",
    "display_name",
    "task_id",
    "phase",
    "local_context_summary",
    "local_observations",
    "working_memory_summary",
    "inbox_summary",
    "current_goal_stack",
    "current_plan_summary",
    "allowed_actions",
    "planning_horizon_config",
    "request_explanation",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a real AgentBrainRequest against local Ollama for diagnosis.")
    parser.add_argument("--input", required=True, help="Path to lifecycle bundle JSON or raw AgentBrainRequest JSON.")
    parser.add_argument("--model", default="qwen2.5:3b-instruct", help="Model name for local provider payload.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Local Ollama base URL.")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="HTTP endpoint path.")
    parser.add_argument("--timeout", type=float, default=3600.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=24576, help="completion_max_tokens used in provider payload.")
    parser.add_argument(
        "--output-dir",
        default="tests/_manual_replay_output",
        help="Directory where diagnostic artifacts are written.",
    )
    parser.add_argument(
        "--print-provider-payload",
        action="store_true",
        help="Print provider payload JSON to stdout.",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Only build and save provider payload, skip HTTP request.",
    )
    parser.add_argument(
        "--save-provider-payload-only",
        action="store_true",
        help="Build/save request and provider payload + summary, skip HTTP/parse/normalize steps.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_source(input_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str | None]:
    source_json = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source_json, dict):
        raise ValueError("Input root must be a JSON object.")

    source_type = "raw_request"
    request_payload = source_json

    if isinstance(source_json.get("request_payload"), dict):
        source_type = "lifecycle_bundle"
        request_payload = source_json["request_payload"]

    if not isinstance(request_payload, dict):
        raise ValueError("Extracted request_payload is not a JSON object.")

    request_summary = source_json.get("request_summary")
    request_kind = request_summary.get("request_kind") if isinstance(request_summary, dict) else None

    return source_json, request_payload, source_type, request_kind


def _validate_request_like(payload: dict[str, Any]) -> list[str]:
    missing = sorted(REQUEST_REQUIRED_KEYS.difference(payload.keys()))
    return missing


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_json, request_payload_json, source_type, request_kind = _load_source(input_path)
    missing_keys = _validate_request_like(request_payload_json)
    if missing_keys:
        raise ValueError(f"Payload does not look like AgentBrainRequest; missing keys: {missing_keys}")

    if request_kind:
        print(f"request_summary.request_kind: {request_kind}")
    if request_kind and request_kind != "planner":
        print(f"WARNING: request_kind is '{request_kind}', not 'planner'. Proceeding anyway.")

    request_packet = AgentBrainRequest.from_dict(request_payload_json)

    config = BrainBackendConfig(
        backend="ollama_local",
        local_base_url=args.base_url,
        local_endpoint=args.endpoint,
        local_model=args.model,
        timeout_s=float(args.timeout),
        completion_max_tokens=int(args.max_tokens),
    )
    provider = OllamaLocalBrainProvider(config=config, fallback=RuleBrain())

    provider_payload = provider._build_request_payload(request_packet)
    provider_payload_text = json.dumps(provider_payload, ensure_ascii=False)
    provider_payload_size = len(provider_payload_text)

    request_payload_path = output_dir / "request_payload.json"
    provider_payload_path = output_dir / "provider_payload.json"
    _write_json(request_payload_path, request_payload_json)
    _write_json(provider_payload_path, provider_payload)

    if args.print_provider_payload:
        print(json.dumps(provider_payload, indent=2, ensure_ascii=False))

    endpoint_url = f"{args.base_url.rstrip('/')}{args.endpoint}"
    print("--- Replay Setup ---")
    print(f"request_id: {request_packet.request_id}")
    print(f"agent_id: {request_packet.agent_id}")
    print(f"model: {args.model}")
    print(f"provider_payload_size_chars: {provider_payload_size}")
    print(f"timeout_s: {args.timeout}")
    print(f"endpoint: {endpoint_url}")

    summary: dict[str, Any] = {
        "input_file_path": str(input_path),
        "source_type": source_type,
        "request_id": request_packet.request_id,
        "agent_id": request_packet.agent_id,
        "request_kind": request_kind,
        "model": args.model,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "timeout": args.timeout,
        "provider_payload_size": provider_payload_size,
        "wallclock_elapsed_s": 0.0,
        "http_response_received": False,
        "raw_response_nonempty": False,
        "json_parsed": False,
        "provider_parse_succeeded": False,
        "normalization_succeeded": False,
        "runtime_disposition": None,
        "normalization_steps": [],
        "parse_source": None,
        "exception": None,
        "final_verdict": None,
    }

    response_json: dict[str, Any] | None = None
    parsed_provider_content: dict[str, Any] | None = None
    normalized_payload: dict[str, Any] | None = None

    if not args.skip_http and not args.save_provider_payload_only:
        started = time.perf_counter()
        raw_response_text = ""
        try:
            http_request = request.Request(
                endpoint_url,
                data=json.dumps(provider_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(http_request, timeout=float(args.timeout)) as response:
                raw_response_text = response.read().decode("utf-8", errors="replace")
            summary["http_response_received"] = True
        except Exception as exc:  # noqa: BLE001
            summary["exception"] = {"stage": "http", "type": type(exc).__name__, "message": str(exc)}
        finally:
            summary["wallclock_elapsed_s"] = round(time.perf_counter() - started, 6)

        if raw_response_text:
            summary["raw_response_nonempty"] = True
            (output_dir / "raw_response.txt").write_text(raw_response_text, encoding="utf-8")
            preview = raw_response_text[:500]
            print("raw_response_preview_500:")
            print(preview)

            try:
                response_json = json.loads(raw_response_text)
                summary["json_parsed"] = True
                _write_json(output_dir / "response_json.json", response_json)
            except json.JSONDecodeError as exc:
                summary["exception"] = {
                    "stage": "json_parse",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        if response_json is not None:
            try:
                parsed_provider_content = provider._parse_response(response_json)
                summary["provider_parse_succeeded"] = True
                summary["parse_source"] = parsed_provider_content.get("parse_source")
                _write_json(output_dir / "parsed_provider_content.json", parsed_provider_content)
                print(f"provider_parse_source: {summary['parse_source']}")
            except Exception as exc:  # noqa: BLE001
                summary["exception"] = {
                    "stage": "provider_parse",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        if parsed_provider_content is not None:
            try:
                normalized_payload, normalization_steps, runtime_disposition = provider._normalize_payload(
                    parsed_provider_content.get("payload")
                )
                summary["normalization_steps"] = normalization_steps
                summary["runtime_disposition"] = runtime_disposition
                if normalized_payload is not None:
                    summary["normalization_succeeded"] = True
                    _write_json(output_dir / "normalized_payload.json", normalized_payload)
                print(f"normalization_step_count: {len(normalization_steps)}")
            except Exception as exc:  # noqa: BLE001
                summary["exception"] = {
                    "stage": "normalize",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

    verdict = "http_no_response"
    if summary["raw_response_nonempty"] and not summary["json_parsed"]:
        verdict = "raw_response_received_but_not_json"
    elif summary["json_parsed"] and not summary["provider_parse_succeeded"]:
        verdict = "json_received_but_provider_parse_failed"
    elif summary["provider_parse_succeeded"] and not summary["normalization_succeeded"]:
        verdict = "provider_payload_extracted_but_normalization_failed"
    elif summary["normalization_succeeded"]:
        verdict = "normalization_succeeded"
    elif args.skip_http or args.save_provider_payload_only:
        verdict = "provider_payload_built_http_skipped"
    summary["final_verdict"] = verdict

    _write_json(output_dir / "diagnostic_summary.json", summary)

    print("--- Diagnostic Summary ---")
    print(f"http_response_received: {summary['http_response_received']}")
    print(f"json_parsed: {summary['json_parsed']}")
    print(f"provider_parse_succeeded: {summary['provider_parse_succeeded']}")
    print(f"normalization_succeeded: {summary['normalization_succeeded']}")
    print(f"runtime_disposition: {summary['runtime_disposition']}")
    print(f"final_verdict: {summary['final_verdict']}")
    print(f"files_saved_to: {output_dir}")

    if args.save_provider_payload_only:
        print("Note: --save-provider-payload-only was set; HTTP, parsing, and normalization were skipped.")
    elif args.skip_http:
        print("Note: --skip-http was set; HTTP, parsing, and normalization were skipped.")

    # Keep source_json consumed to make clear we intentionally loaded the full source.
    _ = source_json

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
