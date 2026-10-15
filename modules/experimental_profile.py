"""Compact first-pass experimental profile: gestalt constructs -> seven constituents -> runtime mechanisms.

Resolution happens ONCE per agent at construction. Runtime code never looks at teamwork_potential or taskwork_potential;
it reads the resolved mechanisms. Explicit constituent overrides replace (never add to) gestalt-generated values.
Mappings live in config/experimental/experimental_profile.json and are hypothesized defaults, not calibrated truth.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

CONSTITUENTS = (
    "dik_push_propensity",
    "dik_request_propensity",
    "externalization_propensity",
    "memory_reliability",
    "coordination_propensity",
    "backup_propensity",
    "task_execution_efficiency",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@lru_cache(maxsize=4)
def _load(path_str: str) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_experimental_profile_config(config_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    base = Path(config_dir) if config_dir else Path(__file__).resolve().parents[1] / "config" / "experimental"
    path = base / "experimental_profile.json"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "config" / "experimental" / "experimental_profile.json"
    return _load(str(path))


def resolve_experimental_profile(
    construct_values: Optional[Dict[str, float]] = None,
    constituent_overrides: Optional[Dict[str, float]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve one agent's seven-variable profile.

    Returns a JSON-serializable record: the gestalt inputs, the generated values, the explicit overrides, the resolved
    constituents (with the source of each), and the mechanism values those constituents own.
    """
    cfg = config or load_experimental_profile_config()
    constructs = {str(k): float(v) for k, v in dict(construct_values or {}).items() if v is not None}
    explicit = {}
    for key, value in dict(constituent_overrides or {}).items():
        if value is None or str(value).strip() == "":
            continue
        if key not in cfg["constituents"]:
            raise ValueError(f"Unknown experimental constituent '{key}'. Known: {sorted(cfg['constituents'])}")
        explicit[str(key)] = _clamp(float(value))
    generated: Dict[str, float] = {}
    resolved: Dict[str, float] = {}
    source: Dict[str, str] = {}
    for name, spec in cfg["constituents"].items():
        value = float(spec["default"])
        origin = "default"
        rule = spec.get("generated_by")
        if rule and rule["construct"] in constructs:
            construct = constructs[rule["construct"]]
            centered = (construct - 0.5) * 2.0 if rule.get("transform", "linear_centered") == "linear_centered" else construct
            value = _clamp(value + float(rule["weight"]) * centered)
            generated[name] = value
            origin = f"generated:{rule['construct']}"
        if name in explicit:
            value = explicit[name]
            origin = "explicit_override"
        resolved[name] = _clamp(value)
        source[name] = origin
    mechanisms: Dict[str, float] = {}
    for name, spec in cfg["constituents"].items():
        for mechanism in spec["mechanisms"]:
            mechanisms[mechanism] = resolved[name]
    return {
        "schema": cfg.get("schema"),
        "status": cfg.get("status"),
        "gestalt_constructs": constructs,
        "generated_constituents": generated,
        "explicit_constituent_overrides": explicit,
        "resolved_constituents": resolved,
        "constituent_source": source,
        "constituent_mechanisms": mechanisms,
    }