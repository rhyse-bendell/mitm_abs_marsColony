from __future__ import annotations

"""Experimental construct/mechanism configuration loader facade.

This module intentionally wraps the existing ConstructMapper so the simulator keeps
one mechanism profile path while allowing a canonical experimental config location.

Layering note:
- `teamwork_potential` and `taskwork_potential` are manipulation constructs.
- Their downstream mechanism hooks can target pilot-capability, mecha-capability,
  or environment/task parameters depending on mapping definitions.
- Legacy `traits` input is preserved as a compatibility alias for mechanism
  overrides while the architecture migrates toward explicit layer labels.
"""

from pathlib import Path

from modules.construct_mapping import ConstructMapper


def default_experimental_config_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    experimental_dir = repo_root / "config" / "experimental"
    if experimental_dir.exists():
        return experimental_dir
    return repo_root / "config"


def load_experimental_mapper(config_dir: str | Path | None = None) -> ConstructMapper:
    return ConstructMapper(config_dir=config_dir or default_experimental_config_dir())


def normalize_mechanism_override_inputs(
    config: dict | None,
    *,
    mechanism_defaults: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Normalize mechanism override inputs with legacy alias support.

    Returns a tuple:
      (normalized_overrides, legacy_traits_alias, explicit_mechanism_overrides)

    Precedence:
      1) baseline mechanism defaults (used only to detect neutral UI autofill)
      2) construct-derived effects (applied by ConstructMapper)
      3) explicit mechanism_overrides (including legacy traits alias)

    Neutral auto-filled defaults are dropped unless explicitly forced via:
      - mechanism_overrides_explicit: true
      - preserve_neutral_mechanism_overrides: true
    """
    payload = config if isinstance(config, dict) else {}
    defaults = {
        str(k): float(v)
        for k, v in dict(mechanism_defaults or {}).items()
    }
    explicit_overrides = {
        str(k): float(v)
        for k, v in dict(payload.get("mechanism_overrides", {}) or {}).items()
        if v is not None
    }
    legacy_traits_alias = {
        str(k): float(v)
        for k, v in dict(payload.get("traits", {}) or {}).items()
        if v is not None
    }
    normalized = dict(legacy_traits_alias)

    preserve_neutral = bool(
        payload.get("mechanism_overrides_explicit")
        or payload.get("preserve_neutral_mechanism_overrides")
    )
    for mechanism_id, override_value in explicit_overrides.items():
        if (
            not preserve_neutral
            and mechanism_id in defaults
            and abs(float(override_value) - float(defaults[mechanism_id])) <= 1e-9
        ):
            continue
        normalized[mechanism_id] = float(override_value)
    return normalized, legacy_traits_alias, explicit_overrides
