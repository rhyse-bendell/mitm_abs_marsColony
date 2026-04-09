from __future__ import annotations

"""Experimental construct/mechanism configuration loader facade.

This module intentionally wraps the existing ConstructMapper so the simulator keeps
one mechanism profile path while allowing a canonical experimental config location.
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
