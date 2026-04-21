"""Configuration loading for the NZ economic forecasting project."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config/config.yaml is missing.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_api_key() -> str:
    """Return the Stats NZ ADE subscription key from the environment.

    Loads .env from the project root if present.

    Raises:
        EnvironmentError: If ADE_API_KEY is not set.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    key = os.getenv("ADE_API_KEY")
    if not key:
        raise EnvironmentError(
            "ADE_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your Stats NZ API key."
        )
    return key


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root.

    Args:
        relative: Relative path string from config (e.g. 'data/raw/').

    Returns:
        Absolute Path object.
    """
    p = _PROJECT_ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p
