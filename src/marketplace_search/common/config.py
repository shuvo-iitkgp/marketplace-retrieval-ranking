"""
src/marketplace_search/common/config.py
───────────────────────────────────────
Lightweight YAML config loader with dot-access support.
Keeps all path resolution relative to the project root so
the code can be run from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Thin wrapper around a YAML config dict that supports both
    dict-style (cfg["key"]) and attribute-style (cfg.key) access.
    Nested dicts are recursively wrapped.
    """

    def __init__(self, data: dict, project_root: Path | None = None) -> None:
        self._data = data
        self._root = project_root or Path.cwd()

    # ------------------------------------------------------------------ #
    # Access                                                               #
    # ------------------------------------------------------------------ #

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        return Config(val, self._root) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        return Config(val, self._root) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Config(val, self._root) if isinstance(val, dict) else val

    def to_dict(self) -> dict:
        return self._data

    # ------------------------------------------------------------------ #
    # Path helpers (resolve relative to project root)                     #
    # ------------------------------------------------------------------ #

    def resolve_path(self, key: str) -> Path:
        """Return an absolute Path for a path-valued config key."""
        raw = self._data[key]
        p = Path(raw)
        return p if p.is_absolute() else self._root / p

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data})"


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #


def load_config(path: str | Path, project_root: Path | None = None) -> Config:
    """
    Load a YAML config file and return a Config instance.

    Parameters
    ----------
    path:
        Path to the YAML file.
    project_root:
        Root directory used to resolve relative paths inside the config.
        Defaults to the directory containing the YAML file.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f)

    root = project_root or path.parent
    return Config(data, root)
