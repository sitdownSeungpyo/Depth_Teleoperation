"""Config loader with file includes — keeps settings split by purpose.

``config/ubp.yaml`` is a thin orchestrator that lists ``include:`` (paths
relative to itself); each included file holds one purpose (tracker, retarget,
robot, filter, joint_limit, publisher, safety, runtime). They are deep-merged in
listed order, then the including file's own top-level keys are merged on top so
the main file can still override any value.

Merge rule: nested dicts merge recursively; scalars and lists replace. So
``joint_limit.yaml`` can add ``robot.joint_limits`` and ``filter.mechanical_limits``
into the ``robot``/``filter`` sections defined in their own files.

Includes may themselves include (resolved recursively). A plain config file with
no ``include:`` key loads exactly as before — backward compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class ConfigError(RuntimeError):
    """Raised for an unusable config layout (missing or circular include)."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving and deep-merging any ``include:`` files."""
    return _load(Path(path), ())


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        # Without this the recursion runs until RecursionError, whose traceback
        # says nothing about which files form the loop.
        chain = " -> ".join(p.name for p in (*stack, resolved))
        raise ConfigError(f"circular config include: {chain}")
    if not resolved.is_file():
        raise ConfigError(f"config include not found: {path}")
    raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return raw  # type: ignore[no-any-return]
    includes = raw.pop("include", None) or []
    merged: dict[str, Any] = {}
    for inc in includes:
        merged = _deep_merge(merged, _load(resolved.parent / inc, (*stack, resolved)))
    return _deep_merge(merged, raw)
