"""Config include loader — merge semantics and the failure modes it must name."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_includes_deep_merge_in_order(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "s:\n  x: 1\n  y: 1\n")
    _write(tmp_path / "b.yaml", "s:\n  y: 2\n  z: 2\n")
    main = _write(tmp_path / "main.yaml", "include:\n  - a.yaml\n  - b.yaml\ns:\n  z: 3\n")
    cfg = load_config(main)
    # a then b, then the including file's own keys last.
    assert cfg["s"] == {"x": 1, "y": 2, "z": 3}


def test_nested_includes_resolve(tmp_path: Path) -> None:
    _write(tmp_path / "leaf.yaml", "s:\n  deep: 1\n")
    _write(tmp_path / "mid.yaml", "include:\n  - leaf.yaml\ns:\n  mid: 1\n")
    main = _write(tmp_path / "main.yaml", "include:\n  - mid.yaml\n")
    assert load_config(main)["s"] == {"deep": 1, "mid": 1}


def test_circular_include_names_the_cycle(tmp_path: Path) -> None:
    """A cycle used to recurse to RecursionError, whose traceback says nothing
    about which files are involved."""
    _write(tmp_path / "a.yaml", "include:\n  - b.yaml\n")
    _write(tmp_path / "b.yaml", "include:\n  - a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_self_include_is_circular(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "include:\n  - a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_include_is_reported(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.yaml", "include:\n  - nope.yaml\n")
    with pytest.raises(ConfigError, match="not found"):
        load_config(main)


@pytest.mark.parametrize("name", ["ubp.yaml", "default.yaml", "loose_visibility.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(REPO_ROOT / "config" / name)
    # Every shipped entrypoint must supply the sections main.run reads.
    for section in ("tracker", "retarget", "filter", "publisher", "safety", "main"):
        assert section in cfg, f"{name} missing {section}"
    assert "include" not in cfg  # consumed by the loader, never leaks through


def test_default_yaml_overrides_survive_its_includes() -> None:
    """default.yaml now layers on the same split files ubp.yaml uses; its own
    keys must still win, otherwise the mock/replay path silently gets the
    camera config."""
    cfg = load_config(REPO_ROOT / "config" / "default.yaml")
    assert cfg["tracker"]["type"] == "mock"
    assert cfg["tracker"]["realsense"]["enable_imu"] is False
    assert cfg["tracker"]["realsense"]["body_backend"] == "mediapipe"
    # ...while inheriting the values it does NOT override.
    assert "mechanical_limits" in cfg["filter"]
    assert cfg["safety"]["watchdog_timeout_s"] == pytest.approx(0.5)


def test_loose_visibility_relaxes_gates_over_default() -> None:
    strict = load_config(REPO_ROOT / "config" / "default.yaml")
    loose = load_config(REPO_ROOT / "config" / "loose_visibility.yaml")
    assert loose["tracker"]["pose"]["min_visibility"] < strict["tracker"]["pose"]["min_visibility"]
    assert loose["tracker"]["pose"]["min_arm_confidence"] == 0.0
