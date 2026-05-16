"""Shared pytest fixtures and auto-generation of JSONL test recordings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable so `core`, `tracker`, `publisher` resolve in tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _ensure_fixtures() -> None:
    tpose = FIXTURE_DIR / "tpose.jsonl"
    arm_circle = FIXTURE_DIR / "arm_circle.jsonl"
    if tpose.exists() and arm_circle.exists():
        return
    from tests.generate_fixtures import main as gen

    gen()


@pytest.fixture(scope="session", autouse=True)
def _fixtures() -> None:
    _ensure_fixtures()


@pytest.fixture
def tpose_jsonl() -> Path:
    _ensure_fixtures()
    return FIXTURE_DIR / "tpose.jsonl"


@pytest.fixture
def arm_circle_jsonl() -> Path:
    _ensure_fixtures()
    return FIXTURE_DIR / "arm_circle.jsonl"
