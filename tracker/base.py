"""SkeletonTracker Protocol shared by all tracker backends (spec §4.1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from core.types import SkeletonFrame


@runtime_checkable
class SkeletonTracker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> SkeletonFrame | None: ...
    def stream(self) -> Iterator[SkeletonFrame]: ...
