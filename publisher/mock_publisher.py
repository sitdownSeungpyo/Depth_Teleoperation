"""MockPublisher — in-memory recording publisher used for CI and the Phase 1 demo (spec §4.5)."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from core.types import JointCommand
from publisher.base import InterpolatingPublisherBase


class MockPublisher(InterpolatingPublisherBase):
    def __init__(
        self,
        rate_hz: int = 100,
        history: int | None = None,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(rate_hz=rate_hz)
        self._history: deque[JointCommand] = deque(maxlen=history)
        self._log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")

    @property
    def history(self) -> list[JointCommand]:
        return list(self._history)

    def _emit(self, command: JointCommand) -> None:
        self._history.append(command)
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": command.timestamp,
                            "source_frame_ts": command.source_frame_ts,
                            "positions": command.positions,
                        }
                    )
                    + "\n"
                )
