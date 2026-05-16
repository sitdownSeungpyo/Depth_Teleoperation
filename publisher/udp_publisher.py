"""UdpPublisherSkeleton — opens a UDP socket and sends a placeholder packet.

This is intentionally a SKELETON (spec §4.5, FR-5.4). The real packet schema is TBD
and depends on the target robot platform; the TODO block below documents the layout
to fill in once the platform is fixed.
"""

from __future__ import annotations

import logging
import socket

from core.types import JointCommand
from publisher.base import InterpolatingPublisherBase

log = logging.getLogger(__name__)


# TODO(robot-platform): replace placeholder payload with the real packet schema.
# Proposed layout per spec §3.3 (paper joint IDs):
#   header  : uint16 magic = 0xRBT1
#   header  : uint16 sequence (wraps)
#   header  : float64 timestamp (perf_counter seconds)
#   payload : repeated { uint8 paper_joint_id, float32 position_rad } * N
#   trailer : uint16 crc16 over header+payload
# Joint IDs in canonical order: 1, 2, 3, 4, 5, 6, 19, 20 (see core.types.JOINT_NAMES_BY_PAPER_ID).


class UdpPublisherSkeleton(InterpolatingPublisherBase):
    def __init__(self, host: str, port: int, rate_hz: int = 100) -> None:
        super().__init__(rate_hz=rate_hz)
        self._addr = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        log.info("UDP publisher (skeleton) targeting %s:%d", host, port)

    def stop(self) -> None:
        super().stop()
        self._socket.close()

    def _emit(self, command: JointCommand) -> None:
        # Placeholder payload — schema to be defined once robot platform is fixed.
        payload = b"PLACEHOLDER:" + repr(command.positions).encode("utf-8", errors="replace")
        try:
            self._socket.sendto(payload, self._addr)
        except OSError as exc:
            log.warning("UDP send failed: %s", exc)
