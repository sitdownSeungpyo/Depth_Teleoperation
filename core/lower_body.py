"""LowerBodyController Protocol (spec §4.7) — future hook, NOT in Phase 1 scope.

Defines the interface only. The default :class:`PassThroughLowerBody` simply forwards
the upper-body command unchanged, so an upper-body-only deployment (the Phase 1
target) ignores the hook entirely.

Real implementations (paper §III + §IV — LIPM, push recovery, walk controller) plug
in here when a lower-body platform is wired up.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import JointCommand, SensorBundle


@runtime_checkable
class LowerBodyController(Protocol):
    def update(self, upper_command: JointCommand, sensors: SensorBundle) -> JointCommand:
        """Return a combined full-body command given the upper-body target + sensors."""
        ...


class PassThroughLowerBody:
    """No-op lower body: passes the upper-body command through unchanged.

    Matches the Phase 1 deployment assumption (supported base, no balance work).
    """

    def update(self, upper_command: JointCommand, sensors: SensorBundle) -> JointCommand:
        del sensors  # unused in pass-through mode
        return upper_command
