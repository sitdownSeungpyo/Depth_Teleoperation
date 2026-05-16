from __future__ import annotations

import logging
import socket
import threading
import time

import pytest

from core.types import JointCommand
from publisher.mock_publisher import MockPublisher
from publisher.udp_publisher import UdpPublisherSkeleton


def _cmd(positions: dict[str, float], ts: float = 0.0) -> JointCommand:
    return JointCommand(timestamp=ts, positions=dict(positions), source_frame_ts=ts)


def test_mock_publisher_reaches_steady_rate() -> None:
    pub = MockPublisher(rate_hz=100)
    pub.start()
    try:
        pub.set_target(_cmd({"r_elbow": 0.0}, ts=time.perf_counter()))
        time.sleep(0.1)
        pub.set_target(_cmd({"r_elbow": 0.5}, ts=time.perf_counter()))
        time.sleep(1.0)  # ~100 emissions at 100 Hz
    finally:
        pub.stop()
    # Windows default timer granularity is ~15 ms, so the lower bound is loose: at
    # least 50 emissions over ~1.1 s (≈ 45 Hz). Upper bound prevents runaway loops.
    assert 50 <= len(pub.history) <= 150, len(pub.history)
    assert pub.history[-1].positions["r_elbow"] == pytest.approx(0.5, abs=1e-6)


def test_mock_publisher_holds_on_stale_input(caplog: pytest.LogCaptureFixture) -> None:
    pub = MockPublisher(rate_hz=100)
    pub.start()
    try:
        ts = time.perf_counter()
        pub.set_target(_cmd({"r_elbow": 0.0}, ts=ts))
        pub.set_target(_cmd({"r_elbow": 1.0}, ts=ts + 0.01))
        # No further updates — drive past the 200 ms staleness threshold.
        with caplog.at_level(logging.WARNING):
            time.sleep(0.4)
    finally:
        pub.stop()
    assert any("stale" in rec.message for rec in caplog.records)
    last = pub.history[-1]
    # Held value should equal the last setpoint.
    assert last.positions["r_elbow"] == pytest.approx(1.0)


def test_udp_publisher_skeleton_sends_packet() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    host, port = sock.getsockname()

    received: list[bytes] = []

    def receiver() -> None:
        try:
            data, _ = sock.recvfrom(4096)
            received.append(data)
        except socket.timeout:
            pass

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()

    pub = UdpPublisherSkeleton(host=host, port=port, rate_hz=200)
    pub.start()
    try:
        pub.set_target(_cmd({"r_elbow": 0.5}, ts=time.perf_counter()))
        thread.join(timeout=1.0)
    finally:
        pub.stop()
        sock.close()
    assert received, "expected at least one UDP packet"
    assert received[0].startswith(b"PLACEHOLDER")
