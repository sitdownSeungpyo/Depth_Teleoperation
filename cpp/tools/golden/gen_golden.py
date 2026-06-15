"""Generate golden reference data from the Python implementation.

Run in the project's 3.11 venv (the one that runs `pytest`). Writes JSON files
under cpp/tests/golden/data/ capturing inputs + expected outputs of the Python
modules. The C++ golden tests read those JSONs and assert numeric equality, so
the port is pinned to the reference bit-for-bit (within tolerance).

    python cpp/tools/golden/gen_golden.py

Deterministic: no RNG, no timestamps — same inputs every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# repo root = cpp/tools/golden/ -> up 3
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.aligner import align_to_torso  # noqa: E402
from core.filter import OneEuroFilter, OneEuroParams  # noqa: E402
from core.retarget import (  # noqa: E402
    Calibration,
    RobotGeometry,
    retarget_full_upper_body,
)
from core.types import SkeletonFrame  # noqa: E402
from tracker.gravity import GravityEstimator  # noqa: E402

OUT = ROOT / "cpp" / "tests" / "golden" / "data"


def _v(a) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


def gen_mathutil() -> dict:
    median_cases = [[1.0], [3.0, 1.0, 2.0], [4.0, 1.0, 2.0, 3.0], [5.0, 5.0, 1.0, 9.0, 2.0]]
    pct_cases = [([1.0, 2.0, 3.0, 4.0], 40.0), ([0.5, 0.1, 0.9, 0.3, 0.7], 50.0),
                 ([10.0, 20.0], 25.0)]
    circ_cases = [[3.0, -3.0], [0.1, 0.2, -0.1], [np.pi, -np.pi + 0.01], [0.0, 0.0]]
    return {
        "median": [{"input": c, "expected": float(np.median(c))} for c in median_cases],
        "percentile": [
            {"input": c, "p": p, "expected": float(np.percentile(c, p))} for c, p in pct_cases
        ],
        "circular_mean": [
            {
                "input": c,
                "expected": (
                    0.0
                    if abs(float(np.mean(np.sin(c)))) < 1e-12
                    and abs(float(np.mean(np.cos(c)))) < 1e-12
                    else float(np.arctan2(float(np.mean(np.sin(c))), float(np.mean(np.cos(c)))))
                ),
            }
            for c in circ_cases
        ],
    }


def gen_one_euro() -> dict:
    params = OneEuroParams(min_cutoff=0.5, beta=0.05, d_cutoff=1.0)
    f = OneEuroFilter(params)
    samples = [(0.0, 0.0), (1.0, 0.033), (1.2, 0.066), (5.0, 0.099), (5.1, 0.132), (5.0, 0.165)]
    expected = [f.update(x, t) for x, t in samples]
    return {
        "params": {"min_cutoff": 0.5, "beta": 0.05, "d_cutoff": 1.0},
        "samples": [[x, t] for x, t in samples],
        "expected": [float(e) for e in expected],
    }


def gen_gravity() -> dict:
    est = GravityEstimator(lpf_alpha=0.2, warmup_frames=3, norm_tol=0.30, axis_sign=1.0)
    g = 9.80665
    samples = [
        [0.0, -g, 0.0],
        [0.1, -g, 0.05],
        [0.0, 0.0, 0.0],       # zero -> rejected
        [0.0, -g * 2.0, 0.0],  # 2g -> outlier rejected
        [-0.05, -g, 0.0],
        [0.0, -g, 0.1],
    ]
    out = []
    for s in samples:
        up = est.update(np.array(s, dtype=np.float64))
        out.append(None if up is None else _v(up))
    return {
        "params": {"lpf_alpha": 0.2, "warmup_frames": 3, "norm_tol": 0.30, "axis_sign": 1.0},
        "samples": samples,
        "expected_up": out,
    }


def _pose(name: str) -> SkeletonFrame:
    """Synthetic camera-frame poses (+x right, +y down image convention for a
    MediaPipe-like world frame; we pass gravity_up=[0,-1,0])."""
    if name == "tpose":
        kp = {
            "head": [0.0, -0.55, 0.0],
            "left_shoulder": [0.20, -0.30, 0.0],
            "right_shoulder": [-0.20, -0.30, 0.0],
            "left_elbow": [0.45, -0.30, 0.0],
            "right_elbow": [-0.45, -0.30, 0.0],
            "left_wrist": [0.70, -0.30, 0.0],
            "right_wrist": [-0.70, -0.30, 0.0],
            "left_hip": [0.12, 0.10, 0.0],
            "right_hip": [-0.12, 0.10, 0.0],
        }
    elif name == "bent":
        kp = {
            "head": [0.0, -0.55, 0.0],
            "left_shoulder": [0.20, -0.30, 0.0],
            "right_shoulder": [-0.20, -0.30, 0.0],
            "left_elbow": [0.42, -0.30, 0.05],
            "right_elbow": [-0.42, -0.30, 0.05],
            "left_wrist": [0.42, -0.10, 0.22],
            "right_wrist": [-0.42, -0.10, 0.22],
            "left_hip": [0.12, 0.10, 0.0],
            "right_hip": [-0.12, 0.10, 0.0],
        }
    else:  # "reach" — frontal reach toward camera (+z)
        kp = {
            "head": [0.0, -0.55, 0.0],
            "left_shoulder": [0.20, -0.30, 0.0],
            "right_shoulder": [-0.20, -0.30, 0.0],
            "left_elbow": [0.28, -0.28, 0.22],
            "right_elbow": [-0.28, -0.28, 0.22],
            "left_wrist": [0.30, -0.26, 0.46],
            "right_wrist": [-0.30, -0.26, 0.46],
            "left_hip": [0.12, 0.10, 0.0],
            "right_hip": [-0.12, 0.10, 0.0],
        }
    keypoints = {k: np.array(v, dtype=np.float64) for k, v in kp.items()}
    confidence = {k: 1.0 for k in kp}
    return SkeletonFrame(timestamp=0.0, keypoints=keypoints, confidence=confidence)


def gen_aligner() -> list:
    cases = []
    grav = [0.0, -1.0, 0.0]
    for name in ("tpose", "bent", "reach"):
        frame = _pose(name)
        aligned = align_to_torso(frame, gravity_up=np.array(grav, dtype=np.float64))
        cases.append(
            {
                "name": name,
                "keypoints": {k: _v(v) for k, v in frame.keypoints.items()},
                "confidence": {k: float(c) for k, c in frame.confidence.items()},
                "gravity_up": grav,
                "expected_rotation": [_v(row) for row in aligned.rotation],
                "expected_rpy": list(aligned.rpy),
                "expected_keypoints": {k: _v(v) for k, v in aligned.keypoints.items()},
            }
        )
    return cases


def gen_retarget() -> list:
    robot = RobotGeometry(upper_arm_length=0.25, lower_arm_length=0.25,
                          shoulder_offset=(0.0, 0.0, 0.0))
    grav = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cal = Calibration(operator_arm_length=0.50)
    cases = []
    for name in ("tpose", "bent", "reach"):
        aligned = align_to_torso(_pose(name), gravity_up=grav)
        for decouple in (False, True):
            out = retarget_full_upper_body(aligned, robot, cal, decouple_pitch_elbow=decouple)
            cases.append(
                {
                    "name": f"{name}_{'decoupled' if decouple else 'coupled'}",
                    "aligned_keypoints": {k: _v(v) for k, v in aligned.keypoints.items()},
                    "rpy": list(aligned.rpy),
                    "robot": {"upper": 0.25, "lower": 0.25},
                    "calibration": {"operator_arm_length": 0.50, "rest_offsets": {}},
                    "decouple": decouple,
                    "expected": {j: float(v) for j, v in out.items()},
                }
            )
    return cases


def gen_numik() -> dict:
    """Numerical IK on the real robot model. Needs `mujoco` installed; the model
    + arm config come from config/ubp.yaml so C++ replays the identical setup."""
    from core.config import load_config
    from core.robot_model import RobotModel

    cfg = load_config(ROOT / "config" / "ubp.yaml")
    rc = dict(cfg["robot"])
    rc["model_path"] = str((ROOT / rc["model_path"]).resolve())  # absolutize for both sides
    rm = RobotModel(rc)

    # Operator points in the torso frame (+x right, +y up, +z fwd). Right side is
    # -x, left is +x. Magnitudes are irrelevant (solve_arm uses directions + the
    # robot's own link lengths), but the sequence matters (warm-start).
    poses = [
        ("right", [-0.18, 0.0, 0.0], [-0.18, -0.26, 0.0], [-0.18, -0.52, 0.0]),   # hang
        ("right", [-0.18, 0.0, 0.0], [-0.18, 0.0, 0.26], [-0.18, 0.0, 0.52]),     # reach fwd
        ("right", [-0.18, 0.0, 0.0], [-0.18, -0.26, 0.0], [-0.18, -0.26, 0.26]),  # bent
        ("left", [0.18, 0.0, 0.0], [0.18, -0.26, 0.0], [0.18, -0.52, 0.0]),
        ("left", [0.18, 0.0, 0.0], [0.18, 0.0, 0.26], [0.18, 0.0, 0.52]),
        ("left", [0.18, 0.0, 0.0], [0.18, -0.26, 0.0], [0.18, -0.26, 0.26]),
    ]
    calls = []
    for side, sh, el, wr in poses:
        sol = rm.solve_arm(side, np.array(sh), np.array(el), np.array(wr))
        calls.append({
            "side": side, "shoulder": sh, "elbow": el, "wrist": wr,
            "expected": (None if sol is None else {k: float(v) for k, v in sol.items()}),
        })

    ik = rc.get("ik", {})
    return {
        "config": {
            "model_path": rc["model_path"],
            "operator_to_robot_R": rc.get("operator_to_robot_R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "ik": {
                "damping": float(ik.get("damping", 0.08)),
                "max_iters": int(ik.get("max_iters", 16)),
                "pos_tol": float(ik.get("pos_tol", 2e-3)),
                "step_clip": float(ik.get("step_clip", 0.35)),
                "max_target_step_m": float(ik.get("max_target_step_m", 0.0)),
            },
            "joint_limits": {k: [float(v[0]), float(v[1])]
                             for k, v in (rc.get("joint_limits") or {}).items()},
            "arms": rc["arms"],
        },
        "link_lengths": {s: [float(v[0]), float(v[1])] for s, v in rm.link_lengths().items()},
        "calls": calls,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "mathutil.json": gen_mathutil(),
        "one_euro.json": gen_one_euro(),
        "gravity.json": gen_gravity(),
        "aligner.json": gen_aligner(),
        "retarget.json": gen_retarget(),
    }
    # numik needs the optional `mujoco` dependency; skip cleanly if unavailable.
    try:
        data["numik.json"] = gen_numik()
    except Exception as exc:  # noqa: BLE001
        print(f"skipping numik golden (mujoco unavailable?): {exc}")
    for fname, payload in data.items():
        (OUT / fname).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {OUT / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
