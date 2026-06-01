#!/usr/bin/env python3
"""SO-101 raw tick home-pose helper.

This helper converts Feetech Present_Position ticks to LeRobot action-space
values for a SO-101 follower calibration JSON, or writes a raw tick JSON that
can be used by robot_client_ros_multi_gate_raw_home.py.

Usage:
  python so101_raw_home_pose_tools.py \
    --calibration calibration.json \
    --raw-ticks so101_start_raw_ticks.json \
    --out-calibrated so101_home_pose.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_RES = 4095
JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_KEY = "gripper"


def load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return obj


def normalize_raw_ticks(raw_obj: dict[str, Any]) -> dict[str, int]:
    if isinstance(raw_obj.get("raw_ticks"), dict):
        raw_obj = raw_obj["raw_ticks"]
    out = {}
    for key, val in raw_obj.items():
        motor = str(key).removesuffix(".pos")
        out[motor] = int(round(float(val)))
    return out


def raw_to_lerobot_action(calibration: dict[str, Any], raw_ticks: dict[str, int]) -> dict[str, float]:
    action = {}
    for motor in JOINT_KEYS:
        if motor not in calibration or motor not in raw_ticks:
            raise KeyError(f"Missing {motor} in calibration/raw ticks")
        c = calibration[motor]
        mid = (float(c["range_min"]) + float(c["range_max"])) / 2.0
        action[f"{motor}.pos"] = (float(raw_ticks[motor]) - mid) * 360.0 / MAX_RES

    if GRIPPER_KEY in calibration and GRIPPER_KEY in raw_ticks:
        c = calibration[GRIPPER_KEY]
        min_tick = float(c["range_min"])
        max_tick = float(c["range_max"])
        drive_mode = int(c.get("drive_mode", 0))
        if max_tick == min_tick:
            raise ValueError("Invalid gripper range: min == max")
        value = ((float(raw_ticks[GRIPPER_KEY]) - min_tick) / (max_tick - min_tick)) * 100.0
        if drive_mode:
            value = 100.0 - value
        action[f"{GRIPPER_KEY}.pos"] = max(0.0, min(100.0, value))

    return action


def validate_raw_ticks(calibration: dict[str, Any], raw_ticks: dict[str, int]) -> list[str]:
    warnings = []
    for motor, tick in raw_ticks.items():
        if motor not in calibration:
            warnings.append(f"{motor}: no calibration entry")
            continue
        c = calibration[motor]
        lo = int(c["range_min"])
        hi = int(c["range_max"])
        if not (lo <= tick <= hi):
            warnings.append(f"{motor}: tick {tick} outside [{lo}, {hi}]")
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True, help="SO-101 calibration JSON path")
    ap.add_argument("--raw-ticks", required=True, help="Raw Present_Position tick JSON path")
    ap.add_argument("--out-calibrated", default="", help="Output LeRobot action home pose JSON")
    ap.add_argument("--out-raw", default="", help="Output normalized raw tick JSON")
    args = ap.parse_args()

    calibration = load_json(args.calibration)
    raw_ticks = normalize_raw_ticks(load_json(args.raw_ticks))

    warnings = validate_raw_ticks(calibration, raw_ticks)
    for w in warnings:
        print(f"[WARN] {w}")

    action = raw_to_lerobot_action(calibration, raw_ticks)

    print("[calibrated action-space home pose]")
    print(json.dumps(action, indent=2, ensure_ascii=False))

    if args.out_calibrated:
        path = Path(args.out_calibrated).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(action, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {path}")

    if args.out_raw:
        path = Path(args.out_raw).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw_ticks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
