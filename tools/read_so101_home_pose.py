#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.robots import make_robot_from_config
from lerobot.robots import so_follower  # noqa: F401
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.utils.import_utils import register_third_party_plugins


def to_float_scalar(value: Any) -> float | None:
    try:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, list) and len(value) == 1:
            return float(value[0])

        if isinstance(value, tuple) and len(value) == 1:
            return float(value[0])

        return float(value)
    except Exception:
        return None


def flatten_state(value: Any) -> list[float] | None:
    try:
        if torch.is_tensor(value):
            return [float(x) for x in value.detach().cpu().flatten().tolist()]

        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]

        return None
    except Exception:
        return None


def get_action_feature_keys(robot) -> list[str]:
    action_features = robot.action_features

    if isinstance(action_features, dict):
        return list(action_features.keys())

    return list(action_features)


def extract_pose_from_observation(robot, obs: dict[str, Any]) -> dict[str, float]:
    action_keys = get_action_feature_keys(robot)

    pose: dict[str, float] = {}
    missing: list[str] = []

    # 1. action key와 observation key가 직접 같은 경우
    for key in action_keys:
        candidates = [
            key,
            f"observation.{key}",
            f"observation.state.{key}",
            key.replace("action.", "observation."),
        ]

        found = False
        for cand in candidates:
            if cand in obs:
                value = to_float_scalar(obs[cand])
                if value is not None:
                    pose[key] = value
                    found = True
                    break

        if not found:
            missing.append(key)

    if len(pose) == len(action_keys):
        return pose

    # 2. observation.state 또는 state 벡터에서 순서대로 읽는 경우
    for state_key in ["observation.state", "state"]:
        if state_key not in obs:
            continue

        state_list = flatten_state(obs[state_key])
        if state_list is None:
            continue

        if len(state_list) >= len(action_keys):
            return {
                key: float(state_list[i])
                for i, key in enumerate(action_keys)
            }

    raise RuntimeError(
        "현재 observation에서 action-space 관절값을 추출하지 못했습니다.\n"
        f"action_keys={action_keys}\n"
        f"missing={missing}\n"
        f"available_obs_keys={sorted(list(obs.keys()))}"
    )


def average_poses(poses: list[dict[str, float]]) -> dict[str, float]:
    if not poses:
        raise ValueError("poses is empty")

    keys = list(poses[0].keys())
    avg = {}

    for key in keys:
        avg[key] = sum(p[key] for p in poses) / len(poses)

    return avg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "ze-ri" / "policies" / "so101_home_pose.json"),
    )

    args = parser.parse_args()

    register_third_party_plugins()

    cfg = SOFollowerConfig(
        port=args.port,
    )

    setattr(cfg, "type", "so101_follower")

    robot = make_robot_from_config(cfg)

    print("[INFO] Connecting robot...")
    robot.connect()

    try:
        print("[INFO] Connected.")
        print("[INFO] robot.action_features:")
        print(robot.action_features)
        print()

        print("[INFO] robot.observation_features:")
        print(robot.observation_features)
        print()

        poses = []

        for i in range(args.samples):
            obs = robot.get_observation()

            if i == 0:
                print("[INFO] raw observation keys:")
                for k in sorted(obs.keys()):
                    print("  -", k)
                print()

            pose = extract_pose_from_observation(robot, obs)
            poses.append(pose)

            print(f"[SAMPLE {i + 1}/{args.samples}]")
            print(json.dumps(pose, indent=4, ensure_ascii=False))
            print()

            time.sleep(args.interval)

        final_pose = average_poses(poses)

        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_pose, f, indent=4, ensure_ascii=False)

        print("[RESULT] averaged home pose:")
        print(json.dumps(final_pose, indent=4, ensure_ascii=False))
        print()
        print(f"[SAVED] {output_path}")

    finally:
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    main()
