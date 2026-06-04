#!/usr/bin/env python3
"""Extract VLA handoff reference states from a LeRobot dataset.

The output JSON is consumed by vla_handoff_supervisor_node.py.  This script is
intentionally statistics-based, not a training script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray | None:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _to_flat_float_list(value: Any) -> list[float] | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    try:
        arr = arr.astype(np.float64).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return None


def _first_existing(sample: dict[str, Any], keys: list[str]) -> tuple[str | None, Any | None]:
    for key in keys:
        if key in sample:
            return key, sample[key]
    return None, None


def _find_image_key(sample: dict[str, Any]) -> str | None:
    preferred = [
        "observation.images.top",
        "observation.images.cam_left",
        "observation.images.cam_right",
        "observation.image",
        "image",
    ]
    for key in preferred:
        if key in sample:
            return key
    for key in sample.keys():
        if "image" in str(key).lower():
            return str(key)
    return None


def _save_image(value: Any, path: Path) -> bool:
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        else:
            arr = _to_numpy(value)
            if arr is None:
                return False
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                pass
            elif arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                return False
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if np.nanmax(arr) <= 1.0:
                    arr *= 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return True
    except Exception:
        return False


def _load_lerobot_dataset(repo_id: str, root: str | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {"repo_id": repo_id}
    if root:
        kwargs["root"] = root
    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        # Some local forks use positional repo_id.
        return LeRobotDataset(repo_id, root=root) if root else LeRobotDataset(repo_id)


def _extract_episode_ranges(dataset: Any, max_scan: int | None = None) -> list[tuple[int, int]]:
    edi = getattr(dataset, "episode_data_index", None)
    if edi is not None:
        try:
            starts = edi.get("from") or edi.get("start") or edi.get("starts")
            ends = edi.get("to") or edi.get("end") or edi.get("ends")
            if starts is not None and ends is not None:
                s_arr = _to_numpy(starts).reshape(-1).astype(int).tolist()
                e_arr = _to_numpy(ends).reshape(-1).astype(int).tolist()
                return [(int(s), int(e)) for s, e in zip(s_arr, e_arr) if int(e) > int(s)]
        except Exception:
            pass

    # Fallback: scan episode_index from samples.  Slower, but robust for small datasets.
    n = len(dataset)
    if max_scan is not None:
        n = min(n, max_scan)
    ranges: list[tuple[int, int]] = []
    cur_ep = None
    start = 0
    for idx in range(n):
        sample = dataset[idx]
        ep = sample.get("episode_index", sample.get("episode_idx", 0))
        try:
            ep_i = int(_to_numpy(ep).reshape(-1)[0])
        except Exception:
            ep_i = 0
        if cur_ep is None:
            cur_ep = ep_i
            start = idx
        elif ep_i != cur_ep:
            ranges.append((start, idx))
            start = idx
            cur_ep = ep_i
    if cur_ep is not None:
        ranges.append((start, n))
    return ranges


def _feature_names(dataset: Any, key: str | None, width: int) -> list[str]:
    if key is None:
        return [f"state_{i}" for i in range(width)]
    try:
        features = getattr(dataset, "features", {}) or {}
        spec = features.get(key, {}) if isinstance(features, dict) else {}
        names = spec.get("names") or spec.get("feature_names") or spec.get("names_list")
        if names and len(names) == width:
            return [str(n) for n in names]
    except Exception:
        pass

    # SO-101 common order fallback.  If width differs, use state_i.
    so101 = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    if width == len(so101):
        return so101
    return [f"state_{i}" for i in range(width)]


def _stats(rows: list[list[float]], names: list[str], tolerance_sigma: float, min_tolerance: float) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    arr = np.asarray(rows, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    tol = np.maximum(min_tolerance, tolerance_sigma * std)
    return (
        {names[i]: round(float(mean[i]), 6) for i in range(len(names))},
        {names[i]: round(float(std[i]), 6) for i in range(len(names))},
        {names[i]: round(float(tol[i]), 6) for i in range(len(names))},
    )


def _merge_existing(output_json: Path) -> dict[str, Any]:
    if not output_json.exists():
        return {"version": 1, "tasks": {}}
    try:
        with output_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "tasks" not in payload:
            payload = {"version": 1, "tasks": payload}
        return payload
    except Exception:
        return {"version": 1, "tasks": {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default="")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--tail-frames", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--output-json", default="config/vla/handoff_reference_manifest.json")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--tolerance-sigma", type=float, default=3.0)
    parser.add_argument("--min-tolerance", type=float, default=0.05)
    parser.add_argument("--gripper-closed-min", type=float, default=math.nan)
    parser.add_argument("--gripper-closed-max", type=float, default=math.nan)
    args = parser.parse_args()

    dataset = _load_lerobot_dataset(args.repo_id, root=args.root or None)
    episode_ranges = _extract_episode_ranges(dataset)
    if args.max_episodes and args.max_episodes > 0:
        episode_ranges = episode_ranges[: args.max_episodes]

    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    reference_images: list[str] = []
    state_key_seen: str | None = None
    action_key_seen: str | None = None
    image_key_seen: str | None = None

    image_dir = Path(args.image_dir) if args.image_dir else Path("config/vla/references") / args.task_id

    for ep_idx, (start, end) in enumerate(episode_ranges):
        tail_start = max(start, end - max(1, args.tail_frames))
        final_sample = dataset[end - 1]
        if image_key_seen is None:
            image_key_seen = _find_image_key(final_sample)
        if image_key_seen and image_key_seen in final_sample:
            rel = image_dir / f"final_ep_{ep_idx:04d}.png"
            if _save_image(final_sample[image_key_seen], rel):
                reference_images.append(str(rel))

        for idx in range(tail_start, end):
            sample = dataset[idx]
            state_key, state_value = _first_existing(sample, ["observation.state", "state"])
            action_key, action_value = _first_existing(sample, ["action"])
            if state_key_seen is None and state_key:
                state_key_seen = state_key
            if action_key_seen is None and action_key:
                action_key_seen = action_key

            state = _to_flat_float_list(state_value) if state_value is not None else None
            action = _to_flat_float_list(action_value) if action_value is not None else None
            if state:
                state_rows.append(state)
            if action:
                action_rows.append(action)

    if not state_rows and not action_rows:
        raise RuntimeError("No observation.state/state/action rows were found in dataset tail frames")

    task_payload: dict[str, Any] = {
        "source_repo_id": args.repo_id,
        "num_episodes": len(episode_ranges),
        "tail_frames_per_episode": args.tail_frames,
        "state_key": state_key_seen,
        "action_key": action_key_seen,
        "image_key": image_key_seen,
        "reference_images": reference_images,
    }

    if state_rows:
        names = _feature_names(dataset, state_key_seen, len(state_rows[0]))
        mean, std, tol = _stats(state_rows, names, args.tolerance_sigma, args.min_tolerance)
        task_payload.update(
            {
                "state_keys": names,
                "final_state_mean": mean,
                "final_state_std": std,
                "tolerance": tol,
            }
        )

    if action_rows:
        names = _feature_names(dataset, action_key_seen, len(action_rows[0]))
        mean, std, tol = _stats(action_rows, names, args.tolerance_sigma, args.min_tolerance)
        task_payload.update(
            {
                "action_keys": names,
                "final_action_mean": mean,
                "final_action_std": std,
                "action_tolerance": tol,
            }
        )

    if not math.isnan(args.gripper_closed_min) and not math.isnan(args.gripper_closed_max):
        task_payload["gripper_closed_range"] = {
            "min": float(args.gripper_closed_min),
            "max": float(args.gripper_closed_max),
        }

    output_json = Path(args.output_json)
    if args.append:
        payload = _merge_existing(output_json)
    else:
        payload = {"version": 1, "tasks": {}}
    payload.setdefault("tasks", {})[args.task_id] = task_payload
    payload["generated_at_sec"] = time_now = __import__("time").time()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {output_json}")
    print(f"[OK] task={args.task_id} episodes={len(episode_ranges)} images={len(reference_images)} generated_at={time_now}")


if __name__ == "__main__":
    main()
