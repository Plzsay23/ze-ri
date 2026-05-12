from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES


logger = logging.getLogger(__name__)


def is_visual_policy_feature(feature: Any) -> bool:
    return isinstance(feature, PolicyFeature) and feature.type is FeatureType.VISUAL


def is_dataset_image_feature(key: str, feature: Any) -> bool:
    if not isinstance(feature, Mapping):
        return False

    dtype = feature.get("dtype")
    if dtype in {"image", "video"}:
        return True

    return key.startswith(f"{OBS_IMAGES}.")


def get_dataset_camera_keys(ds_meta: Any) -> list[str]:
    """
    Prefer dataset.meta.camera_keys.
    Fallback to dataset.meta.features image/video keys.
    """
    camera_keys = list(getattr(ds_meta, "camera_keys", None) or [])
    if camera_keys:
        return camera_keys

    features = getattr(ds_meta, "features", {}) or {}
    return [key for key, ft in features.items() if is_dataset_image_feature(key, ft)]


def get_dataset_visual_features(ds_meta: Any) -> dict[str, PolicyFeature]:
    """
    Convert only dataset image/video features to PolicyFeature.
    Shape is converted from HWC to CHW when needed.
    """
    features = getattr(ds_meta, "features", {}) or {}
    out: dict[str, PolicyFeature] = {}

    for key, ft in features.items():
        if not is_dataset_image_feature(key, ft):
            continue

        shape = tuple(ft["shape"])
        names = ft.get("names") or []

        # Dataset image/video is normally HWC.
        if len(shape) != 3:
            raise ValueError(f"Image feature {key!r} must have 3D shape, got {shape}")

        if len(names) >= 3 and names[2] in {"channel", "channels"}:
            shape = (shape[2], shape[0], shape[1])

        out[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)

    return out


def get_policy_visual_keys(policy_or_cfg: Any) -> list[str]:
    cfg = getattr(policy_or_cfg, "config", policy_or_cfg)
    input_features = getattr(cfg, "input_features", None) or {}
    return [key for key, ft in input_features.items() if is_visual_policy_feature(ft)]


def sync_policy_camera_keys_from_dataset(
    policy_cfg: Any,
    ds_meta: Any,
    *,
    rename_map: dict[str, str] | None = None,
    force: bool = True,
    log: logging.Logger | None = None,
) -> list[str]:
    """
    Make policy_cfg.input_features use dataset camera keys.

    This is the important part:
    - remove old VISUAL features from policy config
    - insert dataset visual features
    - store dataset_camera_keys for debugging/checkpoint metadata

    `force=True` means pretrained configs with fixed camera keys are overwritten.
    """
    log = log or logger
    rename_map = rename_map or {}

    dataset_visual_features = get_dataset_visual_features(ds_meta)
    dataset_camera_keys = list(dataset_visual_features.keys())

    if not dataset_camera_keys:
        log.warning("No dataset camera keys found. Skip camera-key sync.")
        return []

    if policy_cfg.input_features is None:
        policy_cfg.input_features = {}

    old_visual_keys = [
        key for key, ft in policy_cfg.input_features.items() if is_visual_policy_feature(ft)
    ]

    if old_visual_keys and not force:
        log.info("Policy visual keys already exist and force=False. Keep: %s", old_visual_keys)
        return old_visual_keys

    # Keep non-visual input features such as observation.state.
    non_visual = {
        key: ft
        for key, ft in policy_cfg.input_features.items()
        if not is_visual_policy_feature(ft)
    }

    policy_cfg.input_features = {**non_visual, **dataset_visual_features}

    # These fields are added in PreTrainedConfig below.
    if hasattr(policy_cfg, "dataset_camera_keys"):
        policy_cfg.dataset_camera_keys = dataset_camera_keys
    else:
        setattr(policy_cfg, "dataset_camera_keys", dataset_camera_keys)

    if hasattr(policy_cfg, "camera_key_map"):
        policy_cfg.camera_key_map = dict(rename_map)
    else:
        setattr(policy_cfg, "camera_key_map", dict(rename_map))

    log.info("Synced policy camera keys from dataset.")
    log.info("Old visual keys: %s", old_visual_keys)
    log.info("Dataset visual keys: %s", dataset_camera_keys)

    return dataset_camera_keys


def _suffix(key: str) -> str:
    if key.startswith(f"{OBS_IMAGES}."):
        return key.removeprefix(f"{OBS_IMAGES}.")
    return key.split(".")[-1]


def infer_camera_key_map(
    *,
    expected_keys: list[str],
    provided_keys: list[str],
    explicit_map: dict[str, str] | None = None,
    allow_single_camera_fallback: bool = True,
) -> dict[str, str]:
    """
    Return mapping from provided observation key -> expected trained policy key.

    Example:
      provided: observation.images.top
      expected: observation.images.image
      result:   {"observation.images.top": "observation.images.image"}

    explicit_map supports either:
      {"top": "observation.images.image"}
      {"observation.images.top": "observation.images.image"}
      {"observation.images.image": "observation.images.top"}  # also tolerated below
    """
    explicit_map = explicit_map or {}
    expected_set = set(expected_keys)
    provided_set = set(provided_keys)

    mapping: dict[str, str] = {}

    # 1. Same names need no remap.
    for key in expected_set & provided_set:
        mapping[key] = key

    # 2. Explicit map.
    for src, dst in explicit_map.items():
        src_full = src if src.startswith(f"{OBS_IMAGES}.") else f"{OBS_IMAGES}.{src}"
        dst_full = dst if dst.startswith(f"{OBS_IMAGES}.") else f"{OBS_IMAGES}.{dst}"

        if src_full in provided_set and dst_full in expected_set:
            mapping[src_full] = dst_full
        elif dst_full in provided_set and src_full in expected_set:
            # tolerate reversed map
            mapping[dst_full] = src_full

    # 3. Suffix match.
    expected_by_suffix = {_suffix(k): k for k in expected_keys}
    for provided in provided_keys:
        if provided in mapping:
            continue
        s = _suffix(provided)
        if s in expected_by_suffix:
            mapping[provided] = expected_by_suffix[s]

    # 4. Single-camera fallback.
    unmapped_expected = [k for k in expected_keys if k not in set(mapping.values())]
    unmapped_provided = [k for k in provided_keys if k not in mapping]

    if (
        allow_single_camera_fallback
        and len(expected_keys) == 1
        and len(provided_keys) == 1
        and unmapped_expected
        and unmapped_provided
    ):
        mapping[unmapped_provided[0]] = unmapped_expected[0]

    return mapping


def remap_observation_camera_keys(
    observation: dict[str, Any],
    policy_image_features: dict[str, PolicyFeature],
    *,
    camera_key_map: dict[str, str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Convert observation image keys to the keys expected by the trained policy config.
    """
    expected_keys = list(policy_image_features.keys())
    if not expected_keys:
        return observation

    provided_keys = [key for key in observation if key.startswith(f"{OBS_IMAGES}.")]

    mapping = infer_camera_key_map(
        expected_keys=expected_keys,
        provided_keys=provided_keys,
        explicit_map=camera_key_map,
    )

    out = dict(observation)

    for src, dst in mapping.items():
        if src in out and dst not in out:
            out[dst] = out[src]

    missing = [key for key in expected_keys if key not in out]

    if missing and strict:
        raise KeyError(
            "Observation camera keys do not match trained policy camera keys.\n"
            f"Missing policy image keys: {missing}\n"
            f"Expected policy image keys: {expected_keys}\n"
            f"Provided observation image keys: {provided_keys}\n"
            f"Inferred camera_key_map: {mapping}\n"
            "Fix robot camera names, dataset camera names, or pass an explicit rename_map/camera_key_map."
        )

    # Remove extra image keys that the policy was not trained with.
    # This avoids models accidentally consuming lexicographically sorted extra cameras.
    for key in provided_keys:
        if key not in expected_keys and key in out:
            out.pop(key, None)

    return out


def convert_uint8_images_to_float(
    batch: dict[str, Any],
    camera_keys: list[str],
) -> dict[str, Any]:
    """
    Convert uint8 images to float32 [0, 1].
    """
    out = batch
    for cam_key in camera_keys:
        if cam_key in out and isinstance(out[cam_key], torch.Tensor) and out[cam_key].dtype == torch.uint8:
            out[cam_key] = out[cam_key].to(dtype=torch.float32) / 255.0
    return out