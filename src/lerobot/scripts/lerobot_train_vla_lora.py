#!/usr/bin/env python

"""
Train VLA LoRA adapter only.

This script intentionally does NOT replace lerobot_train.py.

Supported policy.type:
- groot
- xvla

Goal:
- Keep existing full fine-tuning script untouched.
- Load a base VLA policy.
- Freeze base model.
- Attach LoRA only to action-side modules.
- Train LoRA adapter only.
- Save/push adapter-only repo with LeRobot config and processors.

Recommended GR00T test:
python -m lerobot.scripts.lerobot_train_vla_lora \
  --policy.type=groot \
  --policy.pretrained_path=nvidia/GR00T-N1.5-3B \
  --policy.repo_id=<HF_USER>/<TASK_NAME>_groot_action_lora \
  --dataset.repo_id=<HF_USER>/<DATASET_NAME> \
  --steps=300 \
  --batch_size=1 \
  --num_workers=4 \
  --policy.push_to_hub=true \
  --peft.method_type=lora \
  --peft.r=8 \
  --peft.lora_alpha=16 \
  --peft.lora_dropout=0.05

Recommended XVLA test:
python -m lerobot.scripts.lerobot_train_vla_lora \
  --policy.type=xvla \
  --policy.pretrained_path=<BASE_XVLA_REPO_OR_LOCAL_PATH> \
  --policy.repo_id=<HF_USER>/<TASK_NAME>_xvla_action_lora \
  --dataset.repo_id=<HF_USER>/<DATASET_NAME> \
  --steps=300 \
  --batch_size=1 \
  --num_workers=4 \
  --policy.push_to_hub=true \
  --peft.method_type=lora \
  --peft.r=8 \
  --peft.lora_alpha=16 \
  --peft.lora_dropout=0.05
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from huggingface_hub import HfApi
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.camera_key_utils import (
    convert_uint8_images_to_float,
    get_dataset_camera_keys,
)
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, format_big_number, init_logging, inside_slurm


SUPPORTED_LORA_POLICIES = {"groot", "xvla"}

GROOT_ACTION_DIT_PREFIX = "_groot_model.action_head.model."

GROOT_ACTION_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
)

# XVLA 실제 파일을 아직 못 봤기 때문에 너무 공격적으로 잡지 않는다.
# action_hub / soft_transformer / action / decoder / policy 계열만 우선 후보로 본다.
XVLA_INCLUDE_HINTS = (
    "action",
    "action_hub",
    "action_head",
    "soft_transformer",
    "policy",
    "decoder",
)

XVLA_EXCLUDE_HINTS = (
    "vision",
    "visual",
    "florence",
    "language",
    "text",
    "tokenizer",
    "token",
    "embed_tokens",
    "embeddings",
    "vlm",
    "backbone",
)

XVLA_LINEAR_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
    ".q_proj",
    ".k_proj",
    ".v_proj",
    ".o_proj",
    ".out_proj",
    ".fc1",
    ".fc2",
    ".proj",
    ".linear",
)


def _normalize_target_modules(value: Any) -> list[str] | None:
    """Normalize target_modules from CLI/config.

    PEFT config may receive:
    - None
    - list[str]
    - tuple[str]
    - single comma-separated string
    """
    if value is None:
        return None

    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
        return [x for x in items if x]

    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]

    return [str(value).strip()]


def _log_targets(policy_type: str, targets: list[str]) -> None:
    logging.info("[%s] discovered %d LoRA target modules.", policy_type, len(targets))
    for target in targets[:30]:
        logging.info("  LoRA target: %s", target)
    if len(targets) > 30:
        logging.info("  ... and %d more", len(targets) - 30)


def _discover_groot_action_lora_targets(policy: torch.nn.Module) -> list[str]:
    """Find exact Linear module names inside GR00T action DiT attention."""
    targets: list[str] = []

    for name, module in policy.named_modules():
        if not name.startswith(GROOT_ACTION_DIT_PREFIX):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.endswith(GROOT_ACTION_SUFFIXES):
            targets.append(name)

    if not targets:
        raise RuntimeError(
            "No GR00T action LoRA target modules were found.\n"
            f"Expected Linear modules under '{GROOT_ACTION_DIT_PREFIX}' ending with:\n"
            "  - to_q\n"
            "  - to_k\n"
            "  - to_v\n"
            "  - to_out.0\n"
            "\n"
            "Check whether GR00T action_head.model still uses diffusers Attention."
        )

    _log_targets("groot", targets)
    return targets


def _looks_like_xvla_action_module(name: str) -> bool:
    lname = name.lower()

    if any(bad in lname for bad in XVLA_EXCLUDE_HINTS):
        return False

    if not any(good in lname for good in XVLA_INCLUDE_HINTS):
        return False

    return name.endswith(XVLA_LINEAR_SUFFIXES)


def _discover_xvla_action_lora_targets(policy: torch.nn.Module) -> list[str]:
    """Heuristic XVLA action-side LoRA target discovery.

    XVLA source files were not provided in this turn, so this intentionally avoids
    touching vision/language/backbone modules and targets action-like Linear modules only.

    If this finds nothing or too much, pass explicit:
      --peft.target_modules=module_a,module_b,...
    """
    targets: list[str] = []

    for name, module in policy.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _looks_like_xvla_action_module(name):
            targets.append(name)

    if not targets:
        # Fallback: print useful candidates to help manual target selection.
        candidates = [
            name
            for name, module in policy.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(hint in name.lower() for hint in XVLA_INCLUDE_HINTS)
        ]

        logging.error("No XVLA action LoRA targets found automatically.")
        logging.error("Possible XVLA Linear candidates:")
        for cand in candidates[:80]:
            logging.error("  candidate: %s", cand)
        if len(candidates) > 80:
            logging.error("  ... and %d more", len(candidates) - 80)

        raise RuntimeError(
            "No XVLA action LoRA target modules were found automatically.\n"
            "Upload these files so the target can be made exact:\n"
            "  src/lerobot/policies/xvla/modeling_xvla.py\n"
            "  src/lerobot/policies/xvla/action_hub.py\n"
            "  src/lerobot/policies/xvla/soft_transformer.py\n"
            "\n"
            "Or pass explicit targets with:\n"
            "  --peft.target_modules=module_a,module_b,..."
        )

    _log_targets("xvla", targets)
    return targets


def _discover_vla_lora_targets(policy_type: str, policy: torch.nn.Module) -> list[str]:
    if policy_type == "groot":
        return _discover_groot_action_lora_targets(policy)

    if policy_type == "xvla":
        return _discover_xvla_action_lora_targets(policy)

    raise ValueError(f"Unsupported policy type for this script: {policy_type}")


def _build_vla_lora_config(
    *,
    policy_type: str,
    policy: torch.nn.Module,
    peft_overrides: dict[str, Any] | None,
):
    require_package("peft", extra="training")
    from peft import LoraConfig

    peft_overrides = dict(peft_overrides or {})

    # We build LoraConfig directly here.
    peft_overrides.pop("method_type", None)

    target_modules = _normalize_target_modules(peft_overrides.pop("target_modules", None))
    if not target_modules:
        target_modules = _discover_vla_lora_targets(policy_type, policy)
    else:
        logging.info("[%s] using user-provided target_modules:", policy_type)
        for target in target_modules:
            logging.info("  LoRA target: %s", target)

    if "full_training_modules" in peft_overrides:
        peft_overrides["modules_to_save"] = peft_overrides.pop("full_training_modules")

    init_type = peft_overrides.pop("init_type", None)
    if init_type is not None:
        peft_overrides["init_lora_weights"] = init_type

    r = peft_overrides.pop("r", None) or 16
    lora_alpha = peft_overrides.pop("lora_alpha", None) or 32
    lora_dropout = peft_overrides.pop("lora_dropout", None) or 0.05
    bias = peft_overrides.pop("bias", None) or "none"

    config_kwargs = {
        "target_modules": target_modules,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
    }

    for key, value in peft_overrides.items():
        if value is not None:
            config_kwargs[key] = value

    logging.info("[%s] LoRA config:", policy_type)
    logging.info(pformat(config_kwargs))

    return LoraConfig(**config_kwargs), target_modules


def _get_lerobot_policy_config(model: torch.nn.Module):
    """Find original LeRobot policy config from PeftModel or base policy."""
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]

    for obj in candidates:
        if obj is None:
            continue

        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "save_pretrained"):
            return cfg

    raise RuntimeError("Could not find LeRobot policy config on the model.")


def _write_adapter_readme(
    *,
    save_dir: Path,
    cfg: TrainPipelineConfig,
    policy_type: str,
    target_modules: list[str],
) -> None:
    base_model = str(cfg.policy.pretrained_path)
    dataset_id = str(cfg.dataset.repo_id)
    repo_id = str(cfg.policy.repo_id)

    target_preview = "\n".join(target_modules[:80])
    if len(target_modules) > 80:
        target_preview += f"\n... and {len(target_modules) - 80} more"

    content = f"""# {policy_type.upper()} LoRA Adapter

This repository contains a LoRA adapter for a LeRobot VLA policy.

## Policy type

```text
{policy_type}