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

GR00T example:
python -m lerobot.scripts.lerobot_train_vla_lora \
  --policy.type=groot \
  --policy.pretrained_path=nvidia/GR00T-N1.5-3B \
  --policy.repo_id=yoohoolala/egg_pick_groot_action_lora \
  --dataset.repo_id=yoohoolala/egg_pick_dataset \
  --steps=300 \
  --batch_size=1 \
  --num_workers=4 \
  --log_freq=10 \
  --policy.push_to_hub=true \
  --peft.method_type=lora \
  --peft.r=8 \
  --peft.lora_alpha=16 \
  --peft.lora_dropout=0.05

XVLA SO101 bimanual example:
python -m lerobot.scripts.lerobot_train_vla_lora \
  --policy.type=xvla \
  --policy.pretrained_path=yoohoolala/base_xvla_model \
  --policy.repo_id=yoohoolala/egg_pick_xvla_action_lora \
  --dataset.repo_id=yoohoolala/egg_pick_dataset \
  --policy.action_mode=so101_bimanual \
  --steps=300 \
  --batch_size=1 \
  --num_workers=4 \
  --log_freq=10 \
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

from lerobot.configs import PreTrainedConfig, parser
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


VLA_LORA_CATEGORY = {
    "groot": {
        "family": "groot",
        "description": "GR00T action DiT LoRA",
    },
    "xvla": {
        "family": "xvla",
        "description": "XVLA SoftPromptedTransformer action LoRA",
    },
}

SUPPORTED_LORA_POLICIES = set(VLA_LORA_CATEGORY.keys())


# GR00T target:
# GrootPolicy._groot_model.action_head.model = DiT(...)
GROOT_ACTION_DIT_PREFIX = "_groot_model.action_head.model."
GROOT_ACTION_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
)


# XVLA target:
# XVLAPolicy.model.transformer = SoftPromptedTransformer(...)
XVLA_TRANSFORMER_PREFIX = "model.transformer.blocks."
XVLA_ACTION_ATTENTION_SUFFIXES = (
    ".attn.qkv",
    ".attn.proj",
)
XVLA_ACTION_MLP_SUFFIXES = (
    ".mlp.fc1",
    ".mlp.fc2",
)

# 1차는 attention only. 성능 부족 시 True로 바꿔서 MLP까지 LoRA를 붙이십시오.
XVLA_ACTION_WITH_MLP = False


def _normalize_target_modules(value: Any) -> list[str] | None:
    """Normalize target_modules from CLI/config."""
    if value is None:
        return None

    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
        return [x for x in items if x]

    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]

    return [str(value).strip()]


def _log_lora_targets(policy_type: str, targets: list[str]) -> None:
    logging.info("=" * 80)
    logging.info("[%s] LoRA category: %s", policy_type, VLA_LORA_CATEGORY[policy_type]["description"])
    logging.info("[%s] discovered %d LoRA target modules.", policy_type, len(targets))

    for target in targets[:40]:
        logging.info("  LoRA target: %s", target)

    if len(targets) > 40:
        logging.info("  ... and %d more", len(targets) - 40)

    logging.info("=" * 80)


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

    _log_lora_targets("groot", targets)
    return targets


def _discover_xvla_action_lora_targets(policy: torch.nn.Module) -> list[str]:
    """Find exact Linear module names inside XVLA SoftPromptedTransformer.

    This targets only the XVLA action transformer, not Florence2 VLM.
    """
    suffixes = list(XVLA_ACTION_ATTENTION_SUFFIXES)

    if XVLA_ACTION_WITH_MLP:
        suffixes.extend(XVLA_ACTION_MLP_SUFFIXES)

    suffixes = tuple(suffixes)

    targets: list[str] = []

    for name, module in policy.named_modules():
        if not name.startswith(XVLA_TRANSFORMER_PREFIX):
            continue

        if not isinstance(module, torch.nn.Linear):
            continue

        if name.endswith(suffixes):
            targets.append(name)

    if not targets:
        raise RuntimeError(
            "No XVLA action LoRA target modules were found.\n"
            f"Expected Linear modules under '{XVLA_TRANSFORMER_PREFIX}' ending with:\n"
            "  - .attn.qkv\n"
            "  - .attn.proj\n"
            "\n"
            "If XVLA structure changed, print named_modules() and update target suffixes."
        )

    _log_lora_targets("xvla", targets)
    return targets


def _discover_vla_lora_targets(policy_type: str, policy: torch.nn.Module) -> list[str]:
    """Dispatch LoRA target discovery by VLA category."""
    policy_type = str(policy_type).lower()

    if policy_type == "groot":
        return _discover_groot_action_lora_targets(policy)

    if policy_type == "xvla":
        return _discover_xvla_action_lora_targets(policy)

    raise ValueError(
        f"Unsupported policy type for VLA LoRA: {policy_type}. "
        f"Supported: {sorted(SUPPORTED_LORA_POLICIES)}"
    )


def _build_vla_lora_config(
    *,
    policy_type: str,
    policy: torch.nn.Module,
    peft_overrides: dict[str, Any] | None,
):
    """Build PEFT LoraConfig for GR00T/XVLA action-side LoRA."""
    require_package("peft", extra="training")
    from peft import LoraConfig

    policy_type = str(policy_type).lower()
    peft_overrides = dict(peft_overrides or {})

    # We build LoraConfig directly here.
    peft_overrides.pop("method_type", None)

    # User may explicitly override target_modules.
    target_modules = _normalize_target_modules(peft_overrides.pop("target_modules", None))

    if not target_modules:
        target_modules = _discover_vla_lora_targets(policy_type, policy)
    else:
        logging.info("[%s] using user-provided target_modules:", policy_type)
        for target in target_modules:
            logging.info("  LoRA target: %s", target)

    # LeRobot CLI name -> PEFT name
    if "full_training_modules" in peft_overrides:
        peft_overrides["modules_to_save"] = peft_overrides.pop("full_training_modules")

    # LeRobot init_type -> PEFT LoRA init key
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

    logging.info("=" * 80)
    logging.info("[%s] Final LoRA config", policy_type)
    logging.info(pformat(config_kwargs))
    logging.info("=" * 80)

    return LoraConfig(**config_kwargs), target_modules


def _get_lerobot_policy_config(model: torch.nn.Module) -> PreTrainedConfig:
    """Find original LeRobot policy config from a PeftModel or base policy."""
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]

    for obj in candidates:
        if obj is None:
            continue

        cfg = getattr(obj, "config", None)
        if isinstance(cfg, PreTrainedConfig):
            return cfg

    raise RuntimeError(
        "Could not find LeRobot policy config on the model. "
        "Expected GrootConfig or XVLAConfig under PeftModel.base_model.model.config."
    )


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

    target_preview = "\n".join(target_modules[:120])
    if len(target_modules) > 120:
        target_preview += f"\n... and {len(target_modules) - 120} more"

    content = f"""# {policy_type.upper()} LoRA Adapter

This repository contains a LoRA adapter for a LeRobot VLA policy.

## Policy type

```text
{policy_type}
```

## Base model

```text
{base_model}
```

## Dataset

```text
{dataset_id}
```

## Adapter repo

```text
{repo_id}
```

## LoRA target modules

Number of exact target modules:

```text
{len(target_modules)}
```

Preview:

```text
{target_preview}
```

## Notes

This is adapter-only. It is intended to be loaded on top of the base model.
"""
    (save_dir / "README.md").write_text(content, encoding="utf-8")


def _save_adapter_artifacts(
    *,
    cfg: TrainPipelineConfig,
    peft_model: torch.nn.Module,
    preprocessor,
    postprocessor,
    policy_type: str,
    target_modules: list[str],
) -> Path:
    save_dir = Path(cfg.output_dir) / f"{policy_type}_lora_adapter"
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Saving %s LoRA adapter to: %s", policy_type, save_dir)

    # PEFT adapter-only:
    #   adapter_config.json
    #   adapter_model.safetensors
    peft_model.save_pretrained(save_dir)

    # Save LeRobot policy config next to adapter.
    policy_config = _get_lerobot_policy_config(peft_model)
    policy_config.save_pretrained(save_dir)

    # Save train config and processors.
    cfg.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    _write_adapter_readme(
        save_dir=save_dir,
        cfg=cfg,
        policy_type=policy_type,
        target_modules=target_modules,
    )

    return save_dir


def _push_adapter_to_hub(
    *,
    cfg: TrainPipelineConfig,
    local_dir: Path,
) -> None:
    if not cfg.policy.repo_id:
        raise ValueError("cfg.policy.repo_id is required when policy.push_to_hub=true.")

    api = HfApi()
    repo_id = api.create_repo(
        repo_id=cfg.policy.repo_id,
        private=cfg.policy.private,
        exist_ok=True,
    ).repo_id

    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        commit_message="Upload VLA LoRA adapter",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        ignore_patterns=["*.tmp", "*.log"],
    )

    logging.info("VLA LoRA adapter pushed to: %s", commit_info.repo_url.url)


def _make_dataloader(cfg: TrainPipelineConfig, dataset, device: torch.device):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )


def _log_trainable_parameters(policy: torch.nn.Module) -> None:
    trainable = []
    frozen_count = 0
    trainable_count = 0
    total_count = 0

    for name, param in policy.named_parameters():
        n = param.numel()
        total_count += n
        if param.requires_grad:
            trainable_count += n
            trainable.append((name, n))
        else:
            frozen_count += n

    logging.info("=" * 80)
    logging.info("Trainable parameter summary")
    logging.info("  trainable params: %d", trainable_count)
    logging.info("  frozen params:    %d", frozen_count)
    logging.info("  total params:     %d", total_count)
    logging.info("  trainable ratio:  %.6f", trainable_count / max(total_count, 1))

    for name, n in trainable[:80]:
        logging.info("  trainable: %s (%d)", name, n)

    if len(trainable) > 80:
        logging.info("  ... and %d more trainable tensors", len(trainable) - 80)

    logging.info("=" * 80)


@parser.wrap()
def train_vla_lora(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator | None" = None,
):
    """Train only VLA action-side LoRA adapter."""
    require_package("accelerate", extra="training")
    require_package("peft", extra="training")

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    cfg.validate()

    policy_type = str(cfg.policy.type).lower()

    if policy_type not in SUPPORTED_LORA_POLICIES:
        raise ValueError(
            f"This script supports only {sorted(SUPPORTED_LORA_POLICIES)}, "
            f"got policy.type={policy_type!r}."
        )

    if cfg.policy.pretrained_path is None:
        raise ValueError(
            "This script requires --policy.pretrained_path=<base model repo/local path>."
        )

    if policy_type == "xvla":
        action_mode = getattr(cfg.policy, "action_mode", None)
        if action_mode != "so101_bimanual":
            logging.warning(
                "[XVLA] Current policy.action_mode=%s. "
                "For SO101 bimanual training, recommended: --policy.action_mode=so101_bimanual",
                action_mode,
            )

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info("=" * 80)
        logging.info("%s LoRA TRAINING MODE", policy_type.upper())
        logging.info("VLA category: %s", VLA_LORA_CATEGORY[policy_type]["description"])
        logging.info(pformat(cfg.to_dict()))
        logging.info("=" * 80)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Dataset
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)

    dataset_camera_keys = get_dataset_camera_keys(dataset.meta)

    # Base VLA policy
    if is_main_process:
        logging.info("Creating base %s policy", policy_type)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if is_main_process:
        logging.info("Dataset camera keys: %s", dataset_camera_keys)
        logging.info("Policy image features: %s", list(policy.config.image_features.keys()))

    # LoRA config + wrapping
    peft_overrides = dataclasses.asdict(cfg.peft) if cfg.peft is not None else {}

    lora_config, target_modules = _build_vla_lora_config(
        policy_type=policy_type,
        policy=policy,
        peft_overrides=peft_overrides,
    )

    # This freezes the base model and trains only LoRA adapter params.
    policy = policy.wrap_with_peft(peft_config=lora_config)

    if is_main_process:
        _log_trainable_parameters(policy)

    accelerator.wait_for_everyone()

    # Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=None,
        dataset_stats=dataset.meta.stats,
    )

    # Optimizer / scheduler / dataloader
    if is_main_process:
        logging.info("Creating optimizer and scheduler")

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    dataloader = _make_dataloader(cfg, dataset, device)

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy,
        optimizer,
        dataloader,
        lr_scheduler,
    )

    dl_iter = cycle(dataloader)
    policy.train()

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=0,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps,
            desc=f"Training {policy_type.upper()} LoRA",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info("Start %s LoRA training", policy_type.upper())

    # Training loop
    step = 0

    for _ in range(cfg.steps):
        start_time = time.perf_counter()

        batch = next(dl_iter)
        batch = convert_uint8_images_to_float(batch, dataset_camera_keys)
        batch = preprocessor(batch)

        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()

        if is_main_process:
            progbar.update(1)

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process

        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                logging.info("Output dict: %s", output_dict)
            train_tracker.reset_averages()

    if is_main_process:
        progbar.close()
        logging.info("End of %s LoRA training", policy_type.upper())

    accelerator.wait_for_everyone()

    # Save / push adapter
    if is_main_process:
        unwrapped_policy = accelerator.unwrap_model(policy)

        adapter_dir = _save_adapter_artifacts(
            cfg=cfg,
            peft_model=unwrapped_policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_type=policy_type,
            target_modules=target_modules,
        )

        logging.info("Saved adapter artifacts at: %s", adapter_dir)

        if cfg.policy.push_to_hub:
            _push_adapter_to_hub(
                cfg=cfg,
                local_dir=adapter_dir,
            )

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train_vla_lora()


if __name__ == "__main__":
    main()
