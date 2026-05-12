#!/usr/bin/env python

"""
Quiet VLA LoRA training script.

Supported:
- policy.type=groot
- policy.path=...xvla... or policy.type=xvla

Console output policy:
- Hyperparameter summary
- Training started
- Progress every log_freq
- Training finished
- Save/push result
- Real errors only
"""

import contextlib
import dataclasses
import importlib.machinery
import logging
import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

# ---------------------------------------------------------------------
# Very early noise/dependency guards.
# Must run before importing LeRobot, diffusers, transformers, etc.
# ---------------------------------------------------------------------

def _peek_policy_family_from_argv() -> str | None:
    """Detect target family from CLI without importing LeRobot."""
    argv = sys.argv[1:]
    values = {}

    for i, arg in enumerate(argv):
        if arg.startswith("--policy.type="):
            values["type"] = arg.split("=", 1)[1].strip().lower()
        elif arg == "--policy.type" and i + 1 < len(argv):
            values["type"] = argv[i + 1].strip().lower()
        elif arg.startswith("--policy.path="):
            values["path"] = arg.split("=", 1)[1].strip().lower()
        elif arg == "--policy.path" and i + 1 < len(argv):
            values["path"] = argv[i + 1].strip().lower()
        elif arg.startswith("--policy.pretrained_path="):
            values["pretrained_path"] = arg.split("=", 1)[1].strip().lower()
        elif arg == "--policy.pretrained_path" and i + 1 < len(argv):
            values["pretrained_path"] = argv[i + 1].strip().lower()

    ptype = values.get("type")
    if ptype in {"groot", "xvla"}:
        return ptype

    joined = " ".join(v for v in values.values() if v)
    if "groot" in joined or "gr00t" in joined:
        return "groot"
    if "xvla" in joined:
        return "xvla"
    return None


EARLY_POLICY_FAMILY = _peek_policy_family_from_argv()


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")


def _install_flash_attn_import_stub() -> None:
    """Provide a safe import-time flash_attn package stub.

    This is intentionally an import-time compatibility shim. It lets modules that
    merely import flash_attn symbols continue loading in environments where a
    broken flash_attn wheel is installed. If the model actually calls one of the
    flash-attn kernels, the stub raises a clear error.

    The stub must behave like a package because XVLA/Florence imports:
        flash_attn.flash_attn_interface
        flash_attn.bert_padding
    """
    existing = sys.modules.get("flash_attn")
    if existing is not None and getattr(existing, "__spec__", None) is not None:
        # If a real package was already imported successfully, keep it.
        return

    def _not_available(*args, **kwargs):
        raise ImportError(
            "flash_attn kernel was called, but flash_attn is unavailable or CUDA-incompatible. "
            "Install a working flash-attn build or switch the model to a non-flash attention path."
        )

    def _index_first_axis(x, indices):
        # Fallback used by some non-kernel paths.
        return x[indices]

    def _unpad_input(hidden_states, attention_mask):
        # Minimal Python fallback following the common flash_attn.bert_padding API.
        import torch

        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
            (1, 0),
        )
        return hidden_states.reshape(-1, hidden_states.shape[-1])[indices], indices, cu_seqlens, max_seqlen_in_batch

    def _pad_input(hidden_states, indices, batch_size, seqlen):
        import torch

        output = torch.zeros(
            batch_size * seqlen,
            *hidden_states.shape[1:],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output[indices] = hidden_states
        return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    bert_padding_mod = types.ModuleType("flash_attn.bert_padding")

    flash_attn_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    flash_attn_mod.__path__ = []
    flash_attn_interface_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.flash_attn_interface", loader=None
    )
    bert_padding_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)

    flash_attn_mod.flash_attn_func = _not_available
    flash_attn_mod.flash_attn_varlen_func = _not_available

    flash_attn_interface_mod.flash_attn_func = _not_available
    flash_attn_interface_mod.flash_attn_varlen_func = _not_available

    bert_padding_mod.index_first_axis = _index_first_axis
    bert_padding_mod.pad_input = _pad_input
    bert_padding_mod.unpad_input = _unpad_input

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding_mod


if EARLY_POLICY_FAMILY == "xvla":
    _install_flash_attn_import_stub()

warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*falling back to 'pyav'.*")
warnings.filterwarnings("ignore", message=r".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*doesn't directly inherit from `GenerationMixin`.*")


import torch
from huggingface_hub import HfApi
from torch.optim import Optimizer

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.camera_key_utils import (
    convert_uint8_images_to_float,
    get_dataset_camera_keys,
)
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, format_big_number, init_logging


# ---------------------------------------------------------------------
# Supported VLA category
# ---------------------------------------------------------------------

VLA_LORA_CATEGORY = {
    "groot": "GR00T action DiT LoRA",
    "xvla": "XVLA SoftPromptedTransformer action LoRA",
}
SUPPORTED_LORA_POLICIES = set(VLA_LORA_CATEGORY.keys())


GROOT_ACTION_DIT_PREFIX = "_groot_model.action_head.model."
GROOT_ACTION_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
)

XVLA_TRANSFORMER_PREFIX = "model.transformer.blocks."
XVLA_ACTION_ATTENTION_SUFFIXES = (
    ".attn.qkv",
    ".attn.proj",
)
XVLA_ACTION_MLP_SUFFIXES = (
    ".mlp.fc1",
    ".mlp.fc2",
)
XVLA_ACTION_WITH_MLP = False

# XVLA base checkpoint has model.transformer.pos_emb length 512.
# Some real datasets produce longer sequences, e.g. 1204.
# Do NOT instantiate XVLA with max_len_seq=2048 before loading checkpoint,
# because strict load will fail on pos_emb shape mismatch.
# Load with checkpoint length first, then expand pos_emb at runtime.
XVLA_CHECKPOINT_MAX_LEN_SEQ = 512
XVLA_RUNTIME_MIN_LEN_SEQ = 2048



@contextlib.contextmanager
def _quiet_external_output(enabled: bool = True):
    """Suppress noisy print/log output from vendor/model loading or forward.

    Exceptions still propagate normally after the context exits.
    """
    if not enabled:
        yield
        return

    root_logger = logging.getLogger()
    old_level = root_logger.level

    try:
        with open(os.devnull, "w") as devnull:
            root_logger.setLevel(logging.ERROR)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        root_logger.setLevel(old_level)


# ---------------------------------------------------------------------
# Compact logging
# ---------------------------------------------------------------------

def _setup_quiet_logging(accelerator=None) -> None:
    init_logging(accelerator=accelerator)

    noisy_loggers = [
        "huggingface_hub",
        "transformers",
        "transformers.generation",
        "diffusers",
        "accelerate",
        "peft",
        "torch",
        "torchvision",
        "lerobot.utils.import_utils",
        "lerobot.datasets",
        "lerobot.datasets.video_utils",
        "lerobot.datasets.dataset_reader",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.optim",
        "lerobot.optim.schedulers",
        "lerobot.policies.xvla.modeling_xvla",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.groot_n1",
        "transformers_modules",
        "transformers_modules.eagle2hg_hyphen_processor_hyphen_groot_hyphen_n1p5",
    ]

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)


# ---------------------------------------------------------------------
# PyAV decoder patch
# ---------------------------------------------------------------------

def _force_pyav_video_decoder() -> None:
    """Force LeRobot dataset decoding to use memory-safe PyAV seeking.

    This patches the exact modules used by dataset_reader._decode_single.
    It is deliberately done before make_dataset()/DataLoader creation.

    Important:
    For the first validation run, use --num_workers=0 so the patch runs in the
    same process as data loading. Multi-worker DataLoader may import modules in
    child processes before the monkey patch is visible.
    """
    import importlib

    try:
        import av
        import numpy as np
    except Exception as exc:
        raise ImportError("PyAV is required. Install with: uv pip install av") from exc

    video_utils = importlib.import_module("lerobot.datasets.video_utils")
    dataset_reader = importlib.import_module("lerobot.datasets.dataset_reader")

    def _as_float_list(timestamps):
        if timestamps is None:
            return []
        if hasattr(timestamps, "detach"):
            timestamps = timestamps.detach().cpu().flatten().tolist()
        elif hasattr(timestamps, "flatten"):
            timestamps = list(timestamps.flatten())
        elif isinstance(timestamps, (list, tuple)):
            timestamps = list(timestamps)
        else:
            timestamps = [timestamps]
        return [float(t) for t in timestamps]

    def _decode_video_frames_pyav(video_path, timestamps=None, tolerance_s=None, backend=None, **kwargs):
        if timestamps is None:
            timestamps = kwargs.get("timestamps", None)

        query_ts = _as_float_list(timestamps)
        if not query_ts:
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8)

        path = str(video_path)
        frames = []

        for ts in query_ts:
            container = av.open(path)
            stream = container.streams.video[0]

            time_base = float(stream.time_base) if stream.time_base is not None else 1.0 / 30.0
            seek_pts = max(int(ts / time_base), 0)

            try:
                container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
            except Exception:
                try:
                    container.seek(0)
                except Exception:
                    pass

            best_frame = None
            best_dt = float("inf")

            for frame in container.decode(stream):
                if frame.pts is not None and stream.time_base is not None:
                    frame_t = float(frame.pts * stream.time_base)
                elif stream.average_rate is not None:
                    frame_t = 0.0
                else:
                    frame_t = 0.0

                dt = abs(frame_t - ts)
                if dt < best_dt:
                    best_dt = dt
                    best_frame = frame

                if frame_t >= ts and best_frame is not None:
                    break

            if best_frame is None:
                container.close()
                raise RuntimeError(f"No frame decoded near timestamp={ts} from video={path}")

            frames.append(best_frame.to_rgb().to_ndarray())
            container.close()

        array = np.stack(frames, axis=0)  # [T,H,W,C]
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    # Patch every public decode entry that may call torchvision.io.VideoReader.
    video_utils.decode_video_frames = _decode_video_frames_pyav
    if hasattr(video_utils, "decode_video_frames_torchvision"):
        video_utils.decode_video_frames_torchvision = _decode_video_frames_pyav
    if hasattr(video_utils, "decode_video_frames_pyav"):
        video_utils.decode_video_frames_pyav = _decode_video_frames_pyav

    # dataset_reader imports decode_video_frames as a module-level symbol.
    dataset_reader.decode_video_frames = _decode_video_frames_pyav

    # Also patch function globals defensively, in case the symbol was captured.
    for obj in list(vars(dataset_reader).values()):
        fn = getattr(obj, "__func__", obj)
        if hasattr(fn, "__globals__") and "decode_video_frames" in fn.__globals__:
            fn.__globals__["decode_video_frames"] = _decode_video_frames_pyav

        if isinstance(obj, type):
            for member in vars(obj).values():
                member_fn = getattr(member, "__func__", member)
                if hasattr(member_fn, "__globals__") and "decode_video_frames" in member_fn.__globals__:
                    member_fn.__globals__["decode_video_frames"] = _decode_video_frames_pyav

    logging.info("video decoder: forced PyAV seek decoder")


# ---------------------------------------------------------------------
# Training step copied locally to avoid importing noisy lerobot_train.py
# ---------------------------------------------------------------------

def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        with _quiet_external_output(enabled=True):
            loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# ---------------------------------------------------------------------
# LoRA target discovery
# ---------------------------------------------------------------------

def _normalize_target_modules(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def _discover_groot_action_lora_targets(policy: torch.nn.Module) -> list[str]:
    targets = []
    for name, module in policy.named_modules():
        if name.startswith(GROOT_ACTION_DIT_PREFIX) and isinstance(module, torch.nn.Linear):
            if name.endswith(GROOT_ACTION_SUFFIXES):
                targets.append(name)

    if not targets:
        raise RuntimeError(
            "No GR00T LoRA targets found under "
            f"{GROOT_ACTION_DIT_PREFIX} ending with {GROOT_ACTION_SUFFIXES}"
        )
    return targets


def _discover_xvla_action_lora_targets(policy: torch.nn.Module) -> list[str]:
    suffixes = list(XVLA_ACTION_ATTENTION_SUFFIXES)
    if XVLA_ACTION_WITH_MLP:
        suffixes.extend(XVLA_ACTION_MLP_SUFFIXES)
    suffixes = tuple(suffixes)

    targets = []
    for name, module in policy.named_modules():
        if name.startswith(XVLA_TRANSFORMER_PREFIX) and isinstance(module, torch.nn.Linear):
            if name.endswith(suffixes):
                targets.append(name)

    if not targets:
        raise RuntimeError(
            "No XVLA LoRA targets found under "
            f"{XVLA_TRANSFORMER_PREFIX} ending with {suffixes}"
        )
    return targets


def _discover_vla_lora_targets(policy_type: str, policy: torch.nn.Module) -> list[str]:
    if policy_type == "groot":
        return _discover_groot_action_lora_targets(policy)
    if policy_type == "xvla":
        return _discover_xvla_action_lora_targets(policy)
    raise ValueError(f"Unsupported policy.type={policy_type}. Supported={sorted(SUPPORTED_LORA_POLICIES)}")


def _build_vla_lora_config(policy_type: str, policy: torch.nn.Module, peft_overrides: dict[str, Any] | None):
    from peft import LoraConfig

    peft_overrides = dict(peft_overrides or {})
    peft_overrides.pop("method_type", None)

    target_modules = _normalize_target_modules(peft_overrides.pop("target_modules", None))
    if not target_modules:
        target_modules = _discover_vla_lora_targets(policy_type, policy)

    if "full_training_modules" in peft_overrides:
        peft_overrides["modules_to_save"] = peft_overrides.pop("full_training_modules")

    init_type = peft_overrides.pop("init_type", None)
    if init_type is not None:
        peft_overrides["init_lora_weights"] = init_type

    r = peft_overrides.pop("r", None) or 8
    lora_alpha = peft_overrides.pop("lora_alpha", None) or 32
    lora_dropout = peft_overrides.pop("lora_dropout", None) or 0.05
    bias = peft_overrides.pop("bias", None) or "none"

    kwargs = {
        "target_modules": target_modules,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
    }

    for key, value in peft_overrides.items():
        if value is not None:
            kwargs[key] = value

    return LoraConfig(**kwargs), target_modules


# ---------------------------------------------------------------------
# Save / push
# ---------------------------------------------------------------------

def _get_lerobot_policy_config(model: torch.nn.Module) -> PreTrainedConfig:
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
    raise RuntimeError("Could not find LeRobot policy config in PEFT wrapped model.")


def _save_adapter_artifacts(
    cfg: TrainPipelineConfig,
    peft_model: torch.nn.Module,
    preprocessor,
    postprocessor,
    policy_type: str,
    target_modules: list[str],
) -> Path:
    save_dir = Path(cfg.output_dir) / f"{policy_type}_lora_adapter"
    save_dir.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(save_dir)

    policy_config = _get_lerobot_policy_config(peft_model)
    policy_config.save_pretrained(save_dir)

    cfg.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    readme = f"""# {policy_type.upper()} LoRA Adapter

Base model:

```text
{cfg.policy.pretrained_path}
```

Dataset:

```text
{cfg.dataset.repo_id}
```

LoRA target modules: {len(target_modules)}
"""
    (save_dir / "README.md").write_text(readme, encoding="utf-8")
    return save_dir


def _push_adapter_to_hub(cfg: TrainPipelineConfig, local_dir: Path) -> None:
    api = HfApi()
    repo_id = api.create_repo(
        repo_id=cfg.policy.repo_id,
        private=cfg.policy.private,
        exist_ok=True,
    ).repo_id

    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        commit_message="Upload VLA LoRA adapter",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        ignore_patterns=["*.tmp", "*.log"],
    )
    logging.info("pushed: %s", info.repo_url.url)



# ---------------------------------------------------------------------
# XVLA pos_emb runtime expansion
# ---------------------------------------------------------------------

def _prepare_xvla_config_for_strict_load(cfg: TrainPipelineConfig, policy_type: str) -> int | None:
    """Avoid checkpoint load mismatch for XVLA pos_emb.

    If --policy.max_len_seq=2048 is passed before make_policy(), the model is
    instantiated with pos_emb [1,2048,H], but lerobot/xvla-base checkpoint
    contains pos_emb [1,512,H]. strict=True loading then fails.

    Therefore:
      1. construct/load with checkpoint length 512
      2. expand pos_emb in memory after load
    """
    if policy_type != "xvla":
        return None

    requested = getattr(cfg.policy, "max_len_seq", XVLA_CHECKPOINT_MAX_LEN_SEQ)
    try:
        requested = int(requested)
    except Exception:
        requested = XVLA_CHECKPOINT_MAX_LEN_SEQ

    target_len = max(requested, XVLA_RUNTIME_MIN_LEN_SEQ)

    if hasattr(cfg.policy, "max_len_seq"):
        cfg.policy.max_len_seq = XVLA_CHECKPOINT_MAX_LEN_SEQ

    return target_len


def _expand_xvla_pos_emb_after_load(policy: torch.nn.Module, target_len: int | None) -> None:
    """Expand XVLA SoftPromptedTransformer.pos_emb after checkpoint load.

    Extra positions are initialized by repeating the last checkpoint position.
    Existing checkpoint positions remain unchanged.
    """
    if target_len is None:
        return

    transformer = None
    if hasattr(policy, "model") and hasattr(policy.model, "transformer"):
        transformer = policy.model.transformer

    if transformer is None or not hasattr(transformer, "pos_emb"):
        raise RuntimeError("XVLA transformer.pos_emb not found. Cannot expand max_len_seq.")

    pos_emb = transformer.pos_emb
    current_len = int(pos_emb.shape[1])

    if current_len >= target_len:
        return

    with torch.no_grad():
        old_data = pos_emb.detach()
        extra = old_data[:, -1:, :].repeat(1, target_len - current_len, 1)
        new_data = torch.cat([old_data, extra], dim=1).to(device=old_data.device, dtype=old_data.dtype)

    transformer.pos_emb = torch.nn.Parameter(new_data, requires_grad=pos_emb.requires_grad)

    logging.info("xvla pos_emb expanded: %d -> %d", current_len, target_len)


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

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


def _count_params(policy: torch.nn.Module) -> tuple[int, int, int]:
    trainable = 0
    total = 0
    tensors = 0
    for _, param in policy.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            tensors += 1
    return trainable, total, tensors


@parser.wrap()
def train_vla_lora(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    require_package("accelerate", extra="training")
    require_package("peft", extra="training")

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    cfg.validate()

    _force_pyav_video_decoder()

    policy_type = str(cfg.policy.type).lower()
    if policy_type not in SUPPORTED_LORA_POLICIES:
        raise ValueError(f"Unsupported policy.type={policy_type}. Supported={sorted(SUPPORTED_LORA_POLICIES)}")

    if cfg.policy.pretrained_path is None:
        raise ValueError("This script requires a base model through --policy.path or --policy.pretrained_path.")

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=(cfg.policy.device == "cpu"),
        )

    _setup_quiet_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = not cfg.cudnn_deterministic
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

    if is_main_process:
        logging.info("=" * 80)
        logging.info("%s LoRA TRAINING", policy_type.upper())
        logging.info("dataset=%s", cfg.dataset.repo_id)
        logging.info("base=%s", cfg.policy.pretrained_path)
        logging.info("repo=%s", cfg.policy.repo_id)
        logging.info("output=%s", cfg.output_dir)
        logging.info(
            "steps=%s | batch_size=%s | workers=%s | device=%s | dtype=%s | r=%s",
            cfg.steps,
            cfg.batch_size,
            cfg.num_workers,
            cfg.policy.device,
            getattr(cfg.policy, "dtype", None),
            getattr(cfg.peft, "r", None) if cfg.peft is not None else None,
        )
        logging.info("=" * 80)

    if is_main_process:
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)

    dataset_camera_keys = get_dataset_camera_keys(dataset.meta)

    xvla_runtime_max_len_seq = _prepare_xvla_config_for_strict_load(cfg, policy_type)

    with _quiet_external_output(enabled=is_main_process):
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    _expand_xvla_pos_emb_after_load(policy, xvla_runtime_max_len_seq)

    peft_overrides = dataclasses.asdict(cfg.peft) if cfg.peft is not None else {}
    lora_config, target_modules = _build_vla_lora_config(policy_type, policy, peft_overrides)

    policy = policy.wrap_with_peft(peft_config=lora_config)

    trainable, total, trainable_tensors = _count_params(policy)

    if is_main_process:
        logging.info(
            "lora=%s | targets=%d | trainable=%s/%s | tensors=%d",
            VLA_LORA_CATEGORY[policy_type],
            len(target_modules),
            format_big_number(trainable),
            format_big_number(total),
            trainable_tensors,
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=None,
        dataset_stats=dataset.meta.stats,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    dataloader = _make_dataloader(cfg, dataset, device)

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)

    dl_iter = cycle(dataloader)
    policy.train()

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
        logging.info(
            "ready: frames=%s | episodes=%s | effective_batch=%s",
            format_big_number(dataset.num_frames),
            dataset.num_episodes,
            cfg.batch_size * accelerator.num_processes,
        )
        logging.info("started")

    for step in range(1, cfg.steps + 1):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = convert_uint8_images_to_float(batch, dataset_camera_keys)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        train_tracker.step()

        if cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process:
            loss = getattr(train_tracker, "loss", None)
            grad = getattr(train_tracker, "grad_norm", None)
            lr = getattr(train_tracker, "lr", None)
            logging.info("progress: %d/%d | loss=%s | grad=%s | lr=%s", step, cfg.steps, loss, grad, lr)
            train_tracker.reset_averages()

    if is_main_process:
        logging.info("finished")

        unwrapped_policy = accelerator.unwrap_model(policy)
        adapter_dir = _save_adapter_artifacts(
            cfg=cfg,
            peft_model=unwrapped_policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_type=policy_type,
            target_modules=target_modules,
        )
        logging.info("saved: %s", adapter_dir)

        if cfg.policy.push_to_hub:
            _push_adapter_to_hub(cfg, adapter_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train_vla_lora()


if __name__ == "__main__":
    main()
