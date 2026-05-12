# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import json
import logging
import os
import pickle  # nosec
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


@dataclass(frozen=True)
class PolicyCandidate:
    policy_id: str
    policy_type: str
    pretrained_name_or_path: str
    description: str = ""


@dataclass(frozen=True)
class RouteDecision:
    policy_id: str
    task_for_policy: str
    confidence: float
    reason: str
    router_model: str = "keyword"


def _load_policy_candidates_from_manifest() -> tuple[list[PolicyCandidate], str | None]:
    manifest_path = os.environ.get("LEROBOT_POLICY_MANIFEST", "").strip()
    if not manifest_path:
        return [], None

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"LEROBOT_POLICY_MANIFEST does not exist: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    candidates = []
    for item in manifest.get("policies", []):
        candidates.append(
            PolicyCandidate(
                policy_id=item["id"],
                policy_type=item.get("policy_type", "xvla"),
                pretrained_name_or_path=item["pretrained_name_or_path"],
                description=item.get("description", ""),
            )
        )

    default_policy_id = manifest.get("default_policy_id")
    if default_policy_id is None and candidates:
        default_policy_id = candidates[0].policy_id

    return candidates, default_policy_id


def _extract_first_image(raw_observation: dict[str, Any]) -> Any | None:
    """Try to extract camera image from LeRobot raw observation.

    Works with common keys:
      - image
      - observation.images.image
      - any key containing 'image'
    """
    preferred_keys = [
        "image",
        "observation.images.image",
        "observation.image",
    ]

    for key in preferred_keys:
        if key in raw_observation:
            return raw_observation[key]

    for key, value in raw_observation.items():
        if "image" in key.lower():
            return value

    return None


def _to_pil_image(image: Any) -> Any | None:
    if image is None:
        return None

    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return None

    if isinstance(image, Image.Image):
        return image

    if torch.is_tensor(image):
        arr = image.detach().cpu()

        # Remove batch dimension if present.
        if arr.ndim == 4:
            arr = arr[0]

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.permute(1, 2, 0)

        arr = arr.numpy()

    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        return None

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract first JSON-looking object from model output."""
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


class KeywordPolicyRouter:
    """Safe fallback router.

    It does not understand images.
    It chooses by matching instruction against policy_id and description.
    """

    def __init__(self, candidates: list[PolicyCandidate], default_policy_id: str | None):
        self.candidates = candidates
        self.default_policy_id = default_policy_id or (candidates[0].policy_id if candidates else "__default__")

    def select(self, instruction: str, raw_observation: dict[str, Any] | None = None) -> RouteDecision:
        text = instruction.lower().strip()

        best_policy_id = self.default_policy_id
        best_score = -1

        for c in self.candidates:
            haystack = f"{c.policy_id} {c.description}".lower()
            score = 0

            # policy_id 직접 입력하면 가장 강하게 매칭.
            if c.policy_id.lower() in text:
                score += 100

            # description 단어 매칭.
            for token in re.split(r"[\s,./|:;(){}\[\]_-]+", haystack):
                token = token.strip()
                if len(token) >= 2 and token in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_policy_id = c.policy_id

        confidence = 0.5 if best_score <= 0 else min(0.95, 0.5 + 0.05 * best_score)

        return RouteDecision(
            policy_id=best_policy_id,
            task_for_policy=instruction,
            confidence=confidence,
            reason=f"keyword_score={best_score}",
            router_model="keyword",
        )


class SmolVLMPolicyRouter:
    """SmolVLM based policy router.

    Enable with:
        export LEROBOT_USE_SMOLVLM_ROUTER=1
        export LEROBOT_ROUTER_MODEL=HuggingFaceTB/SmolVLM-256M-Instruct
    """

    def __init__(
        self,
        candidates: list[PolicyCandidate],
        default_policy_id: str | None,
        model_name_or_path: str,
        device: str = "cuda",
    ):
        self.candidates = candidates
        self.default_policy_id = default_policy_id or (candidates[0].policy_id if candidates else "__default__")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.fallback = KeywordPolicyRouter(candidates, self.default_policy_id)

        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoModel
        except Exception:
            from transformers import AutoModelForVision2Seq as AutoModel

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    def _build_prompt(self, instruction: str) -> str:
        policies_text = "\n".join(
            [
                f"- {c.policy_id}: {c.description or c.pretrained_name_or_path}"
                for c in self.candidates
            ]
        )

        return f"""
You are a robot policy router.

Your job:
Choose exactly one policy_id from the available policies.
Use the user instruction and the camera image.
If uncertain, choose {self.default_policy_id}.

Available policies:
{policies_text}

User instruction:
{instruction}

Return JSON only.
Schema:
{{
  "policy_id": "one of the available policy ids",
  "task_for_policy": "short command to send to the robot policy",
  "confidence": 0.0,
  "reason": "short reason"
}}
""".strip()

    @torch.inference_mode()
    def select(self, instruction: str, raw_observation: dict[str, Any] | None = None) -> RouteDecision:
        raw_observation = raw_observation or {}
        image = _extract_first_image(raw_observation)
        pil_image = _to_pil_image(image)

        if pil_image is None:
            return self.fallback.select(instruction, raw_observation)

        prompt = self._build_prompt(instruction)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            try:
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=text,
                    images=[pil_image],
                    return_tensors="pt",
                )
            except Exception:
                inputs = self.processor(
                    images=pil_image,
                    text=prompt,
                    return_tensors="pt",
                )

            inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )

            output_text = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

            obj = _extract_json_object(output_text)
            if obj is None:
                return self.fallback.select(instruction, raw_observation)

            policy_id = str(obj.get("policy_id", self.default_policy_id)).strip()

            valid_ids = {c.policy_id for c in self.candidates}
            if policy_id not in valid_ids:
                policy_id = self.default_policy_id

            confidence = float(obj.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return RouteDecision(
                policy_id=policy_id,
                task_for_policy=str(obj.get("task_for_policy", instruction)),
                confidence=confidence,
                reason=str(obj.get("reason", "")),
                router_model=self.model_name_or_path,
            )

        except Exception as e:
            logging.warning(f"SmolVLM router failed. Falling back to keyword router: {e}")
            return self.fallback.select(instruction, raw_observation)


def _build_router(
    candidates: list[PolicyCandidate],
    default_policy_id: str | None,
):
    use_smolvlm = os.environ.get("LEROBOT_USE_SMOLVLM_ROUTER", "0").strip() == "1"

    if not use_smolvlm:
        return KeywordPolicyRouter(candidates, default_policy_id)

    model_name_or_path = os.environ.get(
        "LEROBOT_ROUTER_MODEL",
        "HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    device = os.environ.get("LEROBOT_ROUTER_DEVICE", "cuda")

    try:
        return SmolVLMPolicyRouter(
            candidates=candidates,
            default_policy_id=default_policy_id,
            model_name_or_path=model_name_or_path,
            device=device,
        )
    except Exception as e:
        logging.warning(f"Could not initialize SmolVLM router. Using keyword router. Error: {e}")
        return KeywordPolicyRouter(candidates, default_policy_id)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config

        # Connect robot
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        # Debug robot feature keys.
        # stop-home 복귀 / action dict 매핑 확인용.
        self.logger.info(f"[debug] robot.action_features = {self.robot.action_features}")
        self.logger.info(f"[debug] robot.observation_features = {self.robot.observation_features}")

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
            rename_map=getattr(config, "rename_map", {}) or {},
        )

        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination.
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # ---------------------------------------------------------------------
        # Dynamic policy routing
        # ---------------------------------------------------------------------
        # 첫 instruction이 들어오기 전에는 control_loop가 서버로 observation을 보내지 못하게 막는다.
        # prompt_router_loop()에서 set_route()가 호출되면 set된다.
        self.route_ready_event = threading.Event()

        self.policy_candidates, manifest_default_policy_id = _load_policy_candidates_from_manifest()

        if not self.policy_candidates:
            # Manifest가 없으면 기존 단일 모델 호환.
            self.policy_candidates = [
                PolicyCandidate(
                    policy_id="__default__",
                    policy_type=config.policy_type,
                    pretrained_name_or_path=config.pretrained_name_or_path,
                    description="Default single policy",
                )
            ]
            manifest_default_policy_id = "__default__"

        self.route_lock = threading.Lock()
        self.current_policy_id = manifest_default_policy_id or self.policy_candidates[0].policy_id

        # route_ready_event가 막고 있으므로 초기 task가 있어도 바로 실행되지는 않는다.
        # 첫 instruction 입력 후 set_route()에서 실제 task_for_policy가 들어간다.
        self.current_task_text = config.task
        self.current_router_confidence = 1.0
        self.current_router_reason = "initial"
        self.current_router_model = "initial"

        self.latest_raw_observation_lock = threading.Lock()
        self.latest_raw_observation: dict[str, Any] | None = None

        self.router = _build_router(
            candidates=self.policy_candidates,
            default_policy_id=self.current_policy_id,
        )

        self.logger.info(
            f"[router] initialized | current_policy_id={self.current_policy_id} | "
            f"candidates={[c.policy_id for c in self.policy_candidates]}"
        )

        # ---------------------------------------------------------------------
        # Stop / home-return control
        # ---------------------------------------------------------------------
        self.stop_requested_event = threading.Event()
        self.home_return_running_event = threading.Event()

        self.home_return_seconds = float(os.environ.get("LEROBOT_HOME_RETURN_SECONDS", "3.0"))
        self.home_return_fps = float(os.environ.get("LEROBOT_HOME_RETURN_FPS", "25"))

        # Home pose 우선순위:
        # 1. LEROBOT_HOME_POSE_JSON 이 있으면 무조건 그 값만 사용
        # 2. 없을 때만 시작 시점 관절 위치를 home pose로 사용
        self.home_action_dict: dict[str, float] | None = None

        fixed_home_pose_path = os.environ.get("LEROBOT_HOME_POSE_JSON", "").strip()

        if fixed_home_pose_path:
            try:
                fixed_home_pose_path = os.path.expanduser(fixed_home_pose_path)

                with open(fixed_home_pose_path, "r", encoding="utf-8") as f:
                    fixed_home_pose = json.load(f)

                self.home_action_dict = {
                    str(k): float(v)
                    for k, v in fixed_home_pose.items()
                }

                self.logger.warning(
                    f"[stop-home] USING FIXED HOME POSE from {fixed_home_pose_path}: "
                    f"{self.home_action_dict}"
                )

            except Exception as e:
                self.home_action_dict = None
                self.logger.error(
                    f"[stop-home] failed to load fixed home pose from "
                    f"{fixed_home_pose_path}: {e}"
                )

        else:
            try:
                initial_obs = self.robot.get_observation()
                self._update_latest_raw_observation(initial_obs)

                self.home_action_dict = self._extract_action_dict_from_observation(initial_obs)

                if self.home_action_dict is not None:
                    self.logger.warning(
                        f"[stop-home] USING STARTUP OBSERVATION AS HOME POSE: "
                        f"{self.home_action_dict}"
                    )
                else:
                    self.logger.warning(
                        "[stop-home] failed to capture startup home pose. "
                        "Stop will pause inference, but home return may be skipped."
                    )

            except Exception as e:
                self.logger.warning(f"[stop-home] failed to capture startup home pose: {e}")

        self.logger.warning(
            f"[stop-home] home_return_seconds={self.home_return_seconds}, "
            f"home_return_fps={self.home_return_fps}, "
            f"home_action_dict={self.home_action_dict}"
        )

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def _update_latest_raw_observation(self, raw_observation: RawObservation) -> None:
        with self.latest_raw_observation_lock:
            self.latest_raw_observation = dict(raw_observation)

    def _get_latest_raw_observation(self) -> dict[str, Any] | None:
        with self.latest_raw_observation_lock:
            if self.latest_raw_observation is None:
                return None
            return dict(self.latest_raw_observation)

    def _flush_action_queue_for_policy_switch(self) -> None:
        """Drop old action chunks when policy changes.

        Without this, actions generated by the previous policy may continue executing
        for several control steps.
        """
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_chunk_size = -1

        self.must_go.set()
        self.logger.info("[router] action queue flushed due to policy/task switch")

    @staticmethod
    def _to_float_scalar(value: Any) -> float | None:
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

    def _extract_action_dict_from_observation(
        self,
        raw_observation: dict[str, Any],
    ) -> dict[str, float] | None:
        """Extract current robot joint/action values from raw observation.

        robot.send_action() expects keys from self.robot.action_features.
        We try:
          1. raw_observation[action_key]
          2. raw_observation["observation.state"] vector
          3. raw_observation["state"] vector
        """
        action_keys = list(self.robot.action_features)

        action_dict: dict[str, float] = {}
        missing_keys: list[str] = []

        # Case 1: each action feature exists directly in observation.
        for key in action_keys:
            candidates = [
                key,
                f"observation.{key}",
                f"observation.state.{key}",
                key.replace("action.", "observation."),
            ]

            found = False
            for cand in candidates:
                if cand in raw_observation:
                    value = self._to_float_scalar(raw_observation[cand])
                    if value is not None:
                        action_dict[key] = value
                        found = True
                        break

            if not found:
                missing_keys.append(key)

        if len(action_dict) == len(action_keys):
            return action_dict

        # Case 2: state vector exists.
        for state_key in ["observation.state", "state"]:
            if state_key not in raw_observation:
                continue

            state = raw_observation[state_key]

            try:
                if torch.is_tensor(state):
                    state_list = state.detach().cpu().flatten().tolist()
                else:
                    state_list = list(state)

                if len(state_list) >= len(action_keys):
                    return {
                        key: float(state_list[i])
                        for i, key in enumerate(action_keys)
                    }
            except Exception:
                pass

        self.logger.warning(
            f"[stop-home] could not map observation to action dict. "
            f"action_keys={action_keys}, missing={missing_keys}, "
            f"available_obs_keys={sorted(list(raw_observation.keys()))}"
        )
        return None

    def _request_stop_and_home(self) -> None:
        """Stop current inference/action stream and request slow return to home."""
        self.logger.warning("[stop-home] STOP requested")

        # 새 정책 실행 준비 상태를 내린다.
        # control_loop는 이 값이 내려가면 새 observation을 서버로 보내지 않는다.
        self.route_ready_event.clear()

        # 수신/대기 중인 action chunk 폐기.
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_chunk_size = -1

        self.must_go.clear()
        self.stop_requested_event.set()

    def _run_home_return_blocking(self) -> None:
        """Slowly return robot to startup home pose.

        This runs inside control_loop thread.
        While this runs:
          - policy inference is paused
          - received action chunks are discarded
          - robot is commanded directly with interpolated joint values
        """
        if self.home_return_running_event.is_set():
            return

        self.home_return_running_event.set()

        try:
            if self.home_action_dict is None:
                self.logger.error(
                    "[stop-home] home_action_dict is None. "
                    "Cannot return home. Inference remains paused."
                )
                return

            with self.action_queue_lock:
                self.action_queue = Queue()
                self.action_chunk_size = -1

            current_obs = self.robot.get_observation()
            current_action_dict = self._extract_action_dict_from_observation(current_obs)

            if current_action_dict is None:
                self.logger.error(
                    "[stop-home] could not read current joint state. "
                    "Cannot interpolate to home."
                )
                return

            action_keys = list(self.robot.action_features)
            steps = max(1, int(self.home_return_seconds * self.home_return_fps))
            dt = 1.0 / max(1.0, self.home_return_fps)

            self.logger.warning(
                f"[stop-home] returning home slowly | "
                f"seconds={self.home_return_seconds}, steps={steps}"
            )
            self.logger.warning(
                f"[stop-home] HOME TARGET DICT = {self.home_action_dict}"
            )
            self.logger.warning(
                f"[stop-home] CURRENT START DICT = {current_action_dict}"
            )

            for step in range(steps):
                if not self.running:
                    break

                alpha = (step + 1) / steps

                action = {}
                for key in action_keys:
                    start = float(current_action_dict[key])
                    target = float(self.home_action_dict[key])
                    action[key] = start + alpha * (target - start)

                self.robot.send_action(action)
                time.sleep(dt)

            self.logger.warning("[stop-home] home return finished. Waiting for next instruction.")

        finally:
            self.stop_requested_event.clear()
            self.home_return_running_event.clear()
            self.must_go.set()

    def set_route(self, route: RouteDecision) -> None:
        with self.route_lock:
            old_policy_id = self.current_policy_id
            old_task_text = self.current_task_text

            self.current_policy_id = route.policy_id
            self.current_task_text = route.task_for_policy
            self.current_router_confidence = route.confidence
            self.current_router_reason = route.reason
            self.current_router_model = route.router_model

        if old_policy_id != route.policy_id or old_task_text != route.task_for_policy:
            self._flush_action_queue_for_policy_switch()

        self.route_ready_event.set()

        self.logger.info(
            f"[router] selected policy_id={route.policy_id} | "
            f"confidence={route.confidence:.3f} | "
            f"task={route.task_for_policy} | "
            f"reason={route.reason}"
        )

    def prompt_router_loop(self):
        """Keyboard instruction loop.

        Type an instruction while the robot is running.
        The router selects a policy_id and updates task text dynamically.

        Commands:
          :q / quit / exit -> stop client
          :policies -> print available policies
        """
        self.logger.info("[router] prompt loop started. Type instruction and press Enter.")

        while self.running:
            try:
                user_instruction = input("\n[router] instruction> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                self.shutdown_event.set()
                break

            if not user_instruction:
                continue

            if user_instruction in {":q", "quit", "exit"}:
                self.shutdown_event.set()
                break

            if user_instruction.lower() in {"stop", "정지", "멈춰", "home", "홈"}:
                self._request_stop_and_home()
                continue

            if user_instruction == ":policies":
                print("\nAvailable policies:")
                for c in self.policy_candidates:
                    print(f"  - {c.policy_id}: {c.description}")
                continue

            latest_obs = self._get_latest_raw_observation()
            route = self.router.select(user_instruction, latest_obs)
            self.set_route(route)

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Stop/home 중이거나 첫 instruction 대기 중이면 서버에서 온 action chunk를 버린다.
                if self.stop_requested_event.is_set() or self.home_return_running_event.is_set() or not self.route_ready_event.is_set():
                    self.logger.warning("[stop-home] discarding received action chunk")
                    continue

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()

            # Keep latest camera frame for SmolVLM router.
            self._update_latest_raw_observation(raw_observation)

            # Read current dynamic route.
            with self.route_lock:
                current_task_text = self.current_task_text or task
                current_policy_id = self.current_policy_id
                current_router_confidence = self.current_router_confidence
                current_router_reason = self.current_router_reason
                current_router_model = self.current_router_model

            # VLA policy still receives task text.
            raw_observation["task"] = current_task_text

            # Server uses this to select already-loaded policy slot.
            raw_observation["policy_id"] = current_policy_id
            raw_observation["router_confidence"] = current_router_confidence
            raw_observation["router_reason"] = current_router_reason
            raw_observation["router_model"] = current_router_model

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing.
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(
                f"QUEUE SIZE: {current_queue_size} "
                f"(Must go: {observation.must_go}, policy_id={current_policy_id})"
            )

            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Policy: {current_policy_id} | "
                    f"Task: {current_task_text} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | "
                    f"Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            raise

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()

        self.logger.info(
            "[router] Waiting for first instruction. "
            "Type at [router] instruction> to start policy execution."
        )

        while self.running and not self.route_ready_event.is_set():
            time.sleep(0.05)

        if not self.running:
            return None, None

        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            # stop 명령이 들어오면 정책 action 실행/observation 송신을 모두 멈추고
            # home pose로 천천히 복귀한다.
            if self.stop_requested_event.is_set():
                self._run_home_return_blocking()

                self.logger.info(
                    "[router] Waiting for next instruction after stop. "
                    "Type at [router] instruction> to resume."
                )

                while self.running and not self.route_ready_event.is_set():
                    time.sleep(0.05)

                continue

            # 첫 instruction 전이거나 stop 후 다음 instruction 대기 중이면 아무것도 보내지 않는다.
            if not self.route_ready_event.is_set():
                time.sleep(0.05)
                continue

            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Keyboard/VLM router thread.
        prompt_router_thread = threading.Thread(target=client.prompt_router_loop, daemon=True)

        # Start threads
        action_receiver_thread.start()
        prompt_router_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
