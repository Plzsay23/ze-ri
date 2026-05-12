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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import json
import logging
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


@dataclass
class LoadedPolicySlot:
    policy_id: str
    policy_type: str
    pretrained_name_or_path: str
    device: str
    actions_per_chunk: int
    policy: Any
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions.
        # 기존 단일 policy 호환 필드.
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        # Multi-policy slots.
        # LEROBOT_POLICY_MANIFEST가 있으면 여기에 여러 정책을 모두 VRAM에 올린다.
        self.policies: dict[str, LoadedPolicySlot] = {}
        self.default_policy_id: str | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        if self.policy is None:
            return None
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def _read_policy_manifest(self) -> dict[str, Any] | None:
        """Read multi-policy manifest from LEROBOT_POLICY_MANIFEST.

        If env is not set, server behaves exactly like the original single-policy server.
        """
        manifest_path = os.environ.get("LEROBOT_POLICY_MANIFEST", "").strip()
        if not manifest_path:
            return None

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"LEROBOT_POLICY_MANIFEST does not exist: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "policies" not in manifest or not isinstance(manifest["policies"], list):
            raise ValueError("Policy manifest must contain a list field: policies")

        if len(manifest["policies"]) == 0:
            raise ValueError("Policy manifest policies list is empty")

        return manifest

    def _load_policy_slot(
        self,
        *,
        policy_id: str,
        policy_type: str,
        pretrained_name_or_path: str,
        device: str,
        actions_per_chunk: int,
        rename_map: dict[str, str] | None,
    ) -> LoadedPolicySlot:
        """Load one policy and its processors to GPU/target device."""
        if policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_type} not supported. Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"[multi-policy] Loading policy_id={policy_id} | "
            f"type={policy_type} | path={pretrained_name_or_path} | device={device}"
        )

        policy_class = get_policy_class(policy_type)

        start = time.perf_counter()
        policy = policy_class.from_pretrained(pretrained_name_or_path)
        policy.to(device)
        policy.eval()

        device_override = {"device": device}
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": rename_map or {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        elapsed = time.perf_counter() - start

        self.logger.info(
            f"[multi-policy] Loaded policy_id={policy_id} to {device} in {elapsed:.4f}s"
        )

        return LoadedPolicySlot(
            policy_id=policy_id,
            policy_type=policy_type,
            pretrained_name_or_path=pretrained_name_or_path,
            device=device,
            actions_per_chunk=actions_per_chunk,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

    def _setup_policy_slots(self, policy_specs: RemotePolicyConfig) -> None:
        """Setup single-policy or multi-policy mode.

        Multi-policy mode is enabled by env:
            LEROBOT_POLICY_MANIFEST=/path/to/xvla_manifest.json

        The client still sends RemotePolicyConfig once because the existing gRPC API expects it.
        We reuse its lerobot_features, actions_per_chunk, device, and rename_map.
        """
        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        rename_map = getattr(policy_specs, "rename_map", {}) or {}

        manifest = self._read_policy_manifest()

        self.policies.clear()

        if manifest is None:
            # Original behavior, but stored as a single slot.
            slot = self._load_policy_slot(
                policy_id="__default__",
                policy_type=policy_specs.policy_type,
                pretrained_name_or_path=policy_specs.pretrained_name_or_path,
                device=policy_specs.device,
                actions_per_chunk=policy_specs.actions_per_chunk,
                rename_map=rename_map,
            )
            self.policies[slot.policy_id] = slot
            self.default_policy_id = slot.policy_id

        else:
            default_policy_id = manifest.get("default_policy_id")
            policies = manifest["policies"]

            for entry in policies:
                policy_id = entry["id"]
                policy_type = entry.get("policy_type", policy_specs.policy_type)
                pretrained_name_or_path = entry["pretrained_name_or_path"]
                device = entry.get("device", policy_specs.device)
                actions_per_chunk = int(entry.get("actions_per_chunk", policy_specs.actions_per_chunk))

                if policy_id in self.policies:
                    raise ValueError(f"Duplicated policy id in manifest: {policy_id}")

                slot = self._load_policy_slot(
                    policy_id=policy_id,
                    policy_type=policy_type,
                    pretrained_name_or_path=pretrained_name_or_path,
                    device=device,
                    actions_per_chunk=actions_per_chunk,
                    rename_map=rename_map,
                )
                self.policies[policy_id] = slot

            if default_policy_id is None:
                default_policy_id = policies[0]["id"]

            if default_policy_id not in self.policies:
                raise ValueError(
                    f"default_policy_id={default_policy_id} is not in loaded policies: "
                    f"{list(self.policies.keys())}"
                )

            self.default_policy_id = default_policy_id

        # Keep legacy fields pointing to default slot.
        default_slot = self.policies[self.default_policy_id]
        self.policy = default_slot.policy
        self.preprocessor = default_slot.preprocessor
        self.postprocessor = default_slot.postprocessor

        self.logger.info(
            f"[multi-policy] Ready. default_policy_id={self.default_policy_id} | "
            f"loaded={list(self.policies.keys())}"
        )

    def _select_policy_slot(self, observation_t: TimedObservation) -> LoadedPolicySlot:
        """Select policy by observation['policy_id'].

        If missing or invalid, fall back to default_policy_id.
        """
        if not self.policies:
            raise RuntimeError("No policy loaded. Did client call SendPolicyInstructions?")

        raw_obs = observation_t.get_observation()
        requested_policy_id = raw_obs.get("policy_id", None)

        policy_id = requested_policy_id or self.default_policy_id

        if policy_id not in self.policies:
            self.logger.warning(
                f"[multi-policy] Unknown policy_id={policy_id}. "
                f"Fallback to default_policy_id={self.default_policy_id}."
            )
            policy_id = self.default_policy_id

        return self.policies[policy_id]

    @staticmethod
    def _strip_routing_keys(raw_observation: dict[str, Any]) -> dict[str, Any]:
        """Remove router-only keys before LeRobot preprocessor.

        'task' must remain because VLA policies usually consume it.
        """
        obs = dict(raw_observation)
        for key in [
            "policy_id",
            "router_confidence",
            "router_reason",
            "router_model",
        ]:
            obs.pop(key, None)
        return obs

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        start = time.perf_counter()
        self._setup_policy_slots(policy_specs)
        end = time.perf_counter()

        self.logger.info(
            f"Time taken to setup policy slot(s) on device(s): {end - start:.4f} seconds"
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""

        current_raw = obs.get_observation()
        previous_raw = previous_obs.get_observation()

        # 정책이나 task가 바뀌면 이미지가 비슷해도 반드시 새 inference를 돌려야 한다.
        if current_raw.get("policy_id") != previous_raw.get("policy_id"):
            self.logger.info(
                f"[multi-policy] policy_id changed: "
                f"{previous_raw.get('policy_id')} -> {current_raw.get('policy_id')}"
            )
            return True

        if current_raw.get("task") != previous_raw.get("task"):
            self.logger.info("[multi-policy] task text changed. Force processing.")
            return True

        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(
        self,
        slot: LoadedPolicySlot,
        observation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get an action chunk from the selected policy slot."""
        chunk = slot.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # now shape is (B, chunk_size, action_dim)

        return chunk[:, : slot.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        slot = self._select_policy_slot(observation_t)

        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        raw_observation = self._strip_routing_keys(observation_t.get_observation())

        observation: Observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            slot.policy.config.image_features,
            camera_key_map=getattr(slot.policy.config, "camera_key_map", None),
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = slot.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(slot, observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"[multi-policy] policy_id={slot.policy_id} | "
            f"Preprocessing and inference took {inference_time:.4f}s, "
            f"action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = slot.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
