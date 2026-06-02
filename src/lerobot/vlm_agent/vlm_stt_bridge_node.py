#!/usr/bin/env python3
# vlm_stt_bridge_node.py
# Thin ROS2 node wrapper for the refactored Ze-Ri VLM/STT bridge.

import queue
import threading
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32, Int32, String

try:
    from .zeri_vlm_constants import (
        LED_CYAN, LED_MAGENTA, LED_OFF, LED_WHITE, LED_YELLOW,
        MISSION_SEARCH_PERSON, VALID_MISSION_STATES,
    )
    from .zeri_vlm_decision import clamp_led_cmd
    from .zeri_vlm_mission_mixin import ZeriVLMMissionMixin
    from .zeri_vlm_pipeline_mixin import ZeriVLMPipelineMixin
    from .zeri_vlm_publish_mixin import ZeriVLMPublishMixin
    from .zeri_vlm_qos import make_reliable_qos, make_sensor_qos
    from .zeri_vlm_runner import QwenVLMRunner
    from .zeri_vlm_stt_frame_mixin import ZeriVLMSTTFrameMixin
    from .zeri_vlm_types import VLMRequest
    from .zeri_vlm_worker_mixin import ZeriVLMWorkerMixin
except ImportError:
    from zeri_vlm_constants import (
        LED_CYAN, LED_MAGENTA, LED_OFF, LED_WHITE, LED_YELLOW,
        MISSION_SEARCH_PERSON, VALID_MISSION_STATES,
    )
    from zeri_vlm_decision import clamp_led_cmd
    from zeri_vlm_mission_mixin import ZeriVLMMissionMixin
    from zeri_vlm_pipeline_mixin import ZeriVLMPipelineMixin
    from zeri_vlm_publish_mixin import ZeriVLMPublishMixin
    from zeri_vlm_qos import make_reliable_qos, make_sensor_qos
    from zeri_vlm_runner import QwenVLMRunner
    from zeri_vlm_stt_frame_mixin import ZeriVLMSTTFrameMixin
    from zeri_vlm_types import VLMRequest
    from zeri_vlm_worker_mixin import ZeriVLMWorkerMixin


class ZeriVLMSTTBridgeNode(
    Node,
    ZeriVLMPublishMixin,
    ZeriVLMMissionMixin,
    ZeriVLMPipelineMixin,
    ZeriVLMSTTFrameMixin,
    ZeriVLMWorkerMixin,
):
    def __init__(self):
        super().__init__("zeri_vlm_stt_bridge_node")

        self.declare_parameter("rgb_topic", "/zeri/top/rgb/image_raw")
        self.declare_parameter("depth_topic", "/zeri/top/depth/image_raw")
        self.declare_parameter("stt_topic", "/stt/text")

        self.declare_parameter("decision_topic", "/zeri/vlm/decision")
        self.declare_parameter("robot_speech_topic", "/zeri/vlm/robot_speech")
        self.declare_parameter("vlm_input_rgb_topic", "/zeri/vlm/input_rgb")
        self.declare_parameter("vlm_input_depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("inference_status_topic", "/zeri/vlm/inference_status")

        self.declare_parameter("mission_event_topic", "/zeri/mission/event")
        self.declare_parameter("mission_state_topic", "/zeri/mission/state")
        self.declare_parameter("nav_intent_topic", "/zeri/nav/intent")
        self.declare_parameter("target_context_topic", "/zeri/person/target")
        self.declare_parameter("map_marker_topic", "/zeri/map/person_marker")
        self.declare_parameter("arm_home_request_topic", "/zeri/arm/home_request")
        self.declare_parameter("enable_mission_events", True)
        self.declare_parameter("verify_handoff_after_vla_success", True)
        self.declare_parameter("initial_mission_state", MISSION_SEARCH_PERSON)

        self.declare_parameter("tts_status_topic", "/zeri/tts/status")
        self.declare_parameter("stt_mute_topic", "/zeri/stt/mute")

        self.declare_parameter("led_topic", "/zeri/led/cmd")
        self.declare_parameter("led_on_startup", LED_WHITE)
        self.declare_parameter("led_on_loading", LED_CYAN)
        self.declare_parameter("led_on_inference", LED_CYAN)
        self.declare_parameter("led_on_vla_running", LED_MAGENTA)
        self.declare_parameter("led_on_vla_success", LED_WHITE)
        self.declare_parameter("led_on_error", LED_YELLOW)
        self.declare_parameter("led_on_shutdown", LED_OFF)

        self.declare_parameter("vla_task_request_topic", "/zeri/vla/task_request")
        self.declare_parameter("vla_status_topic", "/zeri/vla/status")
        self.declare_parameter("enable_vla", True)
        self.declare_parameter("vla_timeout_sec", 60.0)
        self.declare_parameter("vla_default_task_duration_sec", 20.0)

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("use_vad_gate", False)
        self.declare_parameter("vad_hold_sec", 1.2)

        self.declare_parameter("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("max_new_tokens", 192)
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("queue_size", 4)

        self.declare_parameter("stt_gate_mode", "wake")
        self.declare_parameter("stt_min_chars", 3)
        self.declare_parameter("wake_listen_window_sec", 8.0)
        self.declare_parameter("min_inference_interval_sec", 2.0)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("wake_words", ["제리", "제리야"])

        self.declare_parameter("stt_block_after_tts_sec", 0.8)
        self.declare_parameter("tts_max_wait_sec", 20.0)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.stt_topic = str(self.get_parameter("stt_topic").value)

        self.decision_topic = str(self.get_parameter("decision_topic").value)
        self.robot_speech_topic = str(self.get_parameter("robot_speech_topic").value)
        self.vlm_input_rgb_topic = str(self.get_parameter("vlm_input_rgb_topic").value)
        self.vlm_input_depth_topic = str(self.get_parameter("vlm_input_depth_topic").value)
        self.inference_status_topic = str(self.get_parameter("inference_status_topic").value)

        self.mission_event_topic = str(self.get_parameter("mission_event_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.nav_intent_topic = str(self.get_parameter("nav_intent_topic").value)
        self.target_context_topic = str(self.get_parameter("target_context_topic").value)
        self.map_marker_topic = str(self.get_parameter("map_marker_topic").value)
        self.arm_home_request_topic = str(
            self.get_parameter("arm_home_request_topic").value
        )
        self.enable_mission_events = bool(
            self.get_parameter("enable_mission_events").value
        )
        self.verify_handoff_after_vla_success = bool(
            self.get_parameter("verify_handoff_after_vla_success").value
        )
        initial_mission_state = str(
            self.get_parameter("initial_mission_state").value
        ).upper()
        if initial_mission_state not in VALID_MISSION_STATES:
            initial_mission_state = MISSION_SEARCH_PERSON
        self.mission_state = initial_mission_state

        self.tts_status_topic = str(self.get_parameter("tts_status_topic").value)
        self.stt_mute_topic = str(self.get_parameter("stt_mute_topic").value)

        self.led_topic = str(self.get_parameter("led_topic").value)
        self.led_on_startup = clamp_led_cmd(
            self.get_parameter("led_on_startup").value,
            LED_WHITE,
        )
        self.led_on_loading = clamp_led_cmd(
            self.get_parameter("led_on_loading").value,
            LED_CYAN,
        )
        self.led_on_inference = clamp_led_cmd(
            self.get_parameter("led_on_inference").value,
            LED_CYAN,
        )
        self.led_on_vla_running = clamp_led_cmd(
            self.get_parameter("led_on_vla_running").value,
            LED_MAGENTA,
        )
        self.led_on_vla_success = clamp_led_cmd(
            self.get_parameter("led_on_vla_success").value,
            LED_WHITE,
        )
        self.led_on_error = clamp_led_cmd(
            self.get_parameter("led_on_error").value,
            LED_YELLOW,
        )
        self.led_on_shutdown = clamp_led_cmd(
            self.get_parameter("led_on_shutdown").value,
            LED_OFF,
        )

        self.vla_task_request_topic = str(
            self.get_parameter("vla_task_request_topic").value
        )
        self.vla_status_topic = str(self.get_parameter("vla_status_topic").value)
        self.enable_vla = bool(self.get_parameter("enable_vla").value)
        self.vla_timeout_sec = float(self.get_parameter("vla_timeout_sec").value)
        self.vla_default_task_duration_sec = float(
            self.get_parameter("vla_default_task_duration_sec").value
        )

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.use_vad_gate = bool(self.get_parameter("use_vad_gate").value)
        self.vad_hold_sec = float(self.get_parameter("vad_hold_sec").value)

        model_id = str(self.get_parameter("model_id").value)
        dtype = str(self.get_parameter("dtype").value)
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        queue_size = int(self.get_parameter("queue_size").value)
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )

        self.stt_gate_mode = str(self.get_parameter("stt_gate_mode").value)
        self.stt_min_chars = int(self.get_parameter("stt_min_chars").value)
        self.wake_listen_window_sec = float(
            self.get_parameter("wake_listen_window_sec").value
        )
        self.min_inference_interval_sec = float(
            self.get_parameter("min_inference_interval_sec").value
        )
        self.duplicate_window_sec = float(
            self.get_parameter("duplicate_window_sec").value
        )

        self.stt_block_after_tts_sec = float(
            self.get_parameter("stt_block_after_tts_sec").value
        )
        self.tts_max_wait_sec = float(self.get_parameter("tts_max_wait_sec").value)

        wake_words_param = self.get_parameter("wake_words").value
        self.wake_words = [
            str(word).strip()
            for word in wake_words_param
            if str(word).strip()
        ]

        self.ignore_phrases = {
            "어",
            "음",
            "아",
            "네",
            "예",
            "응",
            "테스트",
            "마이크 테스트",
            "안녕하세요",
        }

        self.latest_target_context: Dict[str, Any] = {}
        self.latest_target_context_time = 0.0
        self.latest_mission_event: Dict[str, Any] = {}
        self.latest_mission_event_time = 0.0
        self.last_decision_context: Dict[str, Any] = {}

        self.last_accepted_text = ""
        self.last_accepted_time = 0.0
        self.last_inference_request_time = 0.0
        self.wake_active_until = 0.0

        self.pipeline_lock = threading.Lock()
        self.pipeline_busy = False

        self.waiting_for_tts = False
        self.tts_active = False
        self.stt_block_until = 0.0
        self.tts_deadline = 0.0
        self.stt_mute_state = False

        self.waiting_for_vla = False
        self.vla_active = False
        self.vla_deadline = 0.0
        self.active_vla_task_id: Optional[str] = None

        self.latest_vad = False
        self.latest_vad_time = 0.0
        self.latest_doa_deg: Optional[float] = None
        self.latest_doa_time = 0.0

        self.text_queue: queue.Queue[VLMRequest] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        self.frame_lock = threading.Lock()
        self.latest_rgb_msg: Optional[RosImage] = None
        self.latest_depth_msg: Optional[RosImage] = None
        self.latest_rgb_time: Optional[float] = None
        self.latest_depth_time: Optional[float] = None

        sensor_qos = make_sensor_qos(depth=5)
        reliable_qos = make_reliable_qos(depth=10)

        self.rgb_sub = self.create_subscription(
            RosImage,
            self.rgb_topic,
            self.rgb_callback,
            sensor_qos,
        )

        self.depth_sub = self.create_subscription(
            RosImage,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )

        self.stt_sub = self.create_subscription(
            String,
            self.stt_topic,
            self.stt_callback,
            reliable_qos,
        )

        self.mission_event_sub = self.create_subscription(
            String,
            self.mission_event_topic,
            self.mission_event_callback,
            reliable_qos,
        )

        self.target_context_sub = self.create_subscription(
            String,
            self.target_context_topic,
            self.target_context_callback,
            reliable_qos,
        )

        self.tts_status_sub = self.create_subscription(
            String,
            self.tts_status_topic,
            self.tts_status_callback,
            reliable_qos,
        )

        self.vla_status_sub = self.create_subscription(
            String,
            self.vla_status_topic,
            self.vla_status_callback,
            reliable_qos,
        )

        self.vad_sub = self.create_subscription(
            Bool,
            self.vad_topic,
            self.vad_callback,
            reliable_qos,
        )

        self.doa_sub = self.create_subscription(
            Float32,
            self.doa_topic,
            self.doa_callback,
            reliable_qos,
        )

        self.decision_publisher = self.create_publisher(
            String,
            self.decision_topic,
            reliable_qos,
        )

        self.robot_speech_publisher = self.create_publisher(
            String,
            self.robot_speech_topic,
            reliable_qos,
        )

        self.led_publisher = self.create_publisher(
            Int32,
            self.led_topic,
            reliable_qos,
        )

        self.vla_task_request_publisher = self.create_publisher(
            String,
            self.vla_task_request_topic,
            reliable_qos,
        )

        self.vlm_input_rgb_publisher = self.create_publisher(
            RosImage,
            self.vlm_input_rgb_topic,
            reliable_qos,
        )

        self.vlm_input_depth_publisher = self.create_publisher(
            RosImage,
            self.vlm_input_depth_topic,
            reliable_qos,
        )

        self.inference_status_publisher = self.create_publisher(
            String,
            self.inference_status_topic,
            reliable_qos,
        )

        self.stt_mute_publisher = self.create_publisher(
            Bool,
            self.stt_mute_topic,
            reliable_qos,
        )

        self.mission_state_publisher = self.create_publisher(
            String,
            self.mission_state_topic,
            reliable_qos,
        )

        self.nav_intent_publisher = self.create_publisher(
            String,
            self.nav_intent_topic,
            reliable_qos,
        )

        self.map_marker_publisher = self.create_publisher(
            String,
            self.map_marker_topic,
            reliable_qos,
        )

        self.arm_home_request_publisher = self.create_publisher(
            Bool,
            self.arm_home_request_topic,
            reliable_qos,
        )

        self.pipeline_timer = self.create_timer(
            0.2,
            self.pipeline_timer_callback,
        )

        self.get_logger().info("Zeri VLM-STT bridge node subscriptions:")
        self.get_logger().info(f"  RGB input:         {self.rgb_topic}")
        self.get_logger().info(f"  Depth input:       {self.depth_topic}")
        self.get_logger().info(f"  STT input:         {self.stt_topic}")
        self.get_logger().info(f"  TTS status:        {self.tts_status_topic}")
        self.get_logger().info(f"  VLA status:        {self.vla_status_topic}")
        self.get_logger().info(f"  VAD input:         {self.vad_topic}")
        self.get_logger().info(f"  DOA input:         {self.doa_topic}")
        self.get_logger().info(f"  Mission event:     {self.mission_event_topic}")
        self.get_logger().info(f"  Target context:    {self.target_context_topic}")

        self.get_logger().info("Zeri VLM-STT bridge node publishers:")
        self.get_logger().info(f"  Decision:          {self.decision_topic}")
        self.get_logger().info(f"  Robot speech:      {self.robot_speech_topic}")
        self.get_logger().info(f"  LED command:       {self.led_topic}")
        self.get_logger().info(f"  VLA task request:  {self.vla_task_request_topic}")
        self.get_logger().info(f"  VLM RGB snap:      {self.vlm_input_rgb_topic}")
        self.get_logger().info(f"  VLM Depth snap:    {self.vlm_input_depth_topic}")
        self.get_logger().info(f"  VLM status:        {self.inference_status_topic}")
        self.get_logger().info(f"  STT mute:          {self.stt_mute_topic}")
        self.get_logger().info(f"  Mission state:     {self.mission_state_topic}")
        self.get_logger().info(f"  Nav intent:        {self.nav_intent_topic}")
        self.get_logger().info(f"  Map marker:        {self.map_marker_topic}")
        self.get_logger().info(f"  Arm home request:  {self.arm_home_request_topic}")

        self.get_logger().info("Runtime settings:")
        self.get_logger().info(f"  enable_vla: {self.enable_vla}")
        self.get_logger().info(f"  vla_timeout_sec: {self.vla_timeout_sec}")
        self.get_logger().info(
            f"  vla_default_task_duration_sec: {self.vla_default_task_duration_sec}"
        )
        self.get_logger().info(f"  stt_gate_mode: {self.stt_gate_mode}")
        self.get_logger().info(f"  wake_words: {self.wake_words}")
        self.get_logger().info(
            f"  wake_listen_window_sec: {self.wake_listen_window_sec}"
        )
        self.get_logger().info(f"  use_vad_gate: {self.use_vad_gate}")
        self.get_logger().info(f"  vad_hold_sec: {self.vad_hold_sec}")
        self.get_logger().info(
            f"  stt_block_after_tts_sec: {self.stt_block_after_tts_sec}"
        )
        self.get_logger().info(f"  tts_max_wait_sec: {self.tts_max_wait_sec}")
        self.get_logger().info(f"  initial_mission_state: {self.mission_state}")
        self.get_logger().info(
            f"  verify_handoff_after_vla_success: "
            f"{self.verify_handoff_after_vla_success}"
        )

        self.publish_stt_mute(False)
        self.publish_mission_state(self.mission_state, reason="startup")

        self.publish_led(self.led_on_loading)
        self.publish_inference_status("loading_vlm_model")
        self.get_logger().info(f"Loading VLM model: {model_id}")

        self.vlm = QwenVLMRunner(
            model_id=model_id,
            dtype_name=dtype,
            max_new_tokens=max_new_tokens,
            default_task_duration_sec=self.vla_default_task_duration_sec,
        )

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

        self.publish_led(self.led_on_startup)
        self.publish_inference_status("waiting_for_camera_frame")
        self.get_logger().info("Zeri VLM-STT bridge node is ready.")


    def destroy_node(self) -> None:
        self.get_logger().info("Stopping VLM-STT bridge node.")

        try:
            self.publish_inference_status("shutting_down")
            self.publish_stt_mute(False)
            self.publish_led(self.led_on_shutdown)
        except Exception:
            pass

        self.stop_event.set()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)

        super().destroy_node()



def main() -> None:
    rclpy.init()

    node: Optional[ZeriVLMSTTBridgeNode] = None

    try:
        node = ZeriVLMSTTBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
