#!/usr/bin/env python3
# src/lerobot/vlm_agent/vlm_stt_bridge_node.py

import copy
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.

현재 시스템에는 실제 로봇 정책 모델이 아직 없다.
그러나 산소마스크 전달 정책 모델이 존재한다고 가정한다.

입력:
- top-view RGB 카메라 이미지
- STT에서 들어온 사용자 텍스트

목표:
- 산소마스크 전달이 필요한 상황인지 판단한다.
- 필요하면 adapter_id를 "oxygen_mask_delivery_lora"로 선택한다.
- 필요하지 않으면 adapter_id를 "idle_lora"로 선택한다.
- 로봇이 사람에게 말할 짧은 발화문도 생성한다.
- 반드시 JSON으로만 답한다.
- JSON 바깥 문장은 절대 쓰지 않는다.

판단 기준:
- "숨쉬기 힘들다", "숨을 못 쉬겠다", "산소가 필요하다", "질식할 것 같다",
  "가스 냄새가 난다", "연기 때문에 숨을 못 쉬겠다" 등은 산소마스크 전달 필요.
- 장면에 사람이 있고 연기/화재/유독가스 위험이 추정되면 산소마스크 전달 필요.
- 장면이 애매해도 사용자 텍스트가 호흡 곤란이면 산소마스크 전달 필요.
- 일반 대화, 인사, 위험 없음이면 idle.

출력 JSON schema:
{
  "need_oxygen_mask": true,
  "selected_task": "oxygen_mask_delivery|idle",
  "adapter_id": "oxygen_mask_delivery_lora|idle_lora",
  "confidence": 0.0,
  "reason": "짧은 한국어 이유",
  "robot_speech": "로봇이 사람에게 말할 짧은 한국어 문장"
}

robot_speech 규칙:
- 산소마스크 전달이 필요하면 안심시키는 문장으로 말한다.
  예: "여기 산소마스크입니다. 천천히 호흡하세요."
- 산소마스크 전달이 필요하지 않으면 짧게 대기 상태를 말한다.
  예: "현재 산소마스크 전달은 필요하지 않습니다. 계속 상황을 확인하겠습니다."
- robot_speech는 TTS로 읽을 수 있도록 한 문장 또는 두 문장 이내로 짧게 작성한다.
"""


USER_PROMPT_TEMPLATE = """
현재 카메라 장면과 STT 텍스트를 보고 산소마스크 전달 모델을 실행해야 하는지 판단해라.

STT 텍스트:
"{stt_text}"

JSON으로만 답해라.
"""


@dataclass
class VLMDecision:
    need_oxygen_mask: bool
    selected_task: str
    adapter_id: str
    confidence: float
    reason: str
    robot_speech: str
    raw_text: str


def make_sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    return None


def normalize_decision(parsed: Optional[Dict[str, Any]], raw_text: str) -> VLMDecision:
    if parsed is None:
        return VLMDecision(
            need_oxygen_mask=False,
            selected_task="idle",
            adapter_id="idle_lora",
            confidence=0.0,
            reason="VLM JSON parsing failed.",
            robot_speech="현재 판단 결과를 해석하지 못했습니다. 다시 말씀해 주세요.",
            raw_text=raw_text,
        )

    need = bool(parsed.get("need_oxygen_mask", False))

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."

    robot_speech = str(parsed.get("robot_speech", "")).strip()

    if need:
        selected_task = "oxygen_mask_delivery"
        adapter_id = "oxygen_mask_delivery_lora"

        if not robot_speech:
            robot_speech = "여기 산소마스크입니다. 천천히 호흡하세요."
    else:
        selected_task = "idle"
        adapter_id = "idle_lora"

        if not robot_speech:
            robot_speech = "현재 산소마스크 전달은 필요하지 않습니다. 계속 상황을 확인하겠습니다."

    return VLMDecision(
        need_oxygen_mask=need,
        selected_task=selected_task,
        adapter_id=adapter_id,
        confidence=confidence,
        reason=reason,
        robot_speech=robot_speech,
        raw_text=raw_text,
    )


def ros_image_to_bgr(msg: RosImage) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("bgr8", "rgb8"):
            channels = 3
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            return arr.copy()

        if encoding in ("bgra8", "rgba8"):
            channels = 4
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgba8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr.copy()

        if encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            arr = arr.reshape((height, row_pixels))[:, :width]
            return cv2.cvtColor(arr.copy(), cv2.COLOR_GRAY2BGR)

    except Exception:
        return None

    return None


def ros_image_to_pil_rgb(msg: RosImage) -> Optional[PILImage.Image]:
    bgr = ros_image_to_bgr(msg)

    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def pil_rgb_to_ros_image(
    image: PILImage.Image,
    stamp,
    frame_id: str = "zeri_vlm_input_rgb",
) -> RosImage:
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.asarray(image, dtype=np.uint8)
    arr = np.ascontiguousarray(arr)

    msg = RosImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()

    return msg


def clone_depth_snapshot_msg(
    msg: RosImage,
    stamp,
    frame_id: str = "zeri_vlm_input_depth",
) -> RosImage:
    cloned = copy.deepcopy(msg)
    cloned.header.stamp = stamp
    cloned.header.frame_id = frame_id
    return cloned


class QwenVLMRunner:
    def __init__(self, model_id: str, dtype_name: str, max_new_tokens: int):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if dtype_name == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self, image: PILImage.Image, stt_text: str) -> VLMDecision:
        prompt = USER_PROMPT_TEMPLATE.format(stt_text=stt_text)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = extract_json_object(output_text)
        return normalize_decision(parsed, output_text)


class ZeriVLMSTTBridgeNode(Node):
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

        self.declare_parameter("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("max_new_tokens", 128)
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("queue_size", 4)

        # STT gate parameters
        self.declare_parameter("stt_gate_mode", "wake")
        self.declare_parameter("stt_min_chars", 3)
        self.declare_parameter("wake_listen_window_sec", 8.0)
        self.declare_parameter("min_inference_interval_sec", 2.0)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("wake_words", ["제리", "제리야"])

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.stt_topic = str(self.get_parameter("stt_topic").value)

        self.decision_topic = str(self.get_parameter("decision_topic").value)
        self.robot_speech_topic = str(self.get_parameter("robot_speech_topic").value)
        self.vlm_input_rgb_topic = str(self.get_parameter("vlm_input_rgb_topic").value)
        self.vlm_input_depth_topic = str(self.get_parameter("vlm_input_depth_topic").value)
        self.inference_status_topic = str(self.get_parameter("inference_status_topic").value)

        model_id = str(self.get_parameter("model_id").value)
        dtype = str(self.get_parameter("dtype").value)
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        queue_size = int(self.get_parameter("queue_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)

        self.stt_gate_mode = str(self.get_parameter("stt_gate_mode").value)
        self.stt_min_chars = int(self.get_parameter("stt_min_chars").value)
        self.wake_listen_window_sec = float(self.get_parameter("wake_listen_window_sec").value)
        self.min_inference_interval_sec = float(
            self.get_parameter("min_inference_interval_sec").value
        )
        self.duplicate_window_sec = float(self.get_parameter("duplicate_window_sec").value)

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

        self.last_accepted_text = ""
        self.last_accepted_time = 0.0
        self.last_inference_request_time = 0.0
        self.wake_active_until = 0.0

        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
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

        self.get_logger().info("Zeri VLM-STT bridge node subscriptions:")
        self.get_logger().info(f"  RGB input:    {self.rgb_topic}")
        self.get_logger().info(f"  Depth input:  {self.depth_topic}")
        self.get_logger().info(f"  STT input:    {self.stt_topic}")

        self.get_logger().info("Zeri VLM-STT bridge node publishers:")
        self.get_logger().info(f"  Decision:     {self.decision_topic}")
        self.get_logger().info(f"  Robot speech: {self.robot_speech_topic}")
        self.get_logger().info(f"  VLM RGB snap: {self.vlm_input_rgb_topic}")
        self.get_logger().info(f"  VLM D snap:   {self.vlm_input_depth_topic}")
        self.get_logger().info(f"  Status:       {self.inference_status_topic}")

        self.get_logger().info("STT gate settings:")
        self.get_logger().info(f"  stt_gate_mode: {self.stt_gate_mode}")
        self.get_logger().info(f"  wake_words: {self.wake_words}")
        self.get_logger().info(f"  wake_listen_window_sec: {self.wake_listen_window_sec}")
        self.get_logger().info(f"  min_inference_interval_sec: {self.min_inference_interval_sec}")
        self.get_logger().info(f"  duplicate_window_sec: {self.duplicate_window_sec}")

        self.publish_inference_status("loading_vlm_model")
        self.get_logger().info(f"Loading VLM model: {model_id}")

        self.vlm = QwenVLMRunner(
            model_id=model_id,
            dtype_name=dtype,
            max_new_tokens=max_new_tokens,
        )

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

        self.publish_inference_status("waiting_for_camera_frame")
        self.get_logger().info("Zeri VLM-STT bridge node is ready.")

    def publish_inference_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.inference_status_publisher.publish(msg)
        self.get_logger().info(f"[VLM STATUS] {status}")

    def normalize_stt_text(self, text: str) -> str:
        text = text.strip()
        text = " ".join(text.split())
        return text

    def contains_any_keyword(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def remove_wake_words(self, text: str) -> str:
        cleaned = text

        for wake_word in sorted(self.wake_words, key=len, reverse=True):
            cleaned = cleaned.replace(wake_word, " ")

        for ch in [",", ".", "!", "?", "~", "…", ":", ";", "，", "。"]:
            cleaned = cleaned.replace(ch, " ")

        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    def filter_stt_text_for_vlm(self, text: str) -> Optional[str]:
        """
        STT 텍스트를 VLM에 넘길지 결정한다.

        동작:
        - "제리 숨쉬기가 힘들어" -> "숨쉬기가 힘들어"를 VLM에 전달
        - "제리" -> wake 상태만 켜고 VLM 추론은 하지 않음
        - wake 상태 8초 안에 들어온 문장 -> VLM에 전달
        - wake 없이 들어온 문장 -> 무시
        """
        now = time.time()
        text = self.normalize_stt_text(text)

        if not text:
            self.publish_inference_status("ignored_empty_stt")
            return None

        if text in self.ignore_phrases:
            self.get_logger().info(f"Ignored phrase STT text: {text}")
            self.publish_inference_status("ignored_phrase_stt")
            return None

        has_wake_word = self.contains_any_keyword(text, self.wake_words)

        if self.stt_gate_mode == "all":
            command_text = text

        elif self.stt_gate_mode == "wake":
            if has_wake_word:
                self.wake_active_until = now + self.wake_listen_window_sec
                command_text = self.remove_wake_words(text)

                if len(command_text) < self.stt_min_chars:
                    self.get_logger().info(
                        f"Wake word detected. Waiting for command for "
                        f"{self.wake_listen_window_sec:.1f}s."
                    )
                    self.publish_inference_status("wake_word_detected_waiting_command")
                    return None

            else:
                if now > self.wake_active_until:
                    self.get_logger().info(f"Ignored STT without wake word: {text}")
                    self.publish_inference_status("ignored_no_wake_word_stt")
                    return None

                command_text = text

        else:
            self.get_logger().warn(
                f"Unknown stt_gate_mode={self.stt_gate_mode}. Fallback to wake mode."
            )

            if not has_wake_word and now > self.wake_active_until:
                self.publish_inference_status("ignored_no_wake_word_stt")
                return None

            command_text = self.remove_wake_words(text) if has_wake_word else text

        command_text = self.normalize_stt_text(command_text)

        if len(command_text) < self.stt_min_chars:
            self.get_logger().info(f"Ignored short command after gate: {command_text}")
            self.publish_inference_status("ignored_short_stt")
            return None

        if command_text in self.ignore_phrases:
            self.get_logger().info(f"Ignored phrase command after gate: {command_text}")
            self.publish_inference_status("ignored_phrase_stt")
            return None

        is_duplicate = (
            self.last_accepted_text == command_text
            and now - self.last_accepted_time < self.duplicate_window_sec
        )

        if is_duplicate:
            self.get_logger().info(f"Ignored duplicate STT command: {command_text}")
            self.publish_inference_status("ignored_duplicate_stt")
            return None

        cooldown_active = (
            now - self.last_inference_request_time < self.min_inference_interval_sec
        )

        if cooldown_active:
            self.get_logger().info(f"Ignored STT due to cooldown: {command_text}")
            self.publish_inference_status("ignored_stt_cooldown")
            return None

        self.last_accepted_text = command_text
        self.last_accepted_time = now
        self.last_inference_request_time = now

        return command_text

    def rgb_callback(self, msg: RosImage) -> None:
        with self.frame_lock:
            self.latest_rgb_msg = msg
            self.latest_rgb_time = time.time()

    def depth_callback(self, msg: RosImage) -> None:
        with self.frame_lock:
            self.latest_depth_msg = msg
            self.latest_depth_time = time.time()

    def stt_callback(self, msg: String) -> None:
        raw_stt_text = msg.data.strip()

        if not raw_stt_text:
            return

        self.get_logger().info(f"Received raw STT text: {raw_stt_text}")

        accepted_text = self.filter_stt_text_for_vlm(raw_stt_text)

        if accepted_text is None:
            return

        self.get_logger().info(
            f"Accepted STT text for VLM: raw='{raw_stt_text}', command='{accepted_text}'"
        )
        self.publish_inference_status("accepted_stt_text")

        try:
            self.text_queue.put_nowait(accepted_text)
        except queue.Full:
            try:
                dropped = self.text_queue.get_nowait()
                self.get_logger().warn(f"Dropped old STT text due to full queue: {dropped}")
            except queue.Empty:
                pass

            try:
                self.text_queue.put_nowait(accepted_text)
            except queue.Full:
                self.get_logger().error("Failed to enqueue STT text.")
                self.publish_inference_status("queue_full_error")

    def get_latest_frames(
        self,
    ) -> Tuple[Optional[RosImage], Optional[RosImage], Optional[float], Optional[float]]:
        with self.frame_lock:
            return (
                self.latest_rgb_msg,
                self.latest_depth_msg,
                self.latest_rgb_time,
                self.latest_depth_time,
            )

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                stt_text = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            start_time = time.time()

            try:
                self.publish_inference_status("getting_latest_camera_frame")

                rgb_msg, depth_msg, rgb_time, depth_time = self.get_latest_frames()

                if rgb_msg is None:
                    self.get_logger().warn("No RGB frame received yet.")
                    self.publish_inference_status("waiting_for_rgb_frame")
                    continue

                image = ros_image_to_pil_rgb(rgb_msg)

                if image is None:
                    self.get_logger().error("Failed to convert RGB ROS image to PIL.")
                    self.publish_inference_status("rgb_conversion_error")
                    continue

                stamp = self.get_clock().now().to_msg()

                vlm_rgb_msg = pil_rgb_to_ros_image(
                    image=image,
                    stamp=stamp,
                    frame_id="zeri_vlm_input_rgb",
                )

                self.vlm_input_rgb_publisher.publish(vlm_rgb_msg)
                self.get_logger().info("Published VLM input RGB snapshot.")

                if depth_msg is not None:
                    vlm_depth_msg = clone_depth_snapshot_msg(
                        msg=depth_msg,
                        stamp=stamp,
                        frame_id="zeri_vlm_input_depth",
                    )
                    self.vlm_input_depth_publisher.publish(vlm_depth_msg)
                    self.get_logger().info("Published VLM input Depth snapshot.")
                else:
                    self.get_logger().warn("No depth frame received yet. Continuing with RGB only.")

                self.publish_inference_status("running_vlm_inference")
                self.get_logger().info("Running VLM inference...")

                decision = self.vlm.infer(
                    image=image,
                    stt_text=stt_text,
                )

                elapsed = time.time() - start_time

                self.publish_inference_status("vlm_inference_done")

                mock_action = "idle"

                if (
                    decision.need_oxygen_mask
                    and decision.adapter_id == "oxygen_mask_delivery_lora"
                    and decision.confidence >= self.confidence_threshold
                ):
                    mock_action = "execute_oxygen_mask_delivery"

                camera_age_sec = None
                if rgb_time is not None:
                    camera_age_sec = round(time.time() - rgb_time, 3)

                result = {
                    "stt_text": stt_text,
                    "need_oxygen_mask": decision.need_oxygen_mask,
                    "selected_task": decision.selected_task,
                    "adapter_id": decision.adapter_id,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "robot_speech": decision.robot_speech,
                    "mock_action": mock_action,
                    "latency_sec": round(elapsed, 3),
                    "camera_age_sec": camera_age_sec,
                    "raw_vlm_output": decision.raw_text,
                    "live_rgb_topic": self.rgb_topic,
                    "live_depth_topic": self.depth_topic,
                    "vlm_input_rgb_topic": self.vlm_input_rgb_topic,
                    "vlm_input_depth_topic": self.vlm_input_depth_topic,
                }

                decision_msg = String()
                decision_msg.data = json.dumps(result, ensure_ascii=False)
                self.decision_publisher.publish(decision_msg)

                speech_msg = String()
                speech_msg.data = decision.robot_speech
                self.robot_speech_publisher.publish(speech_msg)

                self.get_logger().info(f"Published VLM decision: {decision_msg.data}")
                self.get_logger().info(f"Published robot speech: {decision.robot_speech}")
                self.get_logger().info(f"[ROBOT SPEECH] {decision.robot_speech}")

                if mock_action == "execute_oxygen_mask_delivery":
                    self.log_mock_action(decision)
                else:
                    self.log_idle_action(decision)

                self.publish_inference_status("idle")

            except Exception as exc:
                err = f"error: {exc}"
                self.get_logger().error(f"VLM worker error: {exc}")
                self.publish_inference_status(err)

    def log_mock_action(self, decision: VLMDecision) -> None:
        self.get_logger().info("[MOCK ACTION] Selected adapter: oxygen_mask_delivery_lora")
        self.get_logger().info("[MOCK ACTION] Loading oxygen_mask_delivery_policy ... MOCK")
        self.get_logger().info("[MOCK ACTION] Moving mobile base toward victim ... MOCK")
        self.get_logger().info("[MOCK ACTION] Executing SO-101 arm trajectory ... MOCK")
        self.get_logger().info(f"[MOCK ACTION] Reason: {decision.reason}")
        self.get_logger().info(f"[ROBOT SPEECH] {decision.robot_speech}")
        self.get_logger().info("[MOCK ACTION] DONE")

    def log_idle_action(self, decision: VLMDecision) -> None:
        self.get_logger().info("[MOCK ACTION] No oxygen-mask delivery required.")
        self.get_logger().info(f"[MOCK ACTION] Reason: {decision.reason}")
        self.get_logger().info(f"[ROBOT SPEECH] {decision.robot_speech}")
        self.get_logger().info("[MOCK ACTION] IDLE")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping VLM-STT bridge node.")

        try:
            self.publish_inference_status("shutting_down")
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