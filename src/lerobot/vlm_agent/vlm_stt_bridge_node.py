# vlm_stt_bridge_node.py

import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.

현재 시스템에는 실제 로봇 정책 모델이 아직 없다.
그러나 산소마스크 전달 정책 모델이 존재한다고 가정한다.

입력:
- RealSense RGB 카메라 이미지
- STT에서 들어온 사용자 텍스트

목표:
- 산소마스크 전달이 필요한 상황인지 판단한다.
- 필요하면 adapter_id를 "oxygen_mask_delivery_lora"로 선택한다.
- 필요하지 않으면 adapter_id를 "idle_lora"로 선택한다.
- 반드시 JSON으로만 답한다.
- JSON 바깥 문장은 쓰지 않는다.

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
  "reason": "짧은 한국어 이유"
}
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
    raw_text: str


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

    if need:
        selected_task = "oxygen_mask_delivery"
        adapter_id = "oxygen_mask_delivery_lora"
    else:
        selected_task = "idle"
        adapter_id = "idle_lora"

    return VLMDecision(
        need_oxygen_mask=need,
        selected_task=selected_task,
        adapter_id=adapter_id,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
    )


class RealSenseColorCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None

    def start(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        pipeline.start(config)

        # Auto exposure warmup
        for _ in range(30):
            pipeline.wait_for_frames()

        self.pipeline = pipeline

    def capture_rgb(self) -> Image.Image:
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline is not started.")

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to capture color frame.")

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


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

    def infer(self, image: Image.Image, stt_text: str) -> VLMDecision:
        prompt = USER_PROMPT_TEMPLATE.format(stt_text=stt_text)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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

        self.declare_parameter("stt_topic", "/zeri/stt/text")
        self.declare_parameter("decision_topic", "/zeri/vlm/decision")
        self.declare_parameter("serial", "332322071907")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("max_new_tokens", 128)
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("queue_size", 4)

        self.stt_topic = self.get_parameter("stt_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)

        serial = self.get_parameter("serial").value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = int(self.get_parameter("fps").value)

        model_id = self.get_parameter("model_id").value
        dtype = self.get_parameter("dtype").value
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        self.publisher = self.create_publisher(String, self.decision_topic, 10)

        self.subscription = self.create_subscription(
            String,
            self.stt_topic,
            self.stt_callback,
            10,
        )

        self.get_logger().info(f"Subscribing STT topic: {self.stt_topic}")
        self.get_logger().info(f"Publishing decision topic: {self.decision_topic}")

        self.get_logger().info(f"Starting RealSense RGB camera serial={serial}")
        self.camera = RealSenseColorCamera(
            serial=serial,
            width=width,
            height=height,
            fps=fps,
        )
        self.camera.start()

        self.get_logger().info(f"Loading VLM model: {model_id}")
        self.vlm = QwenVLMRunner(
            model_id=model_id,
            dtype_name=dtype,
            max_new_tokens=max_new_tokens,
        )

        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("Zeri VLM-STT bridge node is ready.")

    def stt_callback(self, msg: String):
        stt_text = msg.data.strip()

        if not stt_text:
            return

        self.get_logger().info(f"Received STT text: {stt_text}")

        try:
            self.text_queue.put_nowait(stt_text)
        except queue.Full:
            # 최신 명령을 우선하기 위해 오래된 것 하나 버리고 다시 넣음
            try:
                dropped = self.text_queue.get_nowait()
                self.get_logger().warn(f"Dropped old STT text due to full queue: {dropped}")
            except queue.Empty:
                pass

            try:
                self.text_queue.put_nowait(stt_text)
            except queue.Full:
                self.get_logger().error("Failed to enqueue STT text.")

    def worker_loop(self):
        while not self.stop_event.is_set():
            try:
                stt_text = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            start_time = time.time()

            try:
                self.get_logger().info("Capturing RealSense RGB frame...")
                image = self.camera.capture_rgb()

                self.get_logger().info("Running VLM inference...")
                decision = self.vlm.infer(image=image, stt_text=stt_text)

                elapsed = time.time() - start_time

                mock_action = "idle"
                if (
                    decision.need_oxygen_mask
                    and decision.adapter_id == "oxygen_mask_delivery_lora"
                    and decision.confidence >= self.confidence_threshold
                ):
                    mock_action = "execute_oxygen_mask_delivery"

                result = {
                    "stt_text": stt_text,
                    "need_oxygen_mask": decision.need_oxygen_mask,
                    "selected_task": decision.selected_task,
                    "adapter_id": decision.adapter_id,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "mock_action": mock_action,
                    "latency_sec": round(elapsed, 3),
                    "raw_vlm_output": decision.raw_text,
                }

                out_msg = String()
                out_msg.data = json.dumps(result, ensure_ascii=False)

                self.publisher.publish(out_msg)

                self.get_logger().info(f"Published VLM decision: {out_msg.data}")

                if mock_action == "execute_oxygen_mask_delivery":
                    self.log_mock_action(decision)

            except Exception as exc:
                self.get_logger().error(f"VLM worker error: {exc}")

    def log_mock_action(self, decision: VLMDecision):
        self.get_logger().info("[MOCK ACTION] Selected adapter: oxygen_mask_delivery_lora")
        self.get_logger().info("[MOCK ACTION] Loading oxygen_mask_delivery_policy ... MOCK")
        self.get_logger().info("[MOCK ACTION] Moving mobile base toward victim ... MOCK")
        self.get_logger().info("[MOCK ACTION] Executing SO-101 arm trajectory ... MOCK")
        self.get_logger().info(f"[MOCK ACTION] Reason: {decision.reason}")
        self.get_logger().info("[MOCK ACTION] DONE")

    def destroy_node(self):
        self.get_logger().info("Stopping worker and RealSense camera...")
        self.stop_event.set()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)

        if hasattr(self, "camera"):
            self.camera.stop()

        super().destroy_node()


def main():
    rclpy.init()

    node = None

    try:
        node = ZeriVLMSTTBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()