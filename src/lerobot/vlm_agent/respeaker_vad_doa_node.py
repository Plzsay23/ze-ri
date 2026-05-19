#!/usr/bin/env python3
# src/lerobot/vlm_agent/respeaker_vad_doa_node.py

import argparse
import importlib.util
import json
import os
import threading
import time
from typing import Any, Optional

import rclpy
import usb.core
import usb.util
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def parse_int_auto(value: str | int) -> int:
    if isinstance(value, int):
        return value

    value = str(value).strip()

    if value.lower().startswith("0x"):
        return int(value, 16)

    return int(value)


def load_tuning_class(tuning_py_path: str):
    """
    Seeed usb_4_mic_array/tuning.py 안의 Tuning 클래스를 동적으로 로드한다.
    """
    if tuning_py_path and os.path.exists(tuning_py_path):
        spec = importlib.util.spec_from_file_location(
            "respeaker_tuning",
            tuning_py_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load tuning.py spec: {tuning_py_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Tuning"):
            raise RuntimeError(f"Tuning class not found in: {tuning_py_path}")

        return module.Tuning

    try:
        from tuning import Tuning  # type: ignore

        return Tuning
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Tuning. "
            "Set -p tuning_py_path:=/path/to/usb_4_mic_array/tuning.py"
        ) from exc


class ReSpeakerVadDoaNode(Node):
    """
    ReSpeaker Mic Array v3.0 VAD/DOA publisher.

    Publish:
      /zeri/audio/vad         std_msgs/Bool
      /zeri/audio/doa_deg     std_msgs/Float32
      /zeri/audio/event       std_msgs/String(JSON)
      /zeri/audio/status      std_msgs/String

    Subscribe:
      /zeri/stt/mute          std_msgs/Bool

    목적:
      - VAD true일 때만 STT/VLM 게이트를 열 수 있게 함
      - DOA 값을 로봇 회전/탐색 방향으로 사용할 수 있게 토픽화
      - TTS 출력 중에는 ReSpeaker가 로봇 자기 목소리를 VAD로 잡지 않도록 muted 처리
    """

    def __init__(self) -> None:
        super().__init__("zeri_respeaker_vad_doa_node")

        self.declare_parameter(
            "tuning_py_path",
            "/home/hansungai/tools/usb_4_mic_array/tuning.py",
        )

        self.declare_parameter("vendor_id", "0x2886")
        self.declare_parameter("product_id", "0x0018")

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("event_topic", "/zeri/audio/event")
        self.declare_parameter("status_topic", "/zeri/audio/status")
        self.declare_parameter("stt_mute_topic", "/zeri/stt/mute")

        self.declare_parameter("poll_hz", 10.0)
        self.declare_parameter("event_period_sec", 0.5)
        self.declare_parameter("reconnect_period_sec", 2.0)

        self.declare_parameter("mute_blocks_vad", True)
        self.declare_parameter("publish_doa_when_muted", False)

        self.tuning_py_path = str(self.get_parameter("tuning_py_path").value)

        self.vendor_id = parse_int_auto(self.get_parameter("vendor_id").value)
        self.product_id = parse_int_auto(self.get_parameter("product_id").value)

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.event_topic = str(self.get_parameter("event_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.stt_mute_topic = str(self.get_parameter("stt_mute_topic").value)

        self.poll_hz = float(self.get_parameter("poll_hz").value)
        self.event_period_sec = float(self.get_parameter("event_period_sec").value)
        self.reconnect_period_sec = float(self.get_parameter("reconnect_period_sec").value)

        self.mute_blocks_vad = bool(self.get_parameter("mute_blocks_vad").value)
        self.publish_doa_when_muted = bool(self.get_parameter("publish_doa_when_muted").value)

        qos = make_reliable_qos(depth=10)

        self.vad_pub = self.create_publisher(Bool, self.vad_topic, qos)
        self.doa_pub = self.create_publisher(Float32, self.doa_topic, qos)
        self.event_pub = self.create_publisher(String, self.event_topic, qos)
        self.status_pub = self.create_publisher(String, self.status_topic, qos)

        self.stt_mute_sub = self.create_subscription(
            Bool,
            self.stt_mute_topic,
            self.stt_mute_callback,
            qos,
        )

        self.lock = threading.Lock()

        self.TuningClass = None
        self.dev = None
        self.mic_tuning = None

        self.stt_muted = False
        self.last_vad = False
        self.last_doa_deg: Optional[float] = None

        self.last_event_time = 0.0
        self.last_reconnect_time = 0.0
        self.last_status = ""

        self.get_logger().info("ReSpeaker VAD/DOA node settings:")
        self.get_logger().info(f"  tuning_py_path: {self.tuning_py_path}")
        self.get_logger().info(f"  vendor_id: 0x{self.vendor_id:04x}")
        self.get_logger().info(f"  product_id: 0x{self.product_id:04x}")
        self.get_logger().info(f"  vad_topic: {self.vad_topic}")
        self.get_logger().info(f"  doa_topic: {self.doa_topic}")
        self.get_logger().info(f"  event_topic: {self.event_topic}")
        self.get_logger().info(f"  status_topic: {self.status_topic}")
        self.get_logger().info(f"  stt_mute_topic: {self.stt_mute_topic}")
        self.get_logger().info(f"  poll_hz: {self.poll_hz}")
        self.get_logger().info(f"  mute_blocks_vad: {self.mute_blocks_vad}")

        self.connect_device()

        timer_period = 1.0 / max(1.0, self.poll_hz)
        self.timer = self.create_timer(timer_period, self.poll_once)

    def publish_status(self, status: str, force: bool = False) -> None:
        if not force and status == self.last_status:
            return

        self.last_status = status

        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[AUDIO STATUS] {status}")

    def publish_event(
        self,
        event: str,
        vad: bool,
        doa_deg: Optional[float],
        force: bool = False,
    ) -> None:
        now = time.time()

        if not force and now - self.last_event_time < self.event_period_sec:
            return

        self.last_event_time = now

        payload = {
            "event": event,
            "vad": vad,
            "doa_deg": doa_deg,
            "stt_muted": self.stt_muted,
            "stamp_sec": round(now, 3),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.event_pub.publish(msg)

    def stt_mute_callback(self, msg: Bool) -> None:
        with self.lock:
            self.stt_muted = bool(msg.data)

        if self.stt_muted:
            self.publish_status("stt_muted")
            self.publish_vad(False)
            self.publish_event(
                event="stt_muted",
                vad=False,
                doa_deg=self.last_doa_deg,
                force=True,
            )
        else:
            self.publish_status("stt_unmuted")
            self.publish_event(
                event="stt_unmuted",
                vad=self.last_vad,
                doa_deg=self.last_doa_deg,
                force=True,
            )

    def connect_device(self) -> None:
        now = time.time()

        if now - self.last_reconnect_time < self.reconnect_period_sec:
            return

        self.last_reconnect_time = now

        try:
            if self.TuningClass is None:
                self.TuningClass = load_tuning_class(self.tuning_py_path)

            dev = usb.core.find(
                idVendor=self.vendor_id,
                idProduct=self.product_id,
            )

            if dev is None:
                self.dev = None
                self.mic_tuning = None
                self.publish_status("device_not_found")
                return

            self.dev = dev
            self.mic_tuning = self.TuningClass(dev)

            self.publish_status("connected", force=True)
            self.get_logger().info("ReSpeaker control device connected.")

        except Exception as exc:
            self.dev = None
            self.mic_tuning = None
            self.publish_status(f"connect_error: {exc}", force=True)
            self.get_logger().error(f"Failed to connect ReSpeaker control device: {exc}")

    def read_vad_raw(self) -> Optional[bool]:
        if self.mic_tuning is None:
            return None

        try:
            value = self.mic_tuning.is_voice()
            return bool(int(value))
        except Exception as exc:
            self.get_logger().warn(f"Failed to read VAD: {exc}")
            return None

    def read_doa_raw(self) -> Optional[float]:
        if self.mic_tuning is None:
            return None

        try:
            value = self.mic_tuning.direction
            doa = float(value)

            # normalise to [0, 360)
            doa = doa % 360.0

            return doa
        except Exception as exc:
            self.get_logger().warn(f"Failed to read DOA: {exc}")
            return None

    def publish_vad(self, vad: bool) -> None:
        msg = Bool()
        msg.data = bool(vad)
        self.vad_pub.publish(msg)

    def publish_doa(self, doa_deg: float) -> None:
        msg = Float32()
        msg.data = float(doa_deg)
        self.doa_pub.publish(msg)

    def poll_once(self) -> None:
        if self.mic_tuning is None:
            self.connect_device()
            return

        with self.lock:
            stt_muted = self.stt_muted

        if stt_muted and self.mute_blocks_vad:
            self.publish_vad(False)

            if self.publish_doa_when_muted:
                doa_deg = self.read_doa_raw()
                if doa_deg is not None:
                    self.last_doa_deg = doa_deg
                    self.publish_doa(doa_deg)

            if self.last_vad:
                self.last_vad = False
                self.publish_event(
                    event="speech_end_muted",
                    vad=False,
                    doa_deg=self.last_doa_deg,
                    force=True,
                )

            return

        vad_value = self.read_vad_raw()
        doa_deg = self.read_doa_raw()

        if vad_value is None and doa_deg is None:
            self.publish_status("read_error")
            self.mic_tuning = None
            self.dev = None
            return

        vad = bool(vad_value) if vad_value is not None else False

        self.publish_vad(vad)

        if doa_deg is not None:
            self.last_doa_deg = doa_deg
            self.publish_doa(doa_deg)

        if vad and not self.last_vad:
            self.publish_event(
                event="speech_start",
                vad=True,
                doa_deg=self.last_doa_deg,
                force=True,
            )

        elif not vad and self.last_vad:
            self.publish_event(
                event="speech_end",
                vad=False,
                doa_deg=self.last_doa_deg,
                force=True,
            )

        elif vad:
            self.publish_event(
                event="speech_active",
                vad=True,
                doa_deg=self.last_doa_deg,
                force=False,
            )

        self.last_vad = vad
        self.publish_status("running")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping ReSpeaker VAD/DOA node.")

        try:
            self.publish_vad(False)
            self.publish_status("shutting_down", force=True)
        except Exception:
            pass

        super().destroy_node()


def main() -> None:
    rclpy.init()

    node: Optional[ReSpeakerVadDoaNode] = None

    try:
        node = ReSpeakerVadDoaNode()
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