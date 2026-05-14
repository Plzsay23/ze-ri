#!/usr/bin/env python3
# src/lerobot/vlm_agent/tts_edge_node.py

import os
import queue
import shlex
import subprocess
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


class EdgeTTSNode(Node):
    """
    Ze-Ri Edge TTS node.

    입력:
      /zeri/vlm/robot_speech  std_msgs/String

    출력:
      실제 스피커 출력
      /zeri/tts/status         std_msgs/String

    기본 음성:
      ko-KR-SunHiNeural
    """

    def __init__(self) -> None:
        super().__init__("zeri_tts_edge_node")

        self.declare_parameter("input_topic", "/zeri/vlm/robot_speech")
        self.declare_parameter("status_topic", "/zeri/tts/status")

        self.declare_parameter("edge_tts_bin", "edge-tts")
        self.declare_parameter("voice", "ko-KR-SunHiNeural")
        self.declare_parameter("rate", "+0%")
        self.declare_parameter("volume", "+0%")
        self.declare_parameter("pitch", "+0Hz")

        self.declare_parameter("output_mp3", "/tmp/zeri_tts_output.mp3")
        self.declare_parameter("player", "ffplay")
        self.declare_parameter("player_mode", "ffplay")

        self.declare_parameter("min_chars", 2)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("cooldown_sec", 0.3)
        self.declare_parameter("queue_size", 4)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.edge_tts_bin = str(self.get_parameter("edge_tts_bin").value)
        self.voice = str(self.get_parameter("voice").value)
        self.rate = str(self.get_parameter("rate").value)
        self.volume = str(self.get_parameter("volume").value)
        self.pitch = str(self.get_parameter("pitch").value)

        self.output_mp3 = str(self.get_parameter("output_mp3").value)
        self.player = str(self.get_parameter("player").value)
        self.player_mode = str(self.get_parameter("player_mode").value)

        self.min_chars = int(self.get_parameter("min_chars").value)
        self.duplicate_window_sec = float(
            self.get_parameter("duplicate_window_sec").value
        )
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        self.last_text = ""
        self.last_text_time = 0.0

        qos = make_reliable_qos(depth=10)

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.text_callback,
            qos,
        )

        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            qos,
        )

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

        self.get_logger().info("Zeri Edge TTS node ready.")
        self.get_logger().info(f"  input_topic: {self.input_topic}")
        self.get_logger().info(f"  status_topic: {self.status_topic}")
        self.get_logger().info(f"  edge_tts_bin: {self.edge_tts_bin}")
        self.get_logger().info(f"  voice: {self.voice}")
        self.get_logger().info(f"  rate: {self.rate}")
        self.get_logger().info(f"  volume: {self.volume}")
        self.get_logger().info(f"  pitch: {self.pitch}")
        self.get_logger().info(f"  output_mp3: {self.output_mp3}")
        self.get_logger().info(f"  player: {self.player}")
        self.get_logger().info(f"  player_mode: {self.player_mode}")

        self.publish_status("idle")

    def publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[TTS STATUS] {status}")

    def normalize_text(self, text: str) -> str:
        text = text.strip()
        text = " ".join(text.split())

        replacements = {
            "산소마스크": "산소 마스크",
            "VLM": "브이 엘 엠",
            "STT": "에스 티 티",
            "TTS": "티 티 에스",
            "LoRA": "로라",
            "adapter": "어댑터",
            "Adapter": "어댑터",
            "idle": "대기",
            "oxygen": "산소",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = text.replace(
            "현재 산소 마스크 전달은 필요하지 않습니다.",
            "산소 마스크는 아직 필요하지 않습니다.",
        )
        text = text.replace(
            "계속 상황을 확인하겠습니다.",
            "계속 확인하겠습니다.",
        )

        return text

    def text_callback(self, msg: String) -> None:
        text = self.normalize_text(msg.data)

        if len(text) < self.min_chars:
            self.publish_status("ignored_short_text")
            return

        now = time.time()

        is_duplicate = (
            self.last_text == text
            and now - self.last_text_time < self.duplicate_window_sec
        )

        if is_duplicate:
            self.get_logger().info(f"Ignored duplicate TTS text: {text}")
            self.publish_status("ignored_duplicate_text")
            return

        self.last_text = text
        self.last_text_time = now

        try:
            self.text_queue.put_nowait(text)
            self.publish_status("queued")
        except queue.Full:
            try:
                dropped = self.text_queue.get_nowait()
                self.get_logger().warn(f"Dropped old TTS text: {dropped}")
            except queue.Empty:
                pass

            try:
                self.text_queue.put_nowait(text)
                self.publish_status("queued_after_drop")
            except queue.Full:
                self.publish_status("queue_full_error")

    def build_edge_tts_command(self, text: str) -> list[str]:
        cmd = [
            self.edge_tts_bin,
            "--voice",
            self.voice,
            "--text",
            text,
            "--write-media",
            self.output_mp3,
        ]

        if self.rate:
            cmd.extend(["--rate", self.rate])

        if self.volume:
            cmd.extend(["--volume", self.volume])

        if self.pitch:
            cmd.extend(["--pitch", self.pitch])

        return cmd

    def run_edge_tts(self, text: str) -> None:
        if os.path.exists(self.output_mp3):
            try:
                os.remove(self.output_mp3)
            except OSError:
                pass

        cmd = self.build_edge_tts_command(text)

        self.get_logger().info(
            f"Running Edge TTS: {' '.join(shlex.quote(x) for x in cmd)}"
        )

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"edge-tts failed with code {proc.returncode}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )

        if not os.path.exists(self.output_mp3):
            raise RuntimeError(f"Edge TTS output file was not created: {self.output_mp3}")

        if os.path.getsize(self.output_mp3) == 0:
            raise RuntimeError(f"Edge TTS output file is empty: {self.output_mp3}")

    def build_player_command(self) -> list[str]:
        if self.player_mode == "ffplay":
            return [
                self.player,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                self.output_mp3,
            ]

        if self.player_mode == "mpg123":
            return [
                self.player,
                self.output_mp3,
            ]

        if self.player_mode == "paplay":
            return [
                self.player,
                self.output_mp3,
            ]

        if self.player_mode == "custom":
            return [
                self.player,
                self.output_mp3,
            ]

        raise RuntimeError(
            f"Unknown player_mode={self.player_mode}. "
            "Valid values: ffplay, mpg123, paplay, custom"
        )

    def play_audio(self) -> None:
        cmd = self.build_player_command()

        self.get_logger().info(
            f"Playing audio: {' '.join(shlex.quote(x) for x in cmd)}"
        )

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )

        if proc.returncode != 0:
            stdout = proc.stdout.decode(errors="ignore") if proc.stdout else ""
            stderr = proc.stderr.decode(errors="ignore") if proc.stderr else ""
            raise RuntimeError(
                f"audio player failed with code {proc.returncode}\n"
                f"stdout={stdout}\n"
                f"stderr={stderr}"
            )

    def synthesize_and_play(self, text: str) -> None:
        self.publish_status("synthesizing_edge_tts")
        self.get_logger().info(f"[TTS TEXT] {text}")

        self.run_edge_tts(text)

        self.publish_status("playing")
        self.play_audio()

        self.publish_status("idle")

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.synthesize_and_play(text)

                if self.cooldown_sec > 0:
                    time.sleep(self.cooldown_sec)

            except Exception as exc:
                self.get_logger().error(f"TTS worker error: {exc}")
                self.publish_status(f"error: {exc}")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping Edge TTS node.")

        try:
            self.publish_status("shutting_down")
        except Exception:
            pass

        self.stop_event.set()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)

        super().destroy_node()


def main() -> None:
    rclpy.init()

    node: Optional[EdgeTTSNode] = None

    try:
        node = EdgeTTSNode()
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