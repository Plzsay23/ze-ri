#!/usr/bin/env python3
# src/lerobot/vlm_agent/tts_piper_node.py

import os
import queue
import shlex
import subprocess
import tempfile
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


class PiperTTSNode(Node):
    def __init__(self) -> None:
        super().__init__("zeri_tts_piper_node")

        self.declare_parameter("input_topic", "/zeri/vlm/robot_speech")
        self.declare_parameter("status_topic", "/zeri/tts/status")

        self.declare_parameter("piper_bin", "piper")
        self.declare_parameter("model_path", "")
        self.declare_parameter("config_path", "")

        self.declare_parameter("player", "aplay")
        self.declare_parameter("output_wav", "/tmp/zeri_tts_output.wav")

        self.declare_parameter("min_chars", 2)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("cooldown_sec", 0.5)
        self.declare_parameter("queue_size", 4)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.piper_bin = str(self.get_parameter("piper_bin").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.config_path = str(self.get_parameter("config_path").value)

        self.player = str(self.get_parameter("player").value)
        self.output_wav = str(self.get_parameter("output_wav").value)

        self.min_chars = int(self.get_parameter("min_chars").value)
        self.duplicate_window_sec = float(self.get_parameter("duplicate_window_sec").value)
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

        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("Zeri Piper TTS node ready.")
        self.get_logger().info(f"  input_topic: {self.input_topic}")
        self.get_logger().info(f"  status_topic: {self.status_topic}")
        self.get_logger().info(f"  piper_bin: {self.piper_bin}")
        self.get_logger().info(f"  model_path: {self.model_path}")
        self.get_logger().info(f"  config_path: {self.config_path}")
        self.get_logger().info(f"  player: {self.player}")
        self.get_logger().info(f"  output_wav: {self.output_wav}")

        if not self.model_path:
            self.get_logger().warn(
                "model_path is empty. Run with -p model_path:=/path/to/model.onnx"
            )

        self.publish_status("idle")

    def publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[TTS STATUS] {status}")

    def normalize_text(self, text: str) -> str:
        text = text.strip()
        text = " ".join(text.split())
        return text

    def text_callback(self, msg: String) -> None:
        text = self.normalize_text(msg.data)

        if len(text) < self.min_chars:
            self.publish_status("ignored_short_text")
            return

        now = time.time()

        if self.last_text == text and now - self.last_text_time < self.duplicate_window_sec:
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

    def build_piper_command(self, output_wav: str) -> list[str]:
        cmd = [
            self.piper_bin,
            "--model",
            self.model_path,
            "--output_file",
            output_wav,
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        return cmd

    def run_piper(self, text: str, output_wav: str) -> None:
        if not self.model_path:
            raise RuntimeError("Piper model_path is empty.")

        cmd = self.build_piper_command(output_wav)

        self.get_logger().info(f"Running Piper: {' '.join(shlex.quote(x) for x in cmd)}")

        proc = subprocess.run(
            cmd,
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Piper failed with code {proc.returncode}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )

        if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
            raise RuntimeError(f"Piper output wav was not created: {output_wav}")

    def play_wav(self, wav_path: str) -> None:
        if self.player == "none":
            self.get_logger().info(f"Playback disabled. wav_path={wav_path}")
            return

        cmd = [self.player, wav_path]

        self.get_logger().info(f"Playing wav: {' '.join(shlex.quote(x) for x in cmd)}")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Audio player failed with code {proc.returncode}\n"
                f"stdout={proc.stdout.decode(errors='ignore')}\n"
                f"stderr={proc.stderr.decode(errors='ignore')}"
            )

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.publish_status("synthesizing")
                self.get_logger().info(f"[TTS TEXT] {text}")

                output_wav = self.output_wav

                self.run_piper(text=text, output_wav=output_wav)

                self.publish_status("playing")
                self.play_wav(output_wav)

                self.publish_status("idle")

                if self.cooldown_sec > 0:
                    time.sleep(self.cooldown_sec)

            except Exception as exc:
                self.get_logger().error(f"TTS worker error: {exc}")
                self.publish_status(f"error: {exc}")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping Piper TTS node.")

        self.stop_event.set()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)

        super().destroy_node()


def main() -> None:
    rclpy.init()

    node: Optional[PiperTTSNode] = None

    try:
        node = PiperTTSNode()
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