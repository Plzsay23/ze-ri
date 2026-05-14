#!/usr/bin/env python3
# src/lerobot/vlm_agent/tts_piper_node.py

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


class PiperTTSNode(Node):
    """
    Ze-Ri TTS node.

    입력:
      /zeri/vlm/robot_speech  std_msgs/String

    출력:
      실제 스피커 출력
      /zeri/tts/status         std_msgs/String

    backend:
      - piper_rs:
          neurlang/piper-rs interactive 예제 사용.
          piper-kss-korean.onnx.json처럼 phoneme_type=pygoruut 모델용.

      - piper_cli:
          Python piper-tts CLI 사용.
          일반 Piper 모델용.
    """

    def __init__(self) -> None:
        super().__init__("zeri_tts_piper_node")

        # ROS topics
        self.declare_parameter("input_topic", "/zeri/vlm/robot_speech")
        self.declare_parameter("status_topic", "/zeri/tts/status")

        # backend selection
        self.declare_parameter("backend", "piper_rs")

        # piper-rs backend
        self.declare_parameter(
            "piper_rs_bin",
            "/home/hansungai/tools/piper-rs/target/release/examples/interactive",
        )
        self.declare_parameter(
            "config_path",
            "/home/hansungai/voicebot/piper_models/piper-kss-korean.onnx.json",
        )
        self.declare_parameter("piper_rs_volume", 80)

        # python piper CLI backend
        self.declare_parameter("piper_bin", "piper")
        self.declare_parameter(
            "model_path",
            "/home/hansungai/voicebot/piper_models/piper-kss-korean.onnx",
        )

        # playback for piper_cli backend only
        self.declare_parameter("player", "/home/hansungai/usb_speaker_play.sh")
        self.declare_parameter("output_wav", "/tmp/zeri_tts_output.wav")

        # filtering
        self.declare_parameter("min_chars", 2)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("cooldown_sec", 0.5)
        self.declare_parameter("queue_size", 4)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.backend = str(self.get_parameter("backend").value)

        self.piper_rs_bin = str(self.get_parameter("piper_rs_bin").value)
        self.config_path = str(self.get_parameter("config_path").value)
        self.piper_rs_volume = int(self.get_parameter("piper_rs_volume").value)

        self.piper_bin = str(self.get_parameter("piper_bin").value)
        self.model_path = str(self.get_parameter("model_path").value)

        self.player = str(self.get_parameter("player").value)
        self.output_wav = str(self.get_parameter("output_wav").value)

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

        self.get_logger().info("Zeri TTS node ready.")
        self.get_logger().info(f"  input_topic: {self.input_topic}")
        self.get_logger().info(f"  status_topic: {self.status_topic}")
        self.get_logger().info(f"  backend: {self.backend}")

        self.get_logger().info("piper-rs settings:")
        self.get_logger().info(f"  piper_rs_bin: {self.piper_rs_bin}")
        self.get_logger().info(f"  config_path: {self.config_path}")
        self.get_logger().info(f"  piper_rs_volume: {self.piper_rs_volume}")

        self.get_logger().info("piper CLI settings:")
        self.get_logger().info(f"  piper_bin: {self.piper_bin}")
        self.get_logger().info(f"  model_path: {self.model_path}")
        self.get_logger().info(f"  player: {self.player}")
        self.get_logger().info(f"  output_wav: {self.output_wav}")

        self.validate_initial_config()
        self.publish_status("idle")

    def validate_initial_config(self) -> None:
        if self.backend == "piper_rs":
            if not os.path.exists(self.piper_rs_bin):
                self.get_logger().warn(
                    f"piper_rs_bin does not exist: {self.piper_rs_bin}"
                )

            if not os.path.exists(self.config_path):
                self.get_logger().warn(
                    f"config_path does not exist: {self.config_path}"
                )

        elif self.backend == "piper_cli":
            if not self.model_path:
                self.get_logger().warn(
                    "model_path is empty. "
                    "Run with -p model_path:=/path/to/model.onnx"
                )

            if not os.path.exists(self.model_path):
                self.get_logger().warn(
                    f"model_path does not exist: {self.model_path}"
                )

            if self.config_path and not os.path.exists(self.config_path):
                self.get_logger().warn(
                    f"config_path does not exist: {self.config_path}"
                )

        else:
            self.get_logger().warn(
                f"Unknown backend={self.backend}. "
                "Valid values: piper_rs, piper_cli"
            )

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

    def build_piper_cli_command(self, output_wav: str) -> list[str]:
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

    def run_piper_rs(self, text: str) -> None:
        """
        neurlang/piper-rs interactive backend.

        예시 수동 실행:
          cd /home/hansungai/voicebot/piper_models
          printf "여기 산소마스크입니다.\\n" | \
            /home/hansungai/tools/piper-rs/target/release/examples/interactive \
            /home/hansungai/voicebot/piper_models/piper-kss-korean.onnx.json 80

        piper-rs interactive는 직접 오디오를 출력하므로 wav 재생 단계가 없다.
        """

        if not self.piper_rs_bin:
            raise RuntimeError("piper_rs_bin is empty.")

        if not os.path.exists(self.piper_rs_bin):
            raise RuntimeError(f"piper_rs_bin does not exist: {self.piper_rs_bin}")

        if not self.config_path:
            raise RuntimeError("config_path is empty for piper_rs backend.")

        if not os.path.exists(self.config_path):
            raise RuntimeError(f"config_path does not exist: {self.config_path}")

        config_dir = os.path.dirname(os.path.abspath(self.config_path))

        cmd = [
            self.piper_rs_bin,
            self.config_path,
            str(self.piper_rs_volume),
        ]

        self.get_logger().info(
            f"Running piper-rs: {' '.join(shlex.quote(x) for x in cmd)}"
        )
        self.get_logger().info(f"piper-rs cwd: {config_dir}")

        proc = subprocess.run(
            cmd,
            input=text + "\n",
            text=True,
            cwd=config_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"piper-rs failed with code {proc.returncode}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )

    def run_piper_cli(self, text: str, output_wav: str) -> None:
        """
        Python piper-tts CLI backend.
        일반 Piper voice 모델에서 사용한다.
        """

        if not self.model_path:
            raise RuntimeError("model_path is empty.")

        if not os.path.exists(self.model_path):
            raise RuntimeError(f"model_path does not exist: {self.model_path}")

        if self.config_path and not os.path.exists(self.config_path):
            raise RuntimeError(f"config_path does not exist: {self.config_path}")

        cmd = self.build_piper_cli_command(output_wav)

        self.get_logger().info(
            f"Running Piper CLI: {' '.join(shlex.quote(x) for x in cmd)}"
        )

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
                f"Piper CLI failed with code {proc.returncode}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )

        if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
            raise RuntimeError(f"Piper output wav was not created: {output_wav}")

    def play_wav(self, wav_path: str) -> None:
        if self.player == "none":
            self.get_logger().info(f"Playback disabled. wav_path={wav_path}")
            return

        if not self.player:
            raise RuntimeError("player is empty.")

        cmd = [self.player, wav_path]

        self.get_logger().info(
            f"Playing wav: {' '.join(shlex.quote(x) for x in cmd)}"
        )

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

    def synthesize_and_play(self, text: str) -> None:
        if self.backend == "piper_rs":
            self.publish_status("synthesizing_piper_rs")
            self.run_piper_rs(text)
            self.publish_status("played_by_piper_rs")
            return

        if self.backend == "piper_cli":
            self.publish_status("synthesizing_piper_cli")
            self.run_piper_cli(
                text=text,
                output_wav=self.output_wav,
            )

            self.publish_status("playing")
            self.play_wav(self.output_wav)
            return

        raise RuntimeError(
            f"Unknown backend={self.backend}. "
            "Valid values: piper_rs, piper_cli"
        )

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.get_logger().info(f"[TTS TEXT] {text}")
                self.synthesize_and_play(text)
                self.publish_status("idle")

                if self.cooldown_sec > 0:
                    time.sleep(self.cooldown_sec)

            except Exception as exc:
                self.get_logger().error(f"TTS worker error: {exc}")
                self.publish_status(f"error: {exc}")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping TTS node.")

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