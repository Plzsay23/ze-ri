#!/usr/bin/env python3
import math
import queue
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

import sounddevice as sd
import torch
from silero_vad import load_silero_vad


class SileroSpeechVadNode(Node):
    """
    Publishes human-speech VAD using Silero.

    Topics:
      /zeri/audio/speech_vad   std_msgs/Bool
      /zeri/audio/speech_prob  std_msgs/Float32
      /zeri/audio/rms          std_msgs/Float32

    This does NOT use ReSpeaker raw VAD.
    Keep using the existing ReSpeaker DOA node for /zeri/audio/doa.
    """

    def __init__(self):
        super().__init__("silero_speech_vad_node")

        self.declare_parameter("audio_device", -1)
        self.declare_parameter("channels", 6)
        self.declare_parameter("use_channel_index", 0)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("block_size", 512)

        self.declare_parameter("prob_on", 0.55)
        self.declare_parameter("prob_off", 0.35)
        self.declare_parameter("min_rms", 0.006)
        self.declare_parameter("hangover_frames", 8)

        self.audio_device = int(self.get_parameter("audio_device").value)
        self.channels = int(self.get_parameter("channels").value)
        self.use_channel_index = int(self.get_parameter("use_channel_index").value)
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.block_size = int(self.get_parameter("block_size").value)

        self.prob_on = float(self.get_parameter("prob_on").value)
        self.prob_off = float(self.get_parameter("prob_off").value)
        self.min_rms = float(self.get_parameter("min_rms").value)
        self.hangover_frames = int(self.get_parameter("hangover_frames").value)

        if self.sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8000 or 16000 Hz. Use sample_rate:=16000.")

        if self.use_channel_index < 0 or self.use_channel_index >= self.channels:
            raise ValueError("use_channel_index must be inside channel count.")

        torch.set_num_threads(1)
        self.model = load_silero_vad()
        self.model.eval()

        self.pub_vad = self.create_publisher(Bool, "/zeri/audio/speech_vad", 10)
        self.pub_prob = self.create_publisher(Float32, "/zeri/audio/speech_prob", 10)
        self.pub_rms = self.create_publisher(Float32, "/zeri/audio/rms", 10)

        self.audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self.running = True

        self.speech_state = False
        self.hang_count = 0

        self.worker = threading.Thread(target=self._vad_loop, daemon=True)
        self.worker.start()

        device_arg = None if self.audio_device < 0 else self.audio_device

        self.stream = sd.InputStream(
            device=device_arg,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()

        self.get_logger().info(
            "Silero speech VAD started | "
            f"device={self.audio_device} channels={self.channels} "
            f"use_channel_index={self.use_channel_index} "
            f"sample_rate={self.sample_rate} block_size={self.block_size} "
            f"prob_on={self.prob_on} prob_off={self.prob_off} min_rms={self.min_rms}"
        )

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"audio status: {status}")

        try:
            ch = np.asarray(indata[:, self.use_channel_index], dtype=np.float32).copy()
            if not self.audio_q.full():
                self.audio_q.put_nowait(ch)
        except Exception as exc:
            self.get_logger().warn(f"audio callback error: {exc}")

    def _vad_loop(self):
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(math.sqrt(float(np.mean(chunk * chunk)) + 1e-12))

            if rms < self.min_rms:
                prob = 0.0
            else:
                tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    prob = float(self.model(tensor, self.sample_rate).item())

            if self.speech_state:
                if prob >= self.prob_off and rms >= self.min_rms:
                    self.hang_count = self.hangover_frames
                else:
                    self.hang_count -= 1
                    if self.hang_count <= 0:
                        self.speech_state = False
            else:
                if prob >= self.prob_on and rms >= self.min_rms:
                    self.speech_state = True
                    self.hang_count = self.hangover_frames

            self.pub_vad.publish(Bool(data=bool(self.speech_state)))
            self.pub_prob.publish(Float32(data=float(prob)))
            self.pub_rms.publish(Float32(data=float(rms)))

    def destroy_node(self):
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = SileroSpeechVadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
