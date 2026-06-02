#!/usr/bin/env python3
import os
import re
import tempfile
import time
import wave
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def clean_sensevoice_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def extract_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return clean_sensevoice_text(result)

    if isinstance(result, dict):
        for key in ("text", "sentence", "result"):
            if key in result:
                return clean_sensevoice_text(result[key])
        return clean_sensevoice_text(str(result))

    if isinstance(result, (list, tuple)):
        parts = []
        for item in result:
            t = extract_text(item)
            if t:
                parts.append(t)
        return clean_sensevoice_text(" ".join(parts))

    return clean_sensevoice_text(str(result))


def write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    fd, path = tempfile.mkstemp(prefix="zeri_stt_", suffix=".wav")
    os.close(fd)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return path


class SenseVoiceSTTNode(Node):
    def __init__(self) -> None:
        super().__init__("zeri_sensevoice_stt_node")

        self.declare_parameter("model_dir", "/home/hansungai/ze-ri/models/sensevoice_ko")
        self.declare_parameter("audio_device", -1)
        self.declare_parameter("channels", 6)
        self.declare_parameter("use_channel_index", 0)
        self.declare_parameter("sample_rate", 16000)

        self.declare_parameter("block_sec", 0.20)
        self.declare_parameter("start_rms_threshold", 0.018)
        self.declare_parameter("rms_threshold", 0.010)
        self.declare_parameter("min_speech_sec", 0.45)
        self.declare_parameter("silence_sec", 0.80)
        self.declare_parameter("max_record_sec", 8.0)

        self.declare_parameter("language", "ko")
        self.declare_parameter("use_itn", True)
        self.declare_parameter("text_topic", "/stt/text")
        self.declare_parameter("status_topic", "/zeri/stt/status")
        self.declare_parameter("duplicate_window_sec", 1.5)

        self.model_dir = str(self.get_parameter("model_dir").value)
        self.audio_device = int(self.get_parameter("audio_device").value)
        self.channels = int(self.get_parameter("channels").value)
        self.use_channel_index = int(self.get_parameter("use_channel_index").value)
        self.sample_rate = int(self.get_parameter("sample_rate").value)

        self.block_sec = float(self.get_parameter("block_sec").value)
        self.start_rms_threshold = float(self.get_parameter("start_rms_threshold").value)
        self.rms_threshold = float(self.get_parameter("rms_threshold").value)
        self.min_speech_sec = float(self.get_parameter("min_speech_sec").value)
        self.silence_sec = float(self.get_parameter("silence_sec").value)
        self.max_record_sec = float(self.get_parameter("max_record_sec").value)

        self.language = str(self.get_parameter("language").value)
        self.use_itn = bool(self.get_parameter("use_itn").value)
        self.text_topic = str(self.get_parameter("text_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.duplicate_window_sec = float(self.get_parameter("duplicate_window_sec").value)

        self.text_pub = self.create_publisher(String, self.text_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.last_text = ""
        self.last_text_time = 0.0

        self.model = self.load_model()

        self.get_logger().info("Zeri SenseVoice STT node ready.")
        self.get_logger().info(f"model_dir={self.model_dir}")
        self.get_logger().info(f"audio_device={self.audio_device}")
        self.get_logger().info(f"channels={self.channels}")
        self.get_logger().info(f"use_channel_index={self.use_channel_index}")
        self.get_logger().info(f"sample_rate={self.sample_rate}")
        self.get_logger().info(f"text_topic={self.text_topic}")

    def publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[STT STATUS] {status}")

    def load_model(self) -> Any:
        self.publish_status("loading_model")

        try:
            from funasr_onnx import SenseVoiceSmall
        except Exception as exc:
            raise RuntimeError(f"funasr_onnx import failed: {exc}") from exc

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f"model_dir does not exist: {self.model_dir}")

        model_path = os.path.join(self.model_dir, "model.onnx")
        bpe_path = os.path.join(self.model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")

        if not os.path.exists(model_path):
            raise RuntimeError(f"missing model.onnx: {model_path}")
        if not os.path.exists(bpe_path):
            raise RuntimeError(f"missing bpe model: {bpe_path}")

        try:
            model = SenseVoiceSmall(self.model_dir, batch_size=1, quantize=False)
        except TypeError:
            model = SenseVoiceSmall(self.model_dir)

        self.publish_status("model_loaded")
        return model

    def transcribe_file(self, wav_path: str) -> str:
        errors = []

        calls = [
            lambda: self.model(audio_in=wav_path, language=self.language, use_itn=self.use_itn),
            lambda: self.model(audio_in=[wav_path], language=self.language, use_itn=self.use_itn),
            lambda: self.model(wav_path),
        ]

        for call in calls:
            try:
                result = call()
                text = extract_text(result)
                if text:
                    return text
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError("SenseVoice inference failed: " + " | ".join(errors))

    def publish_text(self, text: str) -> None:
        text = clean_sensevoice_text(text)
        if not text:
            return

        now = time.time()
        if text == self.last_text and now - self.last_text_time < self.duplicate_window_sec:
            self.publish_status("ignored_duplicate")
            return

        self.last_text = text
        self.last_text_time = now

        msg = String()
        msg.data = text
        self.text_pub.publish(msg)

        self.get_logger().info(f"[STT TEXT] {text}")
        self.publish_status("published_text")

    def run_loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(f"sounddevice import failed: {exc}") from exc

        block_size = max(1, int(self.sample_rate * self.block_sec))

        stream_kwargs = {
            "samplerate": self.sample_rate,
            "channels": self.channels,
            "blocksize": block_size,
            "dtype": "float32",
        }
        if self.audio_device >= 0:
            stream_kwargs["device"] = self.audio_device

        self.publish_status("listening")

        recording = False
        chunks = []
        speech_start = 0.0
        last_voice = 0.0

        with sd.InputStream(**stream_kwargs) as stream:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)

                data, overflowed = stream.read(block_size)
                if overflowed:
                    self.get_logger().warn("audio input overflowed")

                if data.ndim == 2:
                    ch = min(max(self.use_channel_index, 0), data.shape[1] - 1)
                    mono = np.asarray(data[:, ch], dtype=np.float32)
                else:
                    mono = np.asarray(data, dtype=np.float32).reshape(-1)

                rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
                now = time.time()

                if not recording:
                    if rms >= self.start_rms_threshold:
                        recording = True
                        chunks = [mono.copy()]
                        speech_start = now
                        last_voice = now
                        self.publish_status("recording")
                    continue

                chunks.append(mono.copy())

                if rms >= self.rms_threshold:
                    last_voice = now

                record_dur = now - speech_start
                silence_dur = now - last_voice

                should_finish = (
                    record_dur >= self.min_speech_sec and silence_dur >= self.silence_sec
                ) or record_dur >= self.max_record_sec

                if not should_finish:
                    continue

                audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

                recording = False
                chunks = []

                if audio.size < int(self.sample_rate * self.min_speech_sec):
                    self.publish_status("ignored_too_short")
                    continue

                wav_path: Optional[str] = None
                try:
                    self.publish_status("transcribing")
                    wav_path = write_temp_wav(audio, self.sample_rate)
                    text = self.transcribe_file(wav_path)
                    self.publish_text(text)
                    self.publish_status("listening")
                except Exception as exc:
                    self.get_logger().error(f"STT inference error: {exc}")
                    self.publish_status(f"error: {exc}")
                finally:
                    if wav_path and os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass


def main() -> None:
    rclpy.init()
    node = SenseVoiceSTTNode()
    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
