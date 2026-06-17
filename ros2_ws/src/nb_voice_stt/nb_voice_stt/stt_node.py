import os
import re
import time
import wave
import queue
import tempfile
import threading
from collections import deque

import numpy as np
import sounddevice as sd

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .stt_engine import STTEngine


class SenseVoiceSTTNode(Node):
    def __init__(self):
        super().__init__("sensevoice_stt_node")

        # -----------------------------
        # STT model params
        # -----------------------------
        self.declare_parameter(
            "model_dir",
            os.path.expanduser("~/ze-ri/models/sensevoice_ko"),
        )
        self.declare_parameter("model_name", "model.onnx")
        self.declare_parameter("language", "ko")
        self.declare_parameter("device", "cpu")

        # -----------------------------
        # Audio / endpointing params
        # -----------------------------
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("channels", 6)
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("silence_duration_ms", 800)
        self.declare_parameter("min_speech_ms", 250)
        self.declare_parameter("pre_roll_ms", 200)
        self.declare_parameter("energy_threshold", 0.01)

        # audio_device:
        #   -1  -> auto-detect ReSpeaker by device name
        #   >=0 -> use explicit sounddevice index
        self.declare_parameter("audio_device", -1)
        self.declare_parameter("use_channel_index", 0)

        # Preferred input name matching
        self.declare_parameter("preferred_input_name", "ReSpeaker 4 Mic Array")
        self.declare_parameter("fallback_input_name", "ArrayUAC10")

        # Optional debug save
        self.declare_parameter("save_segments", False)
        self.declare_parameter(
            "segment_output_dir",
            os.path.join(tempfile.gettempdir(), "stt_segments"),
        )

        # -----------------------------
        # stop interrupt spotter params
        # -----------------------------
        self.declare_parameter("enable_interrupt_spotter", True)
        self.declare_parameter("interrupt_window_ms", 700)
        self.declare_parameter("interrupt_hop_ms", 200)
        self.declare_parameter("interrupt_cooldown_ms", 1000)
        self.declare_parameter("interrupt_min_rms", 0.005)

        model_dir = str(self.get_parameter("model_dir").value)
        model_name = str(self.get_parameter("model_name").value)
        language = str(self.get_parameter("language").value)
        device = str(self.get_parameter("device").value)

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.channels = int(self.get_parameter("channels").value)
        self.frame_ms = int(self.get_parameter("frame_ms").value)
        self.silence_duration_ms = int(self.get_parameter("silence_duration_ms").value)
        self.min_speech_ms = int(self.get_parameter("min_speech_ms").value)
        self.pre_roll_ms = int(self.get_parameter("pre_roll_ms").value)
        self.energy_threshold = float(self.get_parameter("energy_threshold").value)

        audio_device_param = int(self.get_parameter("audio_device").value)
        self.use_channel_index = int(self.get_parameter("use_channel_index").value)

        self.preferred_input_name = str(
            self.get_parameter("preferred_input_name").value
        ).strip()
        self.fallback_input_name = str(
            self.get_parameter("fallback_input_name").value
        ).strip()

        self.save_segments = bool(self.get_parameter("save_segments").value)
        self.segment_output_dir = str(self.get_parameter("segment_output_dir").value)
        os.makedirs(self.segment_output_dir, exist_ok=True)

        self.enable_interrupt_spotter = bool(
            self.get_parameter("enable_interrupt_spotter").value
        )
        self.interrupt_window_ms = int(
            self.get_parameter("interrupt_window_ms").value
        )
        self.interrupt_hop_ms = int(
            self.get_parameter("interrupt_hop_ms").value
        )
        self.interrupt_cooldown_ms = int(
            self.get_parameter("interrupt_cooldown_ms").value
        )
        self.interrupt_min_rms = float(
            self.get_parameter("interrupt_min_rms").value
        )

        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.silence_frames = max(1, self.silence_duration_ms // self.frame_ms)
        self.min_speech_frames = max(1, self.min_speech_ms // self.frame_ms)
        self.pre_roll_frames = max(1, self.pre_roll_ms // self.frame_ms)

        self.text_pub = self.create_publisher(String, "/stt/text", 10)
        self.cmd_pub = self.create_publisher(String, "/voice_cmd", 10)
        self.interrupt_pub = self.create_publisher(String, "/voice_interrupt", 10)

        self.engine = STTEngine(
            model_dir=model_dir,
            model_name=model_name,
            language=language,
            device=device,
        )
        self.engine_lock = threading.Lock()

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self.interrupt_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self.utterance_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

        self._running = True
        self.last_interrupt_time = 0.0
        self.last_rms_log_time = 0.0

        # Resolve audio device
        if audio_device_param < 0:
            self.audio_device = self._find_respeaker_device()
        else:
            self.audio_device = audio_device_param

        # 긴 표현 -> 짧은 표현 순서
        self.command_rules = [
            # speed
            ("속도올려", "L"),
            ("속도증가", "L"),
            ("빠르게", "L"),
            ("속도내려", "M"),
            ("속도감소", "M"),
            ("천천히", "M"),

            # diagonal
            ("왼쪽앞으로", "R"),
            ("좌전방", "R"),
            ("왼쪽앞", "R"),

            ("오른쪽앞으로", "T"),
            ("우전방", "T"),
            ("오른쪽앞", "T"),

            ("왼쪽뒤로", "F"),
            ("좌후방", "F"),
            ("왼쪽뒤", "F"),

            ("오른쪽뒤로", "G"),
            ("우후방", "G"),
            ("오른쪽뒤", "G"),

            # strafe
            ("왼쪽이동", "Q"),
            ("좌측이동", "Q"),
            ("왼쪽으로이동", "Q"),

            ("오른쪽이동", "E"),
            ("우측이동", "E"),
            ("오른쪽으로이동", "E"),

            # basic move
            ("앞으로가", "W"),
            ("앞으로", "W"),
            ("전진", "W"),
            ("직진", "W"),

            ("뒤로가", "S"),
            ("뒤로", "S"),
            ("후진", "S"),

            ("좌회전", "A"),
            ("왼쪽회전", "A"),
            ("왼쪽으로", "A"),
            ("왼쪽", "A"),

            ("우회전", "D"),
            ("오른쪽회전", "D"),
            ("오른쪽으로", "D"),
            ("오른쪽", "D"),

            # fallback stop
            ("멈춰라", "X"),
            ("멈춰", "X"),
            ("정지", "X"),
            ("스톱", "X"),
            ("중지", "X"),
            ("멈", "x"),
            ("몸", "x"),
            ("뭄", "x"),
            ("야", "x"),
            ("섹스", "x"),
            ("그만", "x"),
        ]

        self.stop_keywords = ["멈춰", "멈춰라", "정지", "스톱", "중지", "멈", "뭄", "몸", "야", "섹스", "그만"]

        self.capture_thread = threading.Thread(
            target=self._run_audio_stream,
            daemon=True,
        )
        self.segment_thread = threading.Thread(
            target=self._segment_loop,
            daemon=True,
        )
        self.stt_thread = threading.Thread(
            target=self._stt_loop,
            daemon=True,
        )

        self.capture_thread.start()
        self.segment_thread.start()
        self.stt_thread.start()

        if self.enable_interrupt_spotter:
            self.interrupt_thread = threading.Thread(
                target=self._interrupt_loop,
                daemon=True,
            )
            self.interrupt_thread.start()

        self.get_logger().info("SenseVoice streaming STT node started")
        self.get_logger().info(f"model_dir={model_dir}")
        self.get_logger().info(
            f"audio_device={self.audio_device} ({self._describe_device(self.audio_device)})"
        )
        self.get_logger().info(
            f"channels={self.channels}, use_channel_index={self.use_channel_index}"
        )
        self.get_logger().info(
            f"interrupt_spotter={self.enable_interrupt_spotter}, "
            f"window={self.interrupt_window_ms}ms, hop={self.interrupt_hop_ms}ms"
        )

    def _find_respeaker_device(self) -> int:
        devices = sd.query_devices()

        matches = []
        fallback_matches = []

        for idx, dev in enumerate(devices):
            name = str(dev.get("name", ""))
            max_in = int(dev.get("max_input_channels", 0))

            if max_in <= 0:
                continue

            if self.preferred_input_name and self.preferred_input_name in name:
                matches.append((idx, name, max_in))
            elif self.fallback_input_name and self.fallback_input_name in name:
                fallback_matches.append((idx, name, max_in))

        if matches:
            idx, name, max_in = matches[0]
            self.get_logger().info(
                f"auto-selected preferred input device: idx={idx}, name='{name}', max_input_channels={max_in}"
            )
            return idx

        if fallback_matches:
            idx, name, max_in = fallback_matches[0]
            self.get_logger().info(
                f"auto-selected fallback input device: idx={idx}, name='{name}', max_input_channels={max_in}"
            )
            return idx

        # 마지막 디버그용으로 입력 가능 장치 목록 로그
        self.get_logger().error("ReSpeaker input device not found. Input-capable devices:")
        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0:
                self.get_logger().error(
                    f"  idx={idx}, name='{dev.get('name', '')}', max_input_channels={max_in}"
                )

        raise RuntimeError(
            "ReSpeaker input device not found by name. "
            "Check sounddevice list or set explicit audio_device index."
        )

    def _describe_device(self, device_index: int) -> str:
        try:
            dev = sd.query_devices(device_index)
            return str(dev.get("name", "unknown"))
        except Exception:
            return "unknown"

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"audio status: {status}")

        if indata.ndim == 1:
            chunk = indata.copy().astype(np.float32)
        else:
            ch = min(self.use_channel_index, indata.shape[1] - 1)
            chunk = indata[:, ch].copy().astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        now = time.monotonic()
        if now - self.last_rms_log_time > 1.0:
            self.last_rms_log_time = now
            self.get_logger().info(f"audio rms={rms:.6f}")

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

        if self.enable_interrupt_spotter:
            try:
                self.interrupt_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _run_audio_stream(self):
        while self._running and rclpy.ok():
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.frame_samples,
                    callback=self._audio_callback,
                    device=self.audio_device,
                ):
                    while self._running and rclpy.ok():
                        time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"audio stream error: {e}")
                time.sleep(1.0)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip().lower()
        t = re.sub(r"\s+", "", t)
        t = re.sub(r"[^\w가-힣]", "", t)
        return t

    def _normalize_command(self, text: str):
        t = self._clean_text(text)
        for phrase, cmd in self.command_rules:
            if phrase in t:
                return cmd
        return None

    def _contains_stop_keyword(self, text: str) -> bool:
        t = self._clean_text(text)
        return any(keyword in t for keyword in self.stop_keywords)

    def _is_speech(self, chunk: np.ndarray) -> bool:
        if chunk is None or len(chunk) == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        return rms >= self.energy_threshold

    def _save_audio_to_wav(self, audio: np.ndarray) -> str:
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.segment_output_dir,
            f"utt_{ts}_{int(time.time() * 1000) % 1000:03d}.wav",
        )

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())

        return path

    def _segment_loop(self):
        pre_buffer = []
        speech_buffer = []
        in_speech = False
        silence_count = 0
        speech_frame_count = 0

        while self._running and rclpy.ok():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if len(pre_buffer) >= self.pre_roll_frames:
                pre_buffer.pop(0)
            pre_buffer.append(chunk)

            speech = self._is_speech(chunk)

            if not in_speech:
                if speech:
                    in_speech = True
                    speech_buffer = pre_buffer.copy()
                    speech_frame_count = len(speech_buffer)
                    silence_count = 0
            else:
                speech_buffer.append(chunk)
                speech_frame_count += 1

                if speech:
                    silence_count = 0
                else:
                    silence_count += 1

                if silence_count >= self.silence_frames:
                    if speech_frame_count >= self.min_speech_frames:
                        audio = np.concatenate(speech_buffer, axis=0).astype(np.float32)
                        try:
                            self.utterance_queue.put_nowait(audio)
                        except queue.Full:
                            self.get_logger().warn("utterance queue full, dropping segment")
                    in_speech = False
                    speech_buffer = []
                    silence_count = 0
                    speech_frame_count = 0

    def _publish_text(self, text: str):
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)

    def _publish_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)

    def _publish_interrupt(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.interrupt_pub.publish(msg)

    def _stt_loop(self):
        while self._running and rclpy.ok():
            try:
                audio = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.save_segments:
                    wav_path = self._save_audio_to_wav(audio)
                    self.get_logger().info(f"saved segment: {wav_path}")

                with self.engine_lock:
                    text = self.engine.transcribe_audio(audio, sr=self.sample_rate)

                if not text:
                    self.get_logger().info("(인식 결과 없음)")
                    continue

                self._publish_text(text)
                self.get_logger().info(f"recognized: {text}")

                cmd = self._normalize_command(text)
                if cmd is not None:
                    self._publish_cmd(cmd)
                    self.get_logger().info(f"voice_cmd: {cmd}")

                    # fallback stop
                    if cmd == "X":
                        self._publish_interrupt("stop")
                        self.get_logger().warn("[INTERRUPT-FALLBACK] stop")
                else:
                    self.get_logger().warn(f"no command matched: {text}")

            except Exception as e:
                self.get_logger().error(f"STT loop error: {e}")
                time.sleep(0.2)

    def _interrupt_loop(self):
        window_frames = int(self.sample_rate * self.interrupt_window_ms / 1000)
        hop_blocks = max(1, self.interrupt_hop_ms // self.frame_ms)
        rolling = deque()
        rolling_frames = 0
        block_count = 0

        while self._running and rclpy.ok():
            try:
                chunk = self.interrupt_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rolling.append(chunk)
            rolling_frames += len(chunk)
            block_count += 1

            while rolling_frames > window_frames and len(rolling) > 1:
                old = rolling.popleft()
                rolling_frames -= len(old)

            if block_count < hop_blocks:
                continue

            block_count = 0

            if rolling_frames < max(window_frames // 2, self.frame_samples):
                continue

            audio = np.concatenate(list(rolling), axis=0).astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            if rms < self.interrupt_min_rms:
                continue

            try:
                with self.engine_lock:
                    text = self.engine.transcribe_audio(audio, sr=self.sample_rate)
            except Exception as e:
                self.get_logger().error(f"interrupt spotter error: {e}")
                continue

            if not text:
                continue

            if self._contains_stop_keyword(text):
                now = time.monotonic()
                elapsed_ms = (now - self.last_interrupt_time) * 1000.0
                if elapsed_ms >= self.interrupt_cooldown_ms:
                    self.last_interrupt_time = now
                    self._publish_interrupt("stop")
                    self.get_logger().warn(f"[INTERRUPT] stop detected: {text}")

    def destroy_node(self):
        self._running = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SenseVoiceSTTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
