# zeri_vlm_stt_frame_mixin.py
import queue
import time
from typing import Optional, Tuple

from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

try:
    from .zeri_vlm_constants import MISSION_TRIAGE_DIALOGUE
    from .zeri_vlm_types import VLMRequest
except ImportError:
    from zeri_vlm_constants import MISSION_TRIAGE_DIALOGUE
    from zeri_vlm_types import VLMRequest


class ZeriVLMSTTFrameMixin:
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

            if self.is_stt_input_blocked():
                self.get_logger().info(
                    f"Ignored STT while pipeline is busy: {raw_stt_text}"
                )
                self.publish_inference_status("ignored_stt_pipeline_busy")
                return

            if not self.vad_allows_stt():
                self.get_logger().info(
                    f"Ignored STT because VAD is not active/recent: {raw_stt_text}"
                )
                self.publish_inference_status("ignored_stt_vad_false")
                return

            self.get_logger().info(f"Received raw STT text: {raw_stt_text}")

            accepted_text = self.filter_stt_text_for_vlm(raw_stt_text)

            if accepted_text is None:
                return

            self.get_logger().info(
                f"Accepted STT text for VLM: raw='{raw_stt_text}', command='{accepted_text}'"
            )
            self.publish_inference_status("accepted_stt_text")

            request = VLMRequest(
                stt_text=accepted_text,
                request_kind="stt_triage",
                mission_state=MISSION_TRIAGE_DIALOGUE,
                extra_context={"raw_stt_text": raw_stt_text},
            )
            self.enqueue_vlm_request(
                request,
                reason="accepted_stt_for_vlm",
                block_pipeline=True,
            )


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
