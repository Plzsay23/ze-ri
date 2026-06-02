# zeri_vlm_pipeline_mixin.py
import json
import time

from std_msgs.msg import Bool, Float32, String

try:
    from .zeri_vlm_constants import (
        MISSION_RESUME_SEARCH, MISSION_RETURN_ARM_HOME, MISSION_VERIFY_HANDOFF
    )
except ImportError:
    from zeri_vlm_constants import (
        MISSION_RESUME_SEARCH, MISSION_RETURN_ARM_HOME, MISSION_VERIFY_HANDOFF
    )


class ZeriVLMPipelineMixin:
        def begin_pipeline_block(self, reason: str) -> None:
            with self.pipeline_lock:
                self.pipeline_busy = True

                self.waiting_for_tts = False
                self.tts_active = False
                self.tts_deadline = time.time() + self.tts_max_wait_sec
                self.stt_block_until = 0.0

                self.waiting_for_vla = False
                self.vla_active = False
                self.vla_deadline = 0.0
                self.active_vla_task_id = None

            self.publish_stt_mute(True)
            self.publish_inference_status(f"stt_blocked: {reason}")


        def release_pipeline_block(self, reason: str) -> None:
            cooldown_until = time.time() + self.stt_block_after_tts_sec

            with self.pipeline_lock:
                self.pipeline_busy = False

                self.waiting_for_tts = False
                self.tts_active = False
                self.tts_deadline = 0.0

                self.waiting_for_vla = False
                self.vla_active = False
                self.vla_deadline = 0.0
                self.active_vla_task_id = None

                self.stt_block_until = cooldown_until

            self.publish_inference_status(f"stt_block_cooldown: {reason}")

            if self.stt_block_after_tts_sec <= 0:
                self.publish_stt_mute(False)


        def mark_waiting_for_tts(self) -> None:
            with self.pipeline_lock:
                self.waiting_for_tts = True
                self.tts_deadline = time.time() + self.tts_max_wait_sec

            self.publish_inference_status("waiting_for_tts")


        def mark_tts_done_and_maybe_release(self, reason: str) -> None:
            should_release = False

            with self.pipeline_lock:
                self.waiting_for_tts = False
                self.tts_active = False
                self.tts_deadline = 0.0

                if not self.waiting_for_vla and not self.vla_active:
                    should_release = True

            if should_release:
                self.release_pipeline_block(reason)
            else:
                self.publish_inference_status(f"tts_done_waiting_for_vla: {reason}")


        def finish_vla_and_maybe_release(self, reason: str, success: bool) -> None:
            should_release = False

            with self.pipeline_lock:
                self.waiting_for_vla = False
                self.vla_active = False
                self.vla_deadline = 0.0
                self.active_vla_task_id = None

                if not self.waiting_for_tts and not self.tts_active:
                    should_release = True

            if success:
                self.publish_led(self.led_on_vla_success)
            else:
                self.publish_led(self.led_on_error)

            if should_release:
                self.release_pipeline_block(reason)
            else:
                self.publish_inference_status(f"vla_done_waiting_for_tts: {reason}")


        def is_stt_input_blocked(self) -> bool:
            now = time.time()

            with self.pipeline_lock:
                if self.pipeline_busy:
                    return True

                if self.tts_active:
                    return True

                if self.waiting_for_vla or self.vla_active:
                    return True

                if now < self.stt_block_until:
                    return True

            return False


        def pipeline_timer_callback(self) -> None:
            now = time.time()

            should_tts_timeout = False
            should_vla_timeout = False
            should_unmute = False

            with self.pipeline_lock:
                should_tts_timeout = (
                    self.waiting_for_tts
                    and self.tts_deadline > 0.0
                    and now > self.tts_deadline
                )

                should_vla_timeout = (
                    self.waiting_for_vla
                    and self.vla_deadline > 0.0
                    and now > self.vla_deadline
                )

                should_unmute = (
                    self.stt_mute_state
                    and not self.pipeline_busy
                    and not self.waiting_for_tts
                    and not self.tts_active
                    and not self.waiting_for_vla
                    and not self.vla_active
                    and self.stt_block_until > 0.0
                    and now >= self.stt_block_until
                )

            if should_vla_timeout:
                self.get_logger().warn("VLA wait timeout. Returning to LISTEN state.")
                self.publish_inference_status("vla_wait_timeout_returning_to_listen")
                self.finish_vla_and_maybe_release("vla_wait_timeout", success=False)
                return

            if should_tts_timeout:
                self.get_logger().warn("TTS wait timeout.")
                self.mark_tts_done_and_maybe_release("tts_timeout")
                return

            if should_unmute:
                self.publish_stt_mute(False)
                self.publish_inference_status("stt_unblocked_after_cooldown")


        def tts_status_callback(self, msg: String) -> None:
            status = msg.data.strip()

            active_statuses = {
                "queued",
                "queued_after_drop",
                "synthesizing",
                "synthesizing_edge_tts",
                "synthesizing_piper_rs",
                "synthesizing_piper_cli",
                "playing",
            }

            idle_statuses = {
                "idle",
                "played_by_piper_rs",
            }

            error_like = status.startswith("error")

            if status in active_statuses:
                with self.pipeline_lock:
                    self.tts_active = True
                    self.waiting_for_tts = True
                    self.tts_deadline = time.time() + self.tts_max_wait_sec

                self.publish_inference_status(f"tts_active: {status}")
                return

            if status in idle_statuses or error_like:
                with self.pipeline_lock:
                    was_blocking = (
                        self.pipeline_busy
                        or self.waiting_for_tts
                        or self.tts_active
                        or self.waiting_for_vla
                        or self.vla_active
                    )

                if was_blocking:
                    self.mark_tts_done_and_maybe_release(f"tts_status_{status}")

                return


        def vla_status_callback(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Invalid VLA status JSON: {msg.data}")
                return

            task_id = str(data.get("task_id", "")).strip()
            status = str(data.get("status", "")).strip()
            reason = str(data.get("reason", "")).strip()

            with self.pipeline_lock:
                active_task_id = self.active_vla_task_id

            if not active_task_id:
                if status not in {"idle"}:
                    self.get_logger().info(
                        f"Ignored VLA status because no active VLA task: {msg.data}"
                    )
                return

            if task_id not in {active_task_id, "none"}:
                self.get_logger().info(
                    f"Ignored VLA status for non-active task: "
                    f"status_task_id={task_id}, active_task_id={active_task_id}"
                )
                return

            self.get_logger().info(f"[VLA STATUS RX] {msg.data}")

            if status in {"accepted", "running"}:
                with self.pipeline_lock:
                    self.waiting_for_vla = True
                    self.vla_active = True
                    self.vla_deadline = time.time() + self.vla_timeout_sec

                self.publish_led(self.led_on_vla_running)
                self.publish_inference_status(f"vla_{status}")
                return

            if status == "succeeded":
                if self.verify_handoff_after_vla_success:
                    with self.pipeline_lock:
                        self.waiting_for_vla = False
                        self.vla_active = False
                        self.vla_deadline = 0.0
                        self.active_vla_task_id = None

                    self.publish_led(self.led_on_vla_success)
                    self.publish_mission_state(
                        MISSION_VERIFY_HANDOFF,
                        reason="vla_succeeded",
                    )
                    self.publish_inference_status("vla_succeeded_start_handoff_verify")

                    request = VLMRequest(
                        stt_text=(
                            "VLA 동작이 끝났다. 카메라를 보고 사람이 물건을 "
                            "실제로 받았는지 확인해라."
                        ),
                        request_kind="verify_handoff",
                        mission_state=MISSION_VERIFY_HANDOFF,
                        extra_context={
                            "vla_status": data,
                            "previous_task_id": task_id,
                        },
                    )
                    self.enqueue_vlm_request(
                        request,
                        reason="vla_succeeded_handoff_verify",
                        block_pipeline=False,
                    )
                    return

                self.publish_inference_status("vla_succeeded_returning_to_listen")
                self.finish_vla_and_maybe_release("vla_succeeded", success=True)
                return

            if status in {"failed", "timeout", "rejected"}:
                self.publish_inference_status(
                    f"vla_{status}_returning_to_listen: {reason}"
                )
                self.finish_vla_and_maybe_release(f"vla_{status}", success=False)
                return


        def vad_callback(self, msg: Bool) -> None:
            self.latest_vad = bool(msg.data)

            if self.latest_vad:
                self.latest_vad_time = time.time()


        def doa_callback(self, msg: Float32) -> None:
            self.latest_doa_deg = float(msg.data)
            self.latest_doa_time = time.time()


        def vad_allows_stt(self) -> bool:
            if not self.use_vad_gate:
                return True

            now = time.time()
            return (now - self.latest_vad_time) <= self.vad_hold_sec
