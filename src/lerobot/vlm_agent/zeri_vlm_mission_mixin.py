# zeri_vlm_mission_mixin.py
import json
import queue
import time
from typing import Any, Dict

from std_msgs.msg import String

try:
    from .zeri_vlm_constants import (
        MISSION_APPROACH_PERSON, MISSION_SEARCH_PERSON, MISSION_SELECT_TARGET,
        MISSION_STOP_AT_DISTANCE, MISSION_TRIAGE_DIALOGUE, MISSION_VERIFY_HANDOFF,
    )
    from .zeri_vlm_types import VLMRequest
except ImportError:
    from zeri_vlm_constants import (
        MISSION_APPROACH_PERSON, MISSION_SEARCH_PERSON, MISSION_SELECT_TARGET,
        MISSION_STOP_AT_DISTANCE, MISSION_TRIAGE_DIALOGUE, MISSION_VERIFY_HANDOFF,
    )
    from zeri_vlm_types import VLMRequest


class ZeriVLMMissionMixin:
        def build_mission_context(self, request: VLMRequest) -> Dict[str, Any]:
            target_age_sec = None
            if self.latest_target_context_time > 0.0:
                target_age_sec = round(time.time() - self.latest_target_context_time, 3)

            event_age_sec = None
            if self.latest_mission_event_time > 0.0:
                event_age_sec = round(time.time() - self.latest_mission_event_time, 3)

            doa_age_sec = None
            if self.latest_doa_time > 0.0:
                doa_age_sec = round(time.time() - self.latest_doa_time, 3)

            return {
                "current_mission_state": self.mission_state,
                "request_mission_state": request.mission_state,
                "request_kind": request.request_kind,
                "extra_context": request.extra_context,
                "latest_target_context": self.latest_target_context,
                "latest_target_context_age_sec": target_age_sec,
                "latest_mission_event": self.latest_mission_event,
                "latest_mission_event_age_sec": event_age_sec,
                "latest_doa_deg": self.latest_doa_deg,
                "latest_doa_age_sec": doa_age_sec,
                "latest_vad": self.latest_vad,
                "vla_active": self.vla_active,
                "waiting_for_vla": self.waiting_for_vla,
                "active_vla_task_id": self.active_vla_task_id,
                "supported_vla_tasks": sorted(SUPPORTED_VLA_TASKS),
                "valid_nav_intents": sorted(VALID_NAV_INTENTS),
            }


        def enqueue_vlm_request(
            self,
            request: VLMRequest,
            reason: str,
            block_pipeline: bool = True,
        ) -> bool:
            if block_pipeline:
                if self.is_stt_input_blocked():
                    self.get_logger().info(
                        f"Ignored VLM request while pipeline is busy: {reason}, "
                        f"request_kind={request.request_kind}"
                    )
                    self.publish_inference_status("ignored_vlm_request_pipeline_busy")
                    return False

                self.begin_pipeline_block(reason)

            try:
                self.text_queue.put_nowait(request)
            except queue.Full:
                try:
                    dropped = self.text_queue.get_nowait()
                    self.get_logger().warn(
                        f"Dropped old VLM request due to full queue: {dropped}"
                    )
                except queue.Empty:
                    pass

                try:
                    self.text_queue.put_nowait(request)
                except queue.Full:
                    self.get_logger().error("Failed to enqueue VLM request.")
                    self.publish_inference_status("queue_full_error")
                    self.publish_led(self.led_on_error)
                    if block_pipeline:
                        self.release_pipeline_block("queue_full_error")
                    return False

            self.get_logger().info(
                f"[VLM REQUEST] kind={request.request_kind}, "
                f"state={request.mission_state}, text={request.stt_text}"
            )
            return True


        def target_context_callback(self, msg: String) -> None:
            text = msg.data.strip()
            if not text:
                return

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Invalid target context JSON: {text}")
                return

            self.latest_target_context = data
            self.latest_target_context_time = time.time()
            self.get_logger().info(f"[TARGET CONTEXT] {text}")


        def mission_event_callback(self, msg: String) -> None:
            if not self.enable_mission_events:
                return

            text = msg.data.strip()
            if not text:
                return

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Invalid mission event JSON: {text}")
                return

            event = str(data.get("event", "")).strip().lower()
            self.latest_mission_event = data
            self.latest_mission_event_time = time.time()

            if "target_context" in data and isinstance(data["target_context"], dict):
                self.latest_target_context = data["target_context"]
                self.latest_target_context_time = time.time()

            self.get_logger().info(f"[MISSION EVENT] {text}")

            if event in {"person_detected", "target_selected"}:
                selected_person_id = str(
                    data.get("selected_person_id", data.get("target_person_id", "none"))
                )
                self.publish_mission_state(MISSION_APPROACH_PERSON, reason=event)
                self.publish_nav_intent(
                    "approach_person",
                    mission_state=MISSION_APPROACH_PERSON,
                    selected_person_id=selected_person_id,
                    reason=event,
                    source="mission_event",
                )
                return

            if event in {"person_lost", "target_lost", "search_person"}:
                self.publish_mission_state(MISSION_SEARCH_PERSON, reason=event)
                self.publish_nav_intent(
                    "rotate_search",
                    mission_state=MISSION_SEARCH_PERSON,
                    reason=event,
                    source="mission_event",
                )
                return

            if event in {"arrived_at_person", "stop_at_distance"}:
                self.publish_mission_state(MISSION_STOP_AT_DISTANCE, reason=event)
                self.publish_nav_intent(
                    "hold_position",
                    mission_state=MISSION_STOP_AT_DISTANCE,
                    selected_person_id=str(
                        data.get("selected_person_id", data.get("target_person_id", "none"))
                    ),
                    reason=event,
                    source="mission_event",
                )

                request = VLMRequest(
                    stt_text="요구조자 앞에 도착했다. 의식과 호흡 상태를 확인하는 첫 문진을 시작해라.",
                    request_kind="initial_contact",
                    mission_state=MISSION_TRIAGE_DIALOGUE,
                    extra_context=data,
                )
                self.enqueue_vlm_request(request, reason=event, block_pipeline=True)
                return

            if event in {"request_vlm_triage", "triage_dialogue"}:
                request = VLMRequest(
                    stt_text=str(data.get("stt_text", "요구조자의 상태를 매뉴얼 기준으로 판단해라.")),
                    request_kind="triage_dialogue",
                    mission_state=MISSION_TRIAGE_DIALOGUE,
                    extra_context=data,
                )
                self.enqueue_vlm_request(request, reason=event, block_pipeline=True)
                return

            if event in {"handoff_check", "verify_handoff"}:
                request = VLMRequest(
                    stt_text="VLA 동작이 끝났다. 카메라를 보고 사람이 물건을 실제로 받았는지 확인해라.",
                    request_kind="verify_handoff",
                    mission_state=MISSION_VERIFY_HANDOFF,
                    extra_context=data,
                )
                self.enqueue_vlm_request(request, reason=event, block_pipeline=True)
                return

            if event in {"resume_search", "task_done"}:
                self.publish_mission_state(MISSION_RESUME_SEARCH, reason=event)
                self.publish_nav_intent(
                    "rotate_search",
                    mission_state=MISSION_RESUME_SEARCH,
                    reason=event,
                    source="mission_event",
                )
                self.publish_mission_state(MISSION_SEARCH_PERSON, reason="resume_search_done")
                return

            if bool(data.get("trigger_vlm", False)):
                request = VLMRequest(
                    stt_text=str(data.get("stt_text", "현재 상황을 판단해라.")),
                    request_kind=str(data.get("request_kind", "mission_event")),
                    mission_state=str(data.get("mission_state", self.mission_state)).upper(),
                    extra_context=data,
                )
                self.enqueue_vlm_request(request, reason=event or "trigger_vlm", block_pipeline=True)
