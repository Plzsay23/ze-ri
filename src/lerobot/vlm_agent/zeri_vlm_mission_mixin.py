# zeri_vlm_mission_mixin.py
import json
import queue
import time
from typing import Any, Dict, Optional

from std_msgs.msg import String

try:
    from .zeri_vlm_constants import (
        MISSION_APPROACH_PERSON,
        MISSION_RESUME_SEARCH,
        MISSION_SEARCH_PERSON,
        MISSION_SELECT_TARGET,
        MISSION_STOP_AT_DISTANCE,
        MISSION_TRIAGE_DIALOGUE,
        MISSION_VERIFY_HANDOFF,
        SUPPORTED_VLA_TASKS,
        VALID_NAV_INTENTS,
    )
    from .zeri_vlm_types import VLMRequest
except ImportError:
    from zeri_vlm_constants import (
        MISSION_APPROACH_PERSON,
        MISSION_RESUME_SEARCH,
        MISSION_SEARCH_PERSON,
        MISSION_SELECT_TARGET,
        MISSION_STOP_AT_DISTANCE,
        MISSION_TRIAGE_DIALOGUE,
        MISSION_VERIFY_HANDOFF,
        SUPPORTED_VLA_TASKS,
        VALID_NAV_INTENTS,
    )
    from zeri_vlm_types import VLMRequest


class ZeriVLMMissionMixin:
    def parse_json_payload(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if not raw:
            return {}

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"event": raw}

    def enqueue_vlm_request(
        self,
        request: VLMRequest,
        reason: str,
        block_pipeline: bool = True,
    ) -> bool:
        if block_pipeline and self.is_stt_input_blocked():
            self.get_logger().info(
                f"Ignored VLM request while pipeline is busy: {reason}"
            )
            self.publish_inference_status(
                f"ignored_vlm_request_pipeline_busy: {reason}"
            )
            return False

        if block_pipeline:
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
        data = self.parse_json_payload(msg.data)

        if not data:
            self.publish_inference_status("ignored_empty_target_context")
            return

        self.latest_target_context = data
        self.latest_target_context_time = time.time()

        selected_person_id = str(
            data.get("selected_person_id", data.get("tracking_id", "none"))
        )

        self.get_logger().info(f"[TARGET CONTEXT] {json.dumps(data, ensure_ascii=False)}")

        if selected_person_id and selected_person_id != "none":
            self.publish_mission_state(
                MISSION_SELECT_TARGET,
                reason="target_context_received",
            )
            self.publish_nav_intent(
                "approach_person",
                mission_state=MISSION_APPROACH_PERSON,
                selected_person_id=selected_person_id,
                reason="target_context_received",
                source="target_context",
            )

    def mission_event_callback(self, msg: String) -> None:
        if not self.enable_mission_events:
            self.publish_inference_status("ignored_mission_event_disabled")
            return

        data = self.parse_json_payload(msg.data)

        if not data:
            self.publish_inference_status("ignored_empty_mission_event")
            return

        event = str(data.get("event", "")).strip()
        if not event:
            event = str(data.get("type", "")).strip()

        event = event.lower()

        self.latest_mission_event = data
        self.latest_mission_event_time = time.time()

        target_context = data.get("target_context")
        if isinstance(target_context, dict):
            self.latest_target_context = target_context
            self.latest_target_context_time = time.time()

        selected_person_id = str(
            data.get(
                "selected_person_id",
                self.latest_target_context.get("selected_person_id", "none"),
            )
        )

        self.get_logger().info(f"[MISSION EVENT] {json.dumps(data, ensure_ascii=False)}")

        if event in {"person_detected", "target_detected", "target_selected"}:
            self.publish_mission_state(
                MISSION_APPROACH_PERSON,
                reason=event,
            )
            self.publish_nav_intent(
                "approach_person",
                mission_state=MISSION_APPROACH_PERSON,
                selected_person_id=selected_person_id,
                reason=event,
                source="mission_event",
            )
            return

        if event in {"arrived_at_person", "stop_at_distance", "person_reached"}:
            self.publish_mission_state(
                MISSION_STOP_AT_DISTANCE,
                reason=event,
            )
            self.publish_nav_intent(
                "hold_position",
                mission_state=MISSION_STOP_AT_DISTANCE,
                selected_person_id=selected_person_id,
                reason=event,
                source="mission_event",
            )

            request = VLMRequest(
                stt_text=(
                    "요구조자 앞에 도착했다. "
                    "의식과 호흡 상태를 확인하는 첫 문진을 시작해라."
                ),
                request_kind="initial_contact",
                mission_state=MISSION_TRIAGE_DIALOGUE,
                extra_context={
                    "event": event,
                    "mission_event": data,
                    "selected_person_id": selected_person_id,
                },
            )
            self.enqueue_vlm_request(
                request,
                reason=f"mission_event_{event}",
                block_pipeline=True,
            )
            return

        if event in {"handoff_check", "vla_succeeded", "verify_handoff"}:
            self.publish_mission_state(
                MISSION_VERIFY_HANDOFF,
                reason=event,
            )
            self.publish_nav_intent(
                "hold_position",
                mission_state=MISSION_VERIFY_HANDOFF,
                selected_person_id=selected_person_id,
                reason=event,
                source="mission_event",
            )

            request = VLMRequest(
                stt_text=(
                    "VLA 동작이 끝났다. 카메라를 보고 사람이 물건을 실제로 "
                    "받았는지 확인해라."
                ),
                request_kind="verify_handoff",
                mission_state=MISSION_VERIFY_HANDOFF,
                extra_context={
                    "event": event,
                    "mission_event": data,
                    "selected_person_id": selected_person_id,
                },
            )
            self.enqueue_vlm_request(
                request,
                reason=f"mission_event_{event}",
                block_pipeline=True,
            )
            return

        if event in {"resume_search", "done", "next_person"}:
            self.publish_mission_state(
                MISSION_RESUME_SEARCH,
                reason=event,
            )
            self.publish_nav_intent(
                "rotate_search",
                mission_state=MISSION_SEARCH_PERSON,
                selected_person_id="none",
                reason=event,
                source="mission_event",
            )
            return

        if event in {"search_person", "no_person", "lost_person"}:
            self.publish_mission_state(
                MISSION_SEARCH_PERSON,
                reason=event,
            )
            self.publish_nav_intent(
                "rotate_search",
                mission_state=MISSION_SEARCH_PERSON,
                selected_person_id="none",
                reason=event,
                source="mission_event",
            )
            return

        if event in {"manual_triage", "triage_dialogue"}:
            text = str(
                data.get(
                    "stt_text",
                    "요구조자의 상태를 소방 초동 대응 원칙에 따라 확인해라.",
                )
            )

            request = VLMRequest(
                stt_text=text,
                request_kind="manual_triage",
                mission_state=MISSION_TRIAGE_DIALOGUE,
                extra_context={
                    "event": event,
                    "mission_event": data,
                    "selected_person_id": selected_person_id,
                },
            )
            self.enqueue_vlm_request(
                request,
                reason=f"mission_event_{event}",
                block_pipeline=True,
            )
            return

        self.publish_inference_status(f"ignored_unknown_mission_event: {event}")

    def build_mission_context(self, request: VLMRequest) -> Dict[str, Any]:
        now = time.time()

        target_age_sec: Optional[float] = None
        if self.latest_target_context_time > 0.0:
            target_age_sec = round(now - self.latest_target_context_time, 3)

        mission_event_age_sec: Optional[float] = None
        if self.latest_mission_event_time > 0.0:
            mission_event_age_sec = round(now - self.latest_mission_event_time, 3)

        return {
            "current_mission_state": self.mission_state,
            "request_mission_state": request.mission_state,
            "request_kind": request.request_kind,
            "request_extra_context": request.extra_context,
            "latest_target_context": self.latest_target_context,
            "latest_target_context_age_sec": target_age_sec,
            "latest_mission_event": self.latest_mission_event,
            "latest_mission_event_age_sec": mission_event_age_sec,
            "supported_vla_tasks": sorted(SUPPORTED_VLA_TASKS),
            "valid_nav_intents": sorted(VALID_NAV_INTENTS),
            "enable_vla": self.enable_vla,
            "verify_handoff_after_vla_success": self.verify_handoff_after_vla_success,
        }
