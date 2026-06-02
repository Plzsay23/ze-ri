# zeri_vlm_publish_mixin.py
import json
import time
from typing import Optional

from std_msgs.msg import Bool, Int32, String

try:
    from .zeri_vlm_constants import (
        LED_NAME_MAP, LED_OFF, MISSION_MARK_AND_REPORT, MISSION_RETURN_ARM_HOME, MISSION_RUN_VLA,
        MISSION_TRIAGE_DIALOGUE, SUPPORTED_VLA_TASKS, VALID_MISSION_STATES, VALID_NAV_INTENTS
    )
    from .zeri_vlm_decision import clamp_led_cmd
    from .zeri_vlm_types import VLMDecision, VLMRequest, VLMRequest
except ImportError:
    from zeri_vlm_constants import (
        LED_NAME_MAP, LED_OFF, MISSION_MARK_AND_REPORT, MISSION_RETURN_ARM_HOME, MISSION_RUN_VLA,
        MISSION_TRIAGE_DIALOGUE, SUPPORTED_VLA_TASKS, VALID_MISSION_STATES, VALID_NAV_INTENTS
    )
    from zeri_vlm_decision import clamp_led_cmd
    from zeri_vlm_types import VLMDecision, VLMRequest, VLMRequest


class ZeriVLMPublishMixin:
        def publish_led(self, value: int) -> None:
            led_cmd = clamp_led_cmd(value, default=LED_OFF)

            msg = Int32()
            msg.data = led_cmd
            self.led_publisher.publish(msg)

            led_name = LED_NAME_MAP.get(led_cmd, "UNKNOWN")
            self.get_logger().info(
                f"[LED] publish {self.led_topic} = {led_cmd} ({led_name})"
            )


        def publish_inference_status(self, status: str) -> None:
            msg = String()
            msg.data = status
            self.inference_status_publisher.publish(msg)
            self.get_logger().info(f"[VLM STATUS] {status}")


        def publish_stt_mute(self, muted: bool) -> None:
            with self.pipeline_lock:
                self.stt_mute_state = muted

            msg = Bool()
            msg.data = muted
            self.stt_mute_publisher.publish(msg)

            if muted:
                self.get_logger().info("[STT MUTE] true")
            else:
                self.get_logger().info("[STT MUTE] false")


        def publish_mission_state(self, state: str, reason: str = "") -> None:
            state = str(state).strip().upper()
            if state not in VALID_MISSION_STATES:
                self.get_logger().warn(f"Ignored invalid mission_state: {state}")
                return

            self.mission_state = state

            payload = {
                "mission_state": self.mission_state,
                "reason": reason,
                "source": "zeri_vlm_stt_bridge_node",
                "stamp_sec": time.time(),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.mission_state_publisher.publish(msg)
            self.get_logger().info(f"[MISSION STATE] {msg.data}")


        def publish_nav_intent(
            self,
            nav_intent: str,
            mission_state: Optional[str] = None,
            selected_person_id: str = "none",
            reason: str = "",
            source: str = "vlm_decision",
        ) -> None:
            nav_intent = str(nav_intent).strip().lower()
            if nav_intent not in VALID_NAV_INTENTS:
                self.get_logger().warn(f"Ignored invalid nav_intent: {nav_intent}")
                return

            state = mission_state or self.mission_state
            if state not in VALID_MISSION_STATES:
                state = self.mission_state

            payload = {
                "nav_intent": nav_intent,
                "mission_state": state,
                "selected_person_id": selected_person_id or "none",
                "reason": reason,
                "source": source,
                "target_context": self.latest_target_context,
                "stamp_sec": time.time(),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.nav_intent_publisher.publish(msg)
            self.get_logger().info(f"[NAV INTENT] {msg.data}")


        def publish_map_marker(self, decision: VLMDecision, request: VLMRequest) -> None:
            if not decision.map_mark_required:
                return

            marker_id = f"victim_{int(time.time() * 1000)}"
            target_context = self.latest_target_context if self.latest_target_context else {}

            payload = {
                "marker_id": marker_id,
                "map_mark_type": decision.map_mark_type,
                "hazard_level": decision.hazard_level,
                "scene_status": decision.scene_status,
                "selected_person_id": decision.selected_person_id,
                "report_to_base": decision.report_to_base,
                "reason": decision.reason,
                "source": "zeri_vlm_stt_bridge_node",
                "request_kind": request.request_kind,
                "target_context": target_context,
                "stamp_sec": time.time(),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.map_marker_publisher.publish(msg)
            self.get_logger().info(f"[MAP MARKER] {msg.data}")


        def publish_arm_home_request(self, decision: VLMDecision) -> None:
            if not decision.arm_home_required:
                return

            msg = Bool()
            msg.data = True
            self.arm_home_request_publisher.publish(msg)
            self.publish_mission_state(MISSION_RETURN_ARM_HOME, reason="arm_home_required")
            self.get_logger().info("[ARM HOME REQUEST] true")


        def publish_vla_task_request(self, decision: VLMDecision) -> Optional[str]:
            if not self.enable_vla:
                self.get_logger().info("VLA request skipped because enable_vla=false.")
                return None

            if not decision.vla_required:
                return None

            if decision.selected_task not in SUPPORTED_VLA_TASKS:
                self.get_logger().warn(
                    f"VLA request skipped. Unsupported task: {decision.selected_task}"
                )
                return None

            task_id = f"{decision.selected_task}_{int(time.time() * 1000)}"

            payload = {
                "task_id": task_id,
                "selected_task": decision.selected_task,
                "instruction": decision.vla_instruction,
                "task_duration_sec": decision.task_duration_sec,
                "timeout_sec": self.vla_timeout_sec,
                "mission_state": decision.mission_state,
                "selected_person_id": decision.selected_person_id,
                "target_context": self.latest_target_context,
                "source": "zeri_vlm_stt_bridge_node",
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.vla_task_request_publisher.publish(msg)

            with self.pipeline_lock:
                self.waiting_for_vla = True
                self.vla_active = True
                self.vla_deadline = time.time() + self.vla_timeout_sec
                self.active_vla_task_id = task_id

            self.publish_mission_state(MISSION_RUN_VLA, reason=f"vla_task_requested:{task_id}")
            self.publish_led(self.led_on_vla_running)
            self.publish_inference_status(f"vla_task_requested: {task_id}")

            self.get_logger().info(f"[VLA REQUEST] {msg.data}")

            return task_id
