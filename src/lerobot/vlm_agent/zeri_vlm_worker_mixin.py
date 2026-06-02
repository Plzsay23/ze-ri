# zeri_vlm_worker_mixin.py
import json
import queue
import time
from typing import Optional

from std_msgs.msg import String

try:
    from .zeri_vlm_constants import (
        LED_NAME_MAP, MISSION_MARK_AND_REPORT, MISSION_RETURN_ARM_HOME,
        MISSION_RUN_VLA, MISSION_TRIAGE_DIALOGUE, MISSION_VERIFY_HANDOFF
    )
    from .zeri_vlm_ros_image import (
        clone_depth_snapshot_msg, pil_rgb_to_ros_image, ros_image_to_pil_rgb
    )
    from .zeri_vlm_types import VLMDecision, VLMRequest
except ImportError:
    from zeri_vlm_constants import (
        LED_NAME_MAP, MISSION_MARK_AND_REPORT, MISSION_RETURN_ARM_HOME,
        MISSION_RUN_VLA, MISSION_TRIAGE_DIALOGUE, MISSION_VERIFY_HANDOFF
    )
    from zeri_vlm_ros_image import (
        clone_depth_snapshot_msg, pil_rgb_to_ros_image, ros_image_to_pil_rgb
    )
    from zeri_vlm_types import VLMDecision, VLMRequest


class ZeriVLMWorkerMixin:
        def worker_loop(self) -> None:
            while not self.stop_event.is_set():
                try:
                    request = self.text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                stt_text = request.stt_text
                mission_context = self.build_mission_context(request)
                start_time = time.time()

                try:
                    self.publish_led(self.led_on_inference)
                    self.publish_inference_status("getting_latest_camera_frame")

                    rgb_msg, depth_msg, rgb_time, depth_time = self.get_latest_frames()

                    if rgb_msg is None:
                        self.get_logger().warn("No RGB frame received yet.")
                        self.publish_inference_status("waiting_for_rgb_frame")
                        self.publish_led(self.led_on_error)
                        self.release_pipeline_block("no_rgb_frame")
                        continue

                    image = ros_image_to_pil_rgb(rgb_msg)

                    if image is None:
                        self.get_logger().error("Failed to convert RGB ROS image to PIL.")
                        self.publish_inference_status("rgb_conversion_error")
                        self.publish_led(self.led_on_error)
                        self.release_pipeline_block("rgb_conversion_error")
                        continue

                    stamp = self.get_clock().now().to_msg()

                    vlm_rgb_msg = pil_rgb_to_ros_image(
                        image=image,
                        stamp=stamp,
                        frame_id="zeri_vlm_input_rgb",
                    )

                    self.vlm_input_rgb_publisher.publish(vlm_rgb_msg)
                    self.get_logger().info("Published VLM input RGB snapshot.")

                    if depth_msg is not None:
                        vlm_depth_msg = clone_depth_snapshot_msg(
                            msg=depth_msg,
                            stamp=stamp,
                            frame_id="zeri_vlm_input_depth",
                        )
                        self.vlm_input_depth_publisher.publish(vlm_depth_msg)
                        self.get_logger().info("Published VLM input Depth snapshot.")
                    else:
                        self.get_logger().warn(
                            "No depth frame received yet. Continuing with RGB only."
                        )

                    self.publish_inference_status("running_vlm_inference")
                    self.get_logger().info("Running VLM inference...")

                    decision = self.vlm.infer(
                        image=image,
                        stt_text=stt_text,
                        request_kind=request.request_kind,
                        mission_context=mission_context,
                        request_mission_state=request.mission_state,
                    )

                    elapsed = time.time() - start_time

                    self.publish_inference_status("vlm_inference_done")
                    self.publish_led(decision.led_cmd)
                    self.publish_mission_state(decision.mission_state, reason="vlm_decision")
                    self.publish_nav_intent(
                        decision.nav_intent,
                        mission_state=decision.mission_state,
                        selected_person_id=decision.selected_person_id,
                        reason=decision.reason,
                        source="vlm_decision",
                    )
                    self.publish_map_marker(decision, request)
                    self.publish_arm_home_request(decision)

                    camera_age_sec = None
                    if rgb_time is not None:
                        camera_age_sec = round(time.time() - rgb_time, 3)

                    depth_age_sec = None
                    if depth_time is not None:
                        depth_age_sec = round(time.time() - depth_time, 3)

                    doa_age_sec = None
                    if self.latest_doa_time > 0.0:
                        doa_age_sec = round(time.time() - self.latest_doa_time, 3)

                    vla_task_id = None
                    speech_text = decision.robot_speech.strip()

                    if speech_text:
                        self.mark_waiting_for_tts()

                        speech_msg = String()
                        speech_msg.data = speech_text
                        self.robot_speech_publisher.publish(speech_msg)

                        self.get_logger().info(f"Published robot speech: {speech_text}")
                        self.get_logger().info(f"[ROBOT SPEECH] {speech_text}")

                    if decision.vla_required:
                        vla_task_id = self.publish_vla_task_request(decision)

                    result = {
                        "stt_text": stt_text,
                        "request_kind": request.request_kind,
                        "mission_state": decision.mission_state,
                        "selected_person_id": decision.selected_person_id,
                        "hazard_level": decision.hazard_level,
                        "scene_status": decision.scene_status,
                        "selected_task": decision.selected_task,
                        "nav_intent": decision.nav_intent,
                        "need_oxygen_mask": decision.need_oxygen_mask,
                        "confidence": decision.confidence,
                        "led_cmd": decision.led_cmd,
                        "led_name": LED_NAME_MAP.get(decision.led_cmd, "UNKNOWN"),
                        "reason": decision.reason,
                        "robot_speech": decision.robot_speech,
                        "vla_required": decision.vla_required,
                        "vla_instruction": decision.vla_instruction,
                        "vla_task_id": vla_task_id,
                        "vla_task_request_topic": self.vla_task_request_topic,
                        "vla_status_topic": self.vla_status_topic,
                        "mission_state_topic": self.mission_state_topic,
                        "nav_intent_topic": self.nav_intent_topic,
                        "map_marker_topic": self.map_marker_topic,
                        "arm_home_request_topic": self.arm_home_request_topic,
                        "target_context": self.latest_target_context,
                        "mission_context": mission_context,
                        "robot_control_mode": "VLM_SUPERVISED_STATE_MACHINE",
                        "raw_cmd_vel_generated_by_vlm": False,
                        "latency_sec": round(elapsed, 3),
                        "camera_age_sec": camera_age_sec,
                        "depth_age_sec": depth_age_sec,
                        "doa_deg": self.latest_doa_deg,
                        "doa_age_sec": doa_age_sec,
                        "latest_vad": self.latest_vad,
                        "use_vad_gate": self.use_vad_gate,
                        "raw_vlm_output": decision.raw_text,
                        "live_rgb_topic": self.rgb_topic,
                        "live_depth_topic": self.depth_topic,
                        "vlm_input_rgb_topic": self.vlm_input_rgb_topic,
                        "vlm_input_depth_topic": self.vlm_input_depth_topic,
                        "led_topic": self.led_topic,
                        "stt_mute_topic": self.stt_mute_topic,
                        "tts_status_topic": self.tts_status_topic,
                    }

                    decision_msg = String()
                    decision_msg.data = json.dumps(result, ensure_ascii=False)
                    self.decision_publisher.publish(decision_msg)

                    self.get_logger().info(f"Published VLM decision: {decision_msg.data}")
                    self.log_decision(decision, vla_task_id)

                    if not speech_text and not vla_task_id:
                        self.release_pipeline_block("no_tts_no_vla")

                except Exception as exc:
                    err = f"error: {exc}"
                    self.get_logger().error(f"VLM worker error: {exc}")
                    self.publish_inference_status(err)
                    self.publish_led(self.led_on_error)
                    self.release_pipeline_block("vlm_worker_error")


        def log_decision(self, decision: VLMDecision, vla_task_id: Optional[str]) -> None:
            self.get_logger().info("[VLM DECISION]")
            self.get_logger().info(f"  mission_state: {decision.mission_state}")
            self.get_logger().info(f"  selected_person_id: {decision.selected_person_id}")
            self.get_logger().info(f"  hazard_level: {decision.hazard_level}")
            self.get_logger().info(f"  scene_status: {decision.scene_status}")
            self.get_logger().info(f"  selected_task: {decision.selected_task}")
            self.get_logger().info(f"  nav_intent: {decision.nav_intent}")
            self.get_logger().info(f"  confidence: {decision.confidence}")
            self.get_logger().info(
                f"  LED: {decision.led_cmd} "
                f"({LED_NAME_MAP.get(decision.led_cmd, 'UNKNOWN')})"
            )
            self.get_logger().info(f"  vla_required: {decision.vla_required}")
            self.get_logger().info(f"  vla_instruction: {decision.vla_instruction}")
            self.get_logger().info(f"  vla_task_id: {vla_task_id}")
            self.get_logger().info(f"  handoff_status: {decision.handoff_status}")
            self.get_logger().info(f"  arm_home_required: {decision.arm_home_required}")
            self.get_logger().info(f"  map_mark_required: {decision.map_mark_required}")
            self.get_logger().info(f"  map_mark_type: {decision.map_mark_type}")
            self.get_logger().info(f"  report_to_base: {decision.report_to_base}")
            self.get_logger().info(f"  reason: {decision.reason}")
            self.get_logger().info(f"  robot_speech: {decision.robot_speech}")
