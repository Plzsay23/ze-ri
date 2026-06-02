# zeri_vlm_decision.py
import json
import re
from typing import Any, Dict, Optional

try:
    from .zeri_vlm_constants import (
        LED_BLUE,
        LED_GREEN,
        LED_OFF,
        LED_RED,
        LED_YELLOW,
        MISSION_APPROACH_PERSON,
        MISSION_MARK_AND_REPORT,
        MISSION_RESUME_SEARCH,
        MISSION_RETURN_ARM_HOME,
        MISSION_RUN_VLA,
        MISSION_STOP_AT_DISTANCE,
        MISSION_TRIAGE_DIALOGUE,
        VALID_HANDOFF_STATUSES,
        VALID_MISSION_STATES,
        VALID_NAV_INTENTS,
        VALID_TASKS,
        SUPPORTED_VLA_TASKS,
    )
    from .zeri_vlm_types import VLMDecision
except ImportError:
    from zeri_vlm_constants import (
        LED_BLUE,
        LED_GREEN,
        LED_OFF,
        LED_RED,
        LED_YELLOW,
        MISSION_APPROACH_PERSON,
        MISSION_MARK_AND_REPORT,
        MISSION_RESUME_SEARCH,
        MISSION_RETURN_ARM_HOME,
        MISSION_RUN_VLA,
        MISSION_STOP_AT_DISTANCE,
        MISSION_TRIAGE_DIALOGUE,
        VALID_HANDOFF_STATUSES,
        VALID_MISSION_STATES,
        VALID_NAV_INTENTS,
        VALID_TASKS,
        SUPPORTED_VLA_TASKS,
    )
    from zeri_vlm_types import VLMDecision


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp_led_cmd(value: Any, default: int = LED_OFF) -> int:
    led_cmd = safe_int(value, default=default)
    if led_cmd < 0 or led_cmd > 7:
        return default
    return led_cmd


def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def infer_task_from_text(raw_text: str) -> Optional[str]:
    text = raw_text.lower()

    oxygen_keywords = [
        "숨쉬기",
        "숨 쉬기",
        "숨을 못",
        "숨 못",
        "산소",
        "질식",
        "호흡",
        "가스",
        "연기",
        "oxygen",
        "breath",
        "respiratory",
        "smoke",
        "gas",
    ]

    radio_keywords = [
        "무전",
        "연락",
        "통신",
        "119",
        "구조대",
        "도와줘",
        "도움",
        "신고",
        "radio",
        "rescue",
        "communicat",
        "call",
    ]

    if any(keyword in text for keyword in oxygen_keywords):
        return "oxygen_mask_delivery"

    if any(keyword in text for keyword in radio_keywords):
        return "radio_delivery"

    return None


def resolve_led_cmd_from_fields(
    selected_task: str,
    scene_status: str,
    hazard_level: str,
    confidence: float,
) -> int:
    task = selected_task.lower().strip()
    scene = scene_status.lower().strip()
    hazard = hazard_level.lower().strip()

    if task == "oxygen_mask_delivery":
        return LED_RED

    if task == "radio_delivery":
        return LED_BLUE

    if hazard in {"critical", "danger"}:
        return LED_RED

    if hazard == "urgent":
        return LED_YELLOW

    if scene in {"respiratory_distress", "fire_nearby", "smoke_or_gas", "no_response"}:
        return LED_RED

    if scene == "needs_communication":
        return LED_BLUE

    if confidence < 0.50 or task == "status_check" or scene == "unknown":
        return LED_YELLOW

    if hazard == "normal" or scene == "normal" or task == "idle":
        return LED_GREEN

    return LED_YELLOW


def default_instruction_for_task(selected_task: str) -> str:
    if selected_task == "oxygen_mask_delivery":
        return "Deliver the oxygen mask to the person."

    if selected_task == "radio_delivery":
        return "Deliver the radio device to the person."

    return ""


def normalize_decision(
    parsed: Optional[Dict[str, Any]],
    raw_text: str,
    stt_text: str,
    default_task_duration_sec: float,
    request_mission_state: str = MISSION_TRIAGE_DIALOGUE,
) -> VLMDecision:
    def build_decision(
        *,
        mission_state: str,
        selected_person_id: str = "none",
        hazard_level: str,
        scene_status: str,
        selected_task: str,
        nav_intent: str,
        need_oxygen_mask: bool,
        confidence: float,
        led_cmd: int,
        reason: str,
        robot_speech: str,
        vla_required: bool,
        vla_instruction: str = "",
        task_duration_sec: Optional[float] = None,
        handoff_status: str = "not_applicable",
        arm_home_required: bool = False,
        map_mark_required: bool = False,
        map_mark_type: str = "none",
        report_to_base: bool = False,
    ) -> VLMDecision:
        if mission_state not in VALID_MISSION_STATES:
            mission_state = request_mission_state
            if mission_state not in VALID_MISSION_STATES:
                mission_state = MISSION_TRIAGE_DIALOGUE

        if selected_task not in VALID_TASKS:
            selected_task = "status_check"

        if nav_intent not in VALID_NAV_INTENTS:
            nav_intent = "hold_position"

        if handoff_status not in VALID_HANDOFF_STATUSES:
            handoff_status = "unknown"

        confidence = max(0.0, min(1.0, confidence))

        if task_duration_sec is None:
            task_duration_sec = default_task_duration_sec
        task_duration_sec = max(1.0, float(task_duration_sec))

        if selected_task == "oxygen_mask_delivery":
            vla_required = True
            need_oxygen_mask = True
            vla_instruction = vla_instruction or default_instruction_for_task(selected_task)
            mission_state = MISSION_RUN_VLA
            nav_intent = "hold_position"
            if hazard_level not in {"critical", "danger", "urgent"}:
                hazard_level = "critical"
            if scene_status == "unknown":
                scene_status = "respiratory_distress"

        elif selected_task == "radio_delivery":
            vla_required = True
            vla_instruction = vla_instruction or default_instruction_for_task(selected_task)
            mission_state = MISSION_RUN_VLA
            nav_intent = "hold_position"
            if hazard_level == "normal":
                hazard_level = "urgent"
            if scene_status == "unknown":
                scene_status = "needs_communication"

        else:
            vla_required = False
            vla_instruction = ""

        if mission_state in {
            MISSION_STOP_AT_DISTANCE,
            MISSION_TRIAGE_DIALOGUE,
            MISSION_RUN_VLA,
            MISSION_VERIFY_HANDOFF,
            MISSION_RETURN_ARM_HOME,
            MISSION_MARK_AND_REPORT,
        }:
            nav_intent = "hold_position" if nav_intent not in {"retreat", "go_to_safe_zone"} else nav_intent

        if mission_state == MISSION_VERIFY_HANDOFF and handoff_status == "not_applicable":
            handoff_status = "unknown"

        if handoff_status == "received":
            arm_home_required = True
            if mission_state == MISSION_VERIFY_HANDOFF:
                mission_state = MISSION_RETURN_ARM_HOME
            nav_intent = "hold_position"

        if mission_state == MISSION_RETURN_ARM_HOME:
            arm_home_required = True

        severe_scene = scene_status in {
            "no_response",
            "respiratory_distress",
            "fire_nearby",
            "smoke_or_gas",
        }
        severe_hazard = hazard_level in {"critical", "danger"}
        if severe_hazard or scene_status == "no_response":
            map_mark_required = True
            report_to_base = True
            if map_mark_type in {"", "none"}:
                map_mark_type = "critical_victim"
        elif severe_scene and hazard_level == "urgent":
            report_to_base = True

        if not map_mark_required:
            map_mark_type = "none"

        return VLMDecision(
            mission_state=mission_state,
            selected_person_id=selected_person_id,
            hazard_level=hazard_level,
            scene_status=scene_status,
            selected_task=selected_task,
            nav_intent=nav_intent,
            need_oxygen_mask=need_oxygen_mask,
            confidence=confidence,
            led_cmd=clamp_led_cmd(led_cmd, default=LED_YELLOW),
            reason=reason,
            robot_speech=robot_speech,
            vla_required=vla_required,
            vla_instruction=vla_instruction,
            task_duration_sec=task_duration_sec,
            handoff_status=handoff_status,
            arm_home_required=arm_home_required,
            map_mark_required=map_mark_required,
            map_mark_type=map_mark_type,
            report_to_base=report_to_base,
            raw_text=raw_text,
        )

    if parsed is None:
        fallback_task = infer_task_from_text(stt_text)

        if fallback_task == "oxygen_mask_delivery":
            return build_decision(
                mission_state=MISSION_RUN_VLA,
                hazard_level="critical",
                scene_status="respiratory_distress",
                selected_task="oxygen_mask_delivery",
                nav_intent="hold_position",
                need_oxygen_mask=True,
                confidence=0.50,
                led_cmd=LED_RED,
                reason="VLM JSON parsing failed, but STT text indicates respiratory distress.",
                robot_speech="호흡곤란으로 판단했습니다. 산소 마스크 전달을 준비하겠습니다.",
                vla_required=True,
                vla_instruction=default_instruction_for_task("oxygen_mask_delivery"),
                map_mark_required=True,
                map_mark_type="critical_victim",
                report_to_base=True,
            )

        if fallback_task == "radio_delivery":
            return build_decision(
                mission_state=MISSION_RUN_VLA,
                hazard_level="urgent",
                scene_status="needs_communication",
                selected_task="radio_delivery",
                nav_intent="hold_position",
                need_oxygen_mask=False,
                confidence=0.50,
                led_cmd=LED_BLUE,
                reason="VLM JSON parsing failed, but STT text indicates communication support.",
                robot_speech="구조대와의 통신이 필요하다고 판단했습니다. 무전기 전달을 준비하겠습니다.",
                vla_required=True,
                vla_instruction=default_instruction_for_task("radio_delivery"),
            )

        return build_decision(
            mission_state=request_mission_state,
            hazard_level="caution",
            scene_status="unknown",
            selected_task="status_check",
            nav_intent="hold_position",
            need_oxygen_mask=False,
            confidence=0.0,
            led_cmd=LED_YELLOW,
            reason="VLM JSON parsing failed.",
            robot_speech="현재 판단 결과를 해석하지 못했습니다. 다시 말씀해 주세요.",
            vla_required=False,
        )

    mission_state = normalize_str(
        parsed.get("mission_state"),
        request_mission_state,
    ).upper()
    if mission_state not in VALID_MISSION_STATES:
        mission_state = request_mission_state

    selected_person_id = normalize_str(
        parsed.get("selected_person_id", parsed.get("target_person_id", "none")),
        "none",
    )

    hazard_level = normalize_str(parsed.get("hazard_level"), "caution").lower()
    scene_status = normalize_str(parsed.get("scene_status"), "unknown").lower()
    selected_task = normalize_str(parsed.get("selected_task"), "idle").lower()
    nav_intent = normalize_str(parsed.get("nav_intent"), "hold_position").lower()

    if selected_task not in VALID_TASKS:
        selected_task = "status_check"

    if nav_intent not in VALID_NAV_INTENTS:
        nav_intent = "hold_position"

    confidence = safe_float(parsed.get("confidence", 0.0), default=0.0)
    confidence = max(0.0, min(1.0, confidence))

    inferred_task = infer_task_from_text(stt_text)

    need_oxygen_mask = bool(parsed.get("need_oxygen_mask", False))
    if selected_task == "oxygen_mask_delivery":
        need_oxygen_mask = True

    if inferred_task == "oxygen_mask_delivery" and selected_task not in SUPPORTED_VLA_TASKS:
        selected_task = "oxygen_mask_delivery"
        scene_status = "respiratory_distress"
        hazard_level = "critical"
        need_oxygen_mask = True

    elif inferred_task == "radio_delivery" and selected_task not in SUPPORTED_VLA_TASKS:
        selected_task = "radio_delivery"
        scene_status = "needs_communication"
        hazard_level = "urgent"

    parsed_vla_required = bool(parsed.get("vla_required", selected_task in SUPPORTED_VLA_TASKS))
    vla_required = parsed_vla_required and selected_task in SUPPORTED_VLA_TASKS

    vla_instruction = normalize_str(parsed.get("vla_instruction"), "")
    if vla_required and not vla_instruction:
        vla_instruction = default_instruction_for_task(selected_task)

    task_duration_sec = safe_float(
        parsed.get("task_duration_sec", default_task_duration_sec),
        default=default_task_duration_sec,
    )

    fallback_led = resolve_led_cmd_from_fields(
        selected_task=selected_task,
        scene_status=scene_status,
        hazard_level=hazard_level,
        confidence=confidence,
    )
    led_cmd = clamp_led_cmd(parsed.get("led_cmd", fallback_led), default=fallback_led)

    handoff_status = normalize_str(parsed.get("handoff_status"), "not_applicable").lower()
    if handoff_status not in VALID_HANDOFF_STATUSES:
        handoff_status = "unknown"

    arm_home_required = bool(parsed.get("arm_home_required", False))
    map_mark_required = bool(parsed.get("map_mark_required", False))
    map_mark_type = normalize_str(parsed.get("map_mark_type"), "none").lower()
    report_to_base = bool(parsed.get("report_to_base", False))

    reason = normalize_str(parsed.get("reason"), "No reason provided.")
    robot_speech = normalize_str(parsed.get("robot_speech"), "")

    if not robot_speech:
        if selected_task == "oxygen_mask_delivery":
            robot_speech = "호흡곤란으로 판단했습니다. 산소 마스크 전달을 준비하겠습니다."
        elif selected_task == "radio_delivery":
            robot_speech = "구조대와의 통신이 필요하다고 판단했습니다. 무전기 전달을 준비하겠습니다."
        elif selected_task == "status_check":
            robot_speech = "괜찮으십니까? 제 말이 들리시면 필요한 도움을 말씀해 주세요."
        elif selected_task == "call_rescue":
            robot_speech = "구조대의 도움이 필요한 상황으로 판단했습니다. 현재 위치를 표시하겠습니다."
        elif handoff_status == "received":
            robot_speech = "전달이 완료된 것으로 확인했습니다. 로봇팔을 원위치로 복귀하겠습니다."
        else:
            robot_speech = "현재 긴급 조치가 필요한 상황은 아닌 것으로 판단됩니다."

    return build_decision(
        mission_state=mission_state,
        selected_person_id=selected_person_id,
        hazard_level=hazard_level,
        scene_status=scene_status,
        selected_task=selected_task,
        nav_intent=nav_intent,
        need_oxygen_mask=need_oxygen_mask,
        confidence=confidence,
        led_cmd=led_cmd,
        reason=reason,
        robot_speech=robot_speech,
        vla_required=vla_required,
        vla_instruction=vla_instruction,
        task_duration_sec=task_duration_sec,
        handoff_status=handoff_status,
        arm_home_required=arm_home_required,
        map_mark_required=map_mark_required,
        map_mark_type=map_mark_type,
        report_to_base=report_to_base,
    )
