#!/usr/bin/env python3
# src/lerobot/vlm_agent/vlm_stt_bridge_node.py

import copy
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32, Int32, String
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LED_OFF = 0
LED_RED = 1
LED_GREEN = 2
LED_BLUE = 3
LED_YELLOW = 4
LED_MAGENTA = 5
LED_CYAN = 6
LED_WHITE = 7

LED_NAME_MAP = {
    LED_OFF: "OFF",
    LED_RED: "RED",
    LED_GREEN: "GREEN",
    LED_BLUE: "BLUE",
    LED_YELLOW: "YELLOW",
    LED_MAGENTA: "MAGENTA",
    LED_CYAN: "CYAN",
    LED_WHITE: "WHITE",
}

SUPPORTED_VLA_TASKS = {
    "oxygen_mask_delivery",
    "radio_delivery",
}

VALID_TASKS = {
    "idle",
    "status_check",
    "oxygen_mask_delivery",
    "radio_delivery",
    "call_rescue",
}

VALID_NAV_INTENTS = {
    "stop",
    "hold_position",
    "rotate_search",
    "approach_person",
    "follow_voice",
    "retreat",
    "go_to_safe_zone",
}

MISSION_SEARCH_PERSON = "SEARCH_PERSON"
MISSION_SELECT_TARGET = "SELECT_TARGET"
MISSION_APPROACH_PERSON = "APPROACH_PERSON"
MISSION_STOP_AT_DISTANCE = "STOP_AT_DISTANCE"
MISSION_TRIAGE_DIALOGUE = "TRIAGE_DIALOGUE"
MISSION_RUN_VLA = "RUN_VLA"
MISSION_VERIFY_HANDOFF = "VERIFY_HANDOFF"
MISSION_RETURN_ARM_HOME = "RETURN_ARM_HOME"
MISSION_MARK_AND_REPORT = "MARK_AND_REPORT"
MISSION_RESUME_SEARCH = "RESUME_SEARCH"

VALID_MISSION_STATES = {
    MISSION_SEARCH_PERSON,
    MISSION_SELECT_TARGET,
    MISSION_APPROACH_PERSON,
    MISSION_STOP_AT_DISTANCE,
    MISSION_TRIAGE_DIALOGUE,
    MISSION_RUN_VLA,
    MISSION_VERIFY_HANDOFF,
    MISSION_RETURN_ARM_HOME,
    MISSION_MARK_AND_REPORT,
    MISSION_RESUME_SEARCH,
}

VALID_HANDOFF_STATUSES = {
    "not_applicable",
    "unknown",
    "waiting",
    "received",
    "not_received",
    "failed",
}


SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.

역할:
- 너는 바퀴 속도값, cmd_vel, 모터 raw command를 생성하지 않는다.
- 너는 현재 로봇이 어떤 임무 상태여야 하는지 판단한다.
- 너는 navigation controller가 수행할 고수준 nav_intent만 결정한다.
- 너는 LED 색상, TTS 문장, ACT 기반 VLA task 실행 여부를 결정한다.
- 너는 VLA가 물건을 건넨 뒤 카메라로 사람이 실제로 받았는지 확인한다.
- 너는 응급도가 높은 요구조자를 발견하면 지도 마킹과 본부 보고 필요 여부를 결정한다.

운영 구조:
- 사람 탐지, 추적, 거리 계산, 실제 바퀴 제어는 별도 노드가 수행한다.
- 너는 person_context와 mission_context를 참고해 상태와 의도를 판단한다.
- 실제 VLA 실행은 별도 executor가 수행한다.
- 로봇팔 home 복귀도 별도 노드가 수행한다. 너는 arm_home_required만 판단한다.

현재 사용 가능한 ACT VLA task는 정확히 두 개뿐이다:
1. oxygen_mask_delivery
   - 산소마스크 전달
2. radio_delivery
   - 무전기 전달

이 두 task 외에는 VLA를 실행하면 안 된다.
그 외 상황은 idle, status_check, call_rescue 중 하나로 판단한다.

소방 초동 대응 대화 원칙:
- 먼저 의식과 반응을 확인한다: "제 말 들리십니까? 괜찮으십니까?"
- 호흡곤란, 연기/가스 흡입, 질식 호소는 우선 산소마스크 전달로 판단한다.
- 구조대와 연락, 통신 요청, 신고 요청은 무전기 전달 또는 구조대 호출로 판단한다.
- 무반응, 이동 불가, 중증 외상, 심한 출혈, 호흡 이상, 연기/화재 근접은 critical 또는 danger로 판단한다.
- 직접 처치할 수 없거나 처치가 불확실하면 call_rescue를 선택하고 지도 마킹을 권장한다.
- 판단이 불확실하면 status_check를 선택하고 짧은 추가 질문을 한다.

mission_state 운용 규칙:
- SEARCH_PERSON: 사람 탐색 상태. nav_intent는 rotate_search 또는 hold_position.
- SELECT_TARGET: 여러 사람 중 접근 대상을 고르는 상태. selected_person_id를 지정한다.
- APPROACH_PERSON: 선택된 사람에게 접근하는 상태. nav_intent는 approach_person.
- STOP_AT_DISTANCE: 일정 거리 이내에 들어와 정지하는 상태. nav_intent는 stop 또는 hold_position.
- TRIAGE_DIALOGUE: 요구조자 앞에서 문진/판단하는 상태. nav_intent는 hold_position.
- RUN_VLA: ACT VLA task 실행 상태. nav_intent는 hold_position.
- VERIFY_HANDOFF: VLA 완료 후 사람이 물건을 받았는지 확인하는 상태.
- RETURN_ARM_HOME: 물건 전달 완료 후 로봇팔 home 복귀가 필요한 상태.
- MARK_AND_REPORT: 중증자 위치를 지도에 마킹하고 본부 보고가 필요한 상태.
- RESUME_SEARCH: 처치 루프 종료 후 다시 탐색으로 복귀하는 상태.

LED command rule:
0 = OFF: 대기 또는 종료
1 = RED: 즉시 위험, 호흡곤란, 출혈, 화재, 유독가스, 미반응자
2 = GREEN: 안전, 정상, 비응급
3 = BLUE: 구조대 통신, 무전기 전달, 외부 도움 필요
4 = YELLOW: 주의, 관찰 필요, 판단 불확실
5 = MAGENTA: VLA/로봇 작업 실행 중
6 = CYAN: VLM 판단 중 또는 센싱 중
7 = WHITE: 사용자 응답 대기 또는 판단 완료

출력 JSON schema:
{
  "mission_state": "SEARCH_PERSON|SELECT_TARGET|APPROACH_PERSON|STOP_AT_DISTANCE|TRIAGE_DIALOGUE|RUN_VLA|VERIFY_HANDOFF|RETURN_ARM_HOME|MARK_AND_REPORT|RESUME_SEARCH",
  "selected_person_id": "문자열 또는 none",
  "hazard_level": "normal|caution|urgent|critical|danger",
  "scene_status": "normal|respiratory_distress|needs_communication|no_response|fire_nearby|smoke_or_gas|blocked_path|unknown",
  "selected_task": "idle|status_check|oxygen_mask_delivery|radio_delivery|call_rescue",
  "nav_intent": "stop|hold_position|rotate_search|approach_person|follow_voice|retreat|go_to_safe_zone",
  "vla_required": true,
  "vla_instruction": "Deliver the oxygen mask to the person.",
  "task_duration_sec": 20.0,
  "handoff_status": "not_applicable|unknown|waiting|received|not_received|failed",
  "arm_home_required": false,
  "map_mark_required": false,
  "map_mark_type": "none|victim|critical_victim|hazard|blocked_path",
  "report_to_base": false,
  "led_cmd": 1,
  "confidence": 0.0,
  "reason": "짧은 한국어 이유",
  "robot_speech": "한국어 한두 문장"
}

규칙:
- selected_task가 oxygen_mask_delivery이면:
  - vla_required = true
  - vla_instruction = "Deliver the oxygen mask to the person."
  - mission_state = RUN_VLA
  - led_cmd = 1 또는 5
- selected_task가 radio_delivery이면:
  - vla_required = true
  - vla_instruction = "Deliver the radio device to the person."
  - mission_state = RUN_VLA
  - led_cmd = 3 또는 5
- selected_task가 idle/status_check/call_rescue이면:
  - vla_required = false
- VERIFY_HANDOFF 상태에서는 사람이 물건을 받았다고 보이면 handoff_status="received", arm_home_required=true로 출력한다.
- handoff_status="received"이면 mission_state는 RETURN_ARM_HOME 또는 RESUME_SEARCH로 둔다.
- 무반응자, 중증 호흡곤란, 심한 출혈, 이동 불가자는 map_mark_required=true, map_mark_type="critical_victim", report_to_base=true로 둔다.
- nav_intent는 고수준 의도만 출력한다. 속도값이나 cmd_vel은 절대 출력하지 않는다.
- 반드시 JSON으로만 답한다.
- JSON 바깥 문장은 절대 쓰지 않는다.
"""


USER_PROMPT_TEMPLATE = """
현재 카메라 장면, STT 텍스트, mission_context를 보고 Ze-Ri의 다음 임무 상태를 판단해라.

이번 단계에서는 상태기계, LED, TTS, ACT 기반 VLA task, handoff 확인, 지도 마킹까지 연동한다.
단, 실제 실행 가능한 VLA task는 oxygen_mask_delivery, radio_delivery 두 개뿐이다.

요청 종류:
{request_kind}

현재 mission_context JSON:
{mission_context_json}

STT 텍스트:
"{stt_text}"

JSON으로만 답해라.
"""


@dataclass
class VLMDecision:
    mission_state: str
    selected_person_id: str
    hazard_level: str
    scene_status: str
    selected_task: str
    nav_intent: str
    need_oxygen_mask: bool
    confidence: float
    led_cmd: int
    reason: str
    robot_speech: str
    vla_required: bool
    vla_instruction: str
    task_duration_sec: float
    handoff_status: str
    arm_home_required: bool
    map_mark_required: bool
    map_mark_type: str
    report_to_base: bool
    raw_text: str


@dataclass
class VLMRequest:
    stt_text: str
    request_kind: str = "stt_triage"
    mission_state: str = MISSION_TRIAGE_DIALOGUE
    extra_context: Dict[str, Any] = field(default_factory=dict)


def make_sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


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


def ros_image_to_bgr(msg: RosImage) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("bgr8", "rgb8"):
            channels = 3
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            return arr.copy()

        if encoding in ("bgra8", "rgba8"):
            channels = 4
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgba8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr.copy()

        if encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            arr = arr.reshape((height, row_pixels))[:, :width]
            return cv2.cvtColor(arr.copy(), cv2.COLOR_GRAY2BGR)

    except Exception:
        return None

    return None


def ros_image_to_pil_rgb(msg: RosImage) -> Optional[PILImage.Image]:
    bgr = ros_image_to_bgr(msg)

    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def pil_rgb_to_ros_image(
    image: PILImage.Image,
    stamp,
    frame_id: str = "zeri_vlm_input_rgb",
) -> RosImage:
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.asarray(image, dtype=np.uint8)
    arr = np.ascontiguousarray(arr)

    msg = RosImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()

    return msg


def clone_depth_snapshot_msg(
    msg: RosImage,
    stamp,
    frame_id: str = "zeri_vlm_input_depth",
) -> RosImage:
    cloned = copy.deepcopy(msg)
    cloned.header.stamp = stamp
    cloned.header.frame_id = frame_id
    return cloned


class QwenVLMRunner:
    def __init__(
        self,
        model_id: str,
        dtype_name: str,
        max_new_tokens: int,
        default_task_duration_sec: float,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.default_task_duration_sec = default_task_duration_sec

        if dtype_name == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(
        self,
        image: PILImage.Image,
        stt_text: str,
        request_kind: str,
        mission_context: Dict[str, Any],
        request_mission_state: str,
    ) -> VLMDecision:
        mission_context_json = json.dumps(
            mission_context,
            ensure_ascii=False,
            indent=2,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            request_kind=request_kind,
            mission_context_json=mission_context_json,
            stt_text=stt_text,
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = extract_json_object(output_text)

        return normalize_decision(
            parsed=parsed,
            raw_text=output_text,
            stt_text=stt_text,
            default_task_duration_sec=self.default_task_duration_sec,
            request_mission_state=request_mission_state,
        )

class ZeriVLMSTTBridgeNode(Node):
    def __init__(self):
        super().__init__("zeri_vlm_stt_bridge_node")

        self.declare_parameter("rgb_topic", "/zeri/top/rgb/image_raw")
        self.declare_parameter("depth_topic", "/zeri/top/depth/image_raw")
        self.declare_parameter("stt_topic", "/stt/text")

        self.declare_parameter("decision_topic", "/zeri/vlm/decision")
        self.declare_parameter("robot_speech_topic", "/zeri/vlm/robot_speech")
        self.declare_parameter("vlm_input_rgb_topic", "/zeri/vlm/input_rgb")
        self.declare_parameter("vlm_input_depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("inference_status_topic", "/zeri/vlm/inference_status")

        self.declare_parameter("mission_event_topic", "/zeri/mission/event")
        self.declare_parameter("mission_state_topic", "/zeri/mission/state")
        self.declare_parameter("nav_intent_topic", "/zeri/nav/intent")
        self.declare_parameter("target_context_topic", "/zeri/person/target")
        self.declare_parameter("map_marker_topic", "/zeri/map/person_marker")
        self.declare_parameter("arm_home_request_topic", "/zeri/arm/home_request")
        self.declare_parameter("enable_mission_events", True)
        self.declare_parameter("verify_handoff_after_vla_success", True)
        self.declare_parameter("initial_mission_state", MISSION_SEARCH_PERSON)

        self.declare_parameter("tts_status_topic", "/zeri/tts/status")
        self.declare_parameter("stt_mute_topic", "/zeri/stt/mute")

        self.declare_parameter("led_topic", "/zeri/led/cmd")
        self.declare_parameter("led_on_startup", LED_WHITE)
        self.declare_parameter("led_on_loading", LED_CYAN)
        self.declare_parameter("led_on_inference", LED_CYAN)
        self.declare_parameter("led_on_vla_running", LED_MAGENTA)
        self.declare_parameter("led_on_vla_success", LED_WHITE)
        self.declare_parameter("led_on_error", LED_YELLOW)
        self.declare_parameter("led_on_shutdown", LED_OFF)

        self.declare_parameter("vla_task_request_topic", "/zeri/vla/task_request")
        self.declare_parameter("vla_status_topic", "/zeri/vla/status")
        self.declare_parameter("enable_vla", True)
        self.declare_parameter("vla_timeout_sec", 60.0)
        self.declare_parameter("vla_default_task_duration_sec", 20.0)

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("use_vad_gate", False)
        self.declare_parameter("vad_hold_sec", 1.2)

        self.declare_parameter("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("max_new_tokens", 192)
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("queue_size", 4)

        self.declare_parameter("stt_gate_mode", "wake")
        self.declare_parameter("stt_min_chars", 3)
        self.declare_parameter("wake_listen_window_sec", 8.0)
        self.declare_parameter("min_inference_interval_sec", 2.0)
        self.declare_parameter("duplicate_window_sec", 5.0)
        self.declare_parameter("wake_words", ["제리", "제리야"])

        self.declare_parameter("stt_block_after_tts_sec", 0.8)
        self.declare_parameter("tts_max_wait_sec", 20.0)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.stt_topic = str(self.get_parameter("stt_topic").value)

        self.decision_topic = str(self.get_parameter("decision_topic").value)
        self.robot_speech_topic = str(self.get_parameter("robot_speech_topic").value)
        self.vlm_input_rgb_topic = str(self.get_parameter("vlm_input_rgb_topic").value)
        self.vlm_input_depth_topic = str(self.get_parameter("vlm_input_depth_topic").value)
        self.inference_status_topic = str(self.get_parameter("inference_status_topic").value)

        self.mission_event_topic = str(self.get_parameter("mission_event_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.nav_intent_topic = str(self.get_parameter("nav_intent_topic").value)
        self.target_context_topic = str(self.get_parameter("target_context_topic").value)
        self.map_marker_topic = str(self.get_parameter("map_marker_topic").value)
        self.arm_home_request_topic = str(
            self.get_parameter("arm_home_request_topic").value
        )
        self.enable_mission_events = bool(
            self.get_parameter("enable_mission_events").value
        )
        self.verify_handoff_after_vla_success = bool(
            self.get_parameter("verify_handoff_after_vla_success").value
        )
        initial_mission_state = str(
            self.get_parameter("initial_mission_state").value
        ).upper()
        if initial_mission_state not in VALID_MISSION_STATES:
            initial_mission_state = MISSION_SEARCH_PERSON
        self.mission_state = initial_mission_state

        self.tts_status_topic = str(self.get_parameter("tts_status_topic").value)
        self.stt_mute_topic = str(self.get_parameter("stt_mute_topic").value)

        self.led_topic = str(self.get_parameter("led_topic").value)
        self.led_on_startup = clamp_led_cmd(
            self.get_parameter("led_on_startup").value,
            LED_WHITE,
        )
        self.led_on_loading = clamp_led_cmd(
            self.get_parameter("led_on_loading").value,
            LED_CYAN,
        )
        self.led_on_inference = clamp_led_cmd(
            self.get_parameter("led_on_inference").value,
            LED_CYAN,
        )
        self.led_on_vla_running = clamp_led_cmd(
            self.get_parameter("led_on_vla_running").value,
            LED_MAGENTA,
        )
        self.led_on_vla_success = clamp_led_cmd(
            self.get_parameter("led_on_vla_success").value,
            LED_WHITE,
        )
        self.led_on_error = clamp_led_cmd(
            self.get_parameter("led_on_error").value,
            LED_YELLOW,
        )
        self.led_on_shutdown = clamp_led_cmd(
            self.get_parameter("led_on_shutdown").value,
            LED_OFF,
        )

        self.vla_task_request_topic = str(
            self.get_parameter("vla_task_request_topic").value
        )
        self.vla_status_topic = str(self.get_parameter("vla_status_topic").value)
        self.enable_vla = bool(self.get_parameter("enable_vla").value)
        self.vla_timeout_sec = float(self.get_parameter("vla_timeout_sec").value)
        self.vla_default_task_duration_sec = float(
            self.get_parameter("vla_default_task_duration_sec").value
        )

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.use_vad_gate = bool(self.get_parameter("use_vad_gate").value)
        self.vad_hold_sec = float(self.get_parameter("vad_hold_sec").value)

        model_id = str(self.get_parameter("model_id").value)
        dtype = str(self.get_parameter("dtype").value)
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        queue_size = int(self.get_parameter("queue_size").value)
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )

        self.stt_gate_mode = str(self.get_parameter("stt_gate_mode").value)
        self.stt_min_chars = int(self.get_parameter("stt_min_chars").value)
        self.wake_listen_window_sec = float(
            self.get_parameter("wake_listen_window_sec").value
        )
        self.min_inference_interval_sec = float(
            self.get_parameter("min_inference_interval_sec").value
        )
        self.duplicate_window_sec = float(
            self.get_parameter("duplicate_window_sec").value
        )

        self.stt_block_after_tts_sec = float(
            self.get_parameter("stt_block_after_tts_sec").value
        )
        self.tts_max_wait_sec = float(self.get_parameter("tts_max_wait_sec").value)

        wake_words_param = self.get_parameter("wake_words").value
        self.wake_words = [
            str(word).strip()
            for word in wake_words_param
            if str(word).strip()
        ]

        self.ignore_phrases = {
            "어",
            "음",
            "아",
            "네",
            "예",
            "응",
            "테스트",
            "마이크 테스트",
            "안녕하세요",
        }

        self.latest_target_context: Dict[str, Any] = {}
        self.latest_target_context_time = 0.0
        self.latest_mission_event: Dict[str, Any] = {}
        self.latest_mission_event_time = 0.0
        self.last_decision_context: Dict[str, Any] = {}

        self.last_accepted_text = ""
        self.last_accepted_time = 0.0
        self.last_inference_request_time = 0.0
        self.wake_active_until = 0.0

        self.pipeline_lock = threading.Lock()
        self.pipeline_busy = False

        self.waiting_for_tts = False
        self.tts_active = False
        self.stt_block_until = 0.0
        self.tts_deadline = 0.0
        self.stt_mute_state = False

        self.waiting_for_vla = False
        self.vla_active = False
        self.vla_deadline = 0.0
        self.active_vla_task_id: Optional[str] = None

        self.latest_vad = False
        self.latest_vad_time = 0.0
        self.latest_doa_deg: Optional[float] = None
        self.latest_doa_time = 0.0

        self.text_queue: queue.Queue[VLMRequest] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        self.frame_lock = threading.Lock()
        self.latest_rgb_msg: Optional[RosImage] = None
        self.latest_depth_msg: Optional[RosImage] = None
        self.latest_rgb_time: Optional[float] = None
        self.latest_depth_time: Optional[float] = None

        sensor_qos = make_sensor_qos(depth=5)
        reliable_qos = make_reliable_qos(depth=10)

        self.rgb_sub = self.create_subscription(
            RosImage,
            self.rgb_topic,
            self.rgb_callback,
            sensor_qos,
        )

        self.depth_sub = self.create_subscription(
            RosImage,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )

        self.stt_sub = self.create_subscription(
            String,
            self.stt_topic,
            self.stt_callback,
            reliable_qos,
        )

        self.mission_event_sub = self.create_subscription(
            String,
            self.mission_event_topic,
            self.mission_event_callback,
            reliable_qos,
        )

        self.target_context_sub = self.create_subscription(
            String,
            self.target_context_topic,
            self.target_context_callback,
            reliable_qos,
        )

        self.tts_status_sub = self.create_subscription(
            String,
            self.tts_status_topic,
            self.tts_status_callback,
            reliable_qos,
        )

        self.vla_status_sub = self.create_subscription(
            String,
            self.vla_status_topic,
            self.vla_status_callback,
            reliable_qos,
        )

        self.vad_sub = self.create_subscription(
            Bool,
            self.vad_topic,
            self.vad_callback,
            reliable_qos,
        )

        self.doa_sub = self.create_subscription(
            Float32,
            self.doa_topic,
            self.doa_callback,
            reliable_qos,
        )

        self.decision_publisher = self.create_publisher(
            String,
            self.decision_topic,
            reliable_qos,
        )

        self.robot_speech_publisher = self.create_publisher(
            String,
            self.robot_speech_topic,
            reliable_qos,
        )

        self.led_publisher = self.create_publisher(
            Int32,
            self.led_topic,
            reliable_qos,
        )

        self.vla_task_request_publisher = self.create_publisher(
            String,
            self.vla_task_request_topic,
            reliable_qos,
        )

        self.vlm_input_rgb_publisher = self.create_publisher(
            RosImage,
            self.vlm_input_rgb_topic,
            reliable_qos,
        )

        self.vlm_input_depth_publisher = self.create_publisher(
            RosImage,
            self.vlm_input_depth_topic,
            reliable_qos,
        )

        self.inference_status_publisher = self.create_publisher(
            String,
            self.inference_status_topic,
            reliable_qos,
        )

        self.stt_mute_publisher = self.create_publisher(
            Bool,
            self.stt_mute_topic,
            reliable_qos,
        )

        self.mission_state_publisher = self.create_publisher(
            String,
            self.mission_state_topic,
            reliable_qos,
        )

        self.nav_intent_publisher = self.create_publisher(
            String,
            self.nav_intent_topic,
            reliable_qos,
        )

        self.map_marker_publisher = self.create_publisher(
            String,
            self.map_marker_topic,
            reliable_qos,
        )

        self.arm_home_request_publisher = self.create_publisher(
            Bool,
            self.arm_home_request_topic,
            reliable_qos,
        )

        self.pipeline_timer = self.create_timer(
            0.2,
            self.pipeline_timer_callback,
        )

        self.get_logger().info("Zeri VLM-STT bridge node subscriptions:")
        self.get_logger().info(f"  RGB input:         {self.rgb_topic}")
        self.get_logger().info(f"  Depth input:       {self.depth_topic}")
        self.get_logger().info(f"  STT input:         {self.stt_topic}")
        self.get_logger().info(f"  TTS status:        {self.tts_status_topic}")
        self.get_logger().info(f"  VLA status:        {self.vla_status_topic}")
        self.get_logger().info(f"  VAD input:         {self.vad_topic}")
        self.get_logger().info(f"  DOA input:         {self.doa_topic}")
        self.get_logger().info(f"  Mission event:     {self.mission_event_topic}")
        self.get_logger().info(f"  Target context:    {self.target_context_topic}")

        self.get_logger().info("Zeri VLM-STT bridge node publishers:")
        self.get_logger().info(f"  Decision:          {self.decision_topic}")
        self.get_logger().info(f"  Robot speech:      {self.robot_speech_topic}")
        self.get_logger().info(f"  LED command:       {self.led_topic}")
        self.get_logger().info(f"  VLA task request:  {self.vla_task_request_topic}")
        self.get_logger().info(f"  VLM RGB snap:      {self.vlm_input_rgb_topic}")
        self.get_logger().info(f"  VLM Depth snap:    {self.vlm_input_depth_topic}")
        self.get_logger().info(f"  VLM status:        {self.inference_status_topic}")
        self.get_logger().info(f"  STT mute:          {self.stt_mute_topic}")
        self.get_logger().info(f"  Mission state:     {self.mission_state_topic}")
        self.get_logger().info(f"  Nav intent:        {self.nav_intent_topic}")
        self.get_logger().info(f"  Map marker:        {self.map_marker_topic}")
        self.get_logger().info(f"  Arm home request:  {self.arm_home_request_topic}")

        self.get_logger().info("Runtime settings:")
        self.get_logger().info(f"  enable_vla: {self.enable_vla}")
        self.get_logger().info(f"  vla_timeout_sec: {self.vla_timeout_sec}")
        self.get_logger().info(
            f"  vla_default_task_duration_sec: {self.vla_default_task_duration_sec}"
        )
        self.get_logger().info(f"  stt_gate_mode: {self.stt_gate_mode}")
        self.get_logger().info(f"  wake_words: {self.wake_words}")
        self.get_logger().info(
            f"  wake_listen_window_sec: {self.wake_listen_window_sec}"
        )
        self.get_logger().info(f"  use_vad_gate: {self.use_vad_gate}")
        self.get_logger().info(f"  vad_hold_sec: {self.vad_hold_sec}")
        self.get_logger().info(
            f"  stt_block_after_tts_sec: {self.stt_block_after_tts_sec}"
        )
        self.get_logger().info(f"  tts_max_wait_sec: {self.tts_max_wait_sec}")
        self.get_logger().info(f"  initial_mission_state: {self.mission_state}")
        self.get_logger().info(
            f"  verify_handoff_after_vla_success: "
            f"{self.verify_handoff_after_vla_success}"
        )

        self.publish_stt_mute(False)
        self.publish_mission_state(self.mission_state, reason="startup")

        self.publish_led(self.led_on_loading)
        self.publish_inference_status("loading_vlm_model")
        self.get_logger().info(f"Loading VLM model: {model_id}")

        self.vlm = QwenVLMRunner(
            model_id=model_id,
            dtype_name=dtype,
            max_new_tokens=max_new_tokens,
            default_task_duration_sec=self.vla_default_task_duration_sec,
        )

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

        self.publish_led(self.led_on_startup)
        self.publish_inference_status("waiting_for_camera_frame")
        self.get_logger().info("Zeri VLM-STT bridge node is ready.")

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

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping VLM-STT bridge node.")

        try:
            self.publish_inference_status("shutting_down")
            self.publish_stt_mute(False)
            self.publish_led(self.led_on_shutdown)
        except Exception:
            pass

        self.stop_event.set()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)

        super().destroy_node()


def main() -> None:
    rclpy.init()

    node: Optional[ZeriVLMSTTBridgeNode] = None

    try:
        node = ZeriVLMSTTBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()