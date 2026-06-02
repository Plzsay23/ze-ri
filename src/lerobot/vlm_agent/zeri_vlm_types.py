# zeri_vlm_types.py
from dataclasses import dataclass, field
from typing import Any, Dict

try:
    from .zeri_vlm_constants import MISSION_TRIAGE_DIALOGUE
except ImportError:
    from zeri_vlm_constants import MISSION_TRIAGE_DIALOGUE


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
