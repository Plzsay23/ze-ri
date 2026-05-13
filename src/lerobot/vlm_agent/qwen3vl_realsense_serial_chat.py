import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.

현재 시스템에는 실제 로봇 정책 모델이 아직 없다.
그러나 산소마스크 전달 정책 모델이 존재한다고 가정하고,
카메라 장면과 사용자 명령을 보고 산소마스크 전달 모델을 실행해야 하는지 판단한다.

너의 역할은 단 하나다.

1. 산소마스크 전달이 필요한 상황인지 판단한다.
2. 필요하면 adapter_id를 "oxygen_mask_delivery_lora"로 선택한다.
3. 필요하지 않으면 adapter_id를 "idle_lora"로 선택한다.
4. 반드시 JSON으로만 답한다.
5. JSON 바깥의 설명 문장은 절대 쓰지 않는다.

판단 기준:
- 사용자가 "숨쉬기 힘들다", "산소가 필요하다", "질식할 것 같다", "가스 냄새가 난다", "연기 때문에 숨을 못 쉬겠다" 등 호흡 곤란을 말하면 산소마스크 전달 필요.
- 장면에 사람이 보이고, 연기/화재/유독가스 위험이 추정되면 산소마스크 전달 필요.
- 장면만 애매하더라도 사용자 명령이 호흡 곤란이면 산소마스크 전달 필요.
- 단순 인사, 일반 설명 요청, 아무 위험 없음이면 idle.

출력 JSON schema:
{
  "need_oxygen_mask": true,
  "selected_task": "oxygen_mask_delivery|idle",
  "adapter_id": "oxygen_mask_delivery_lora|idle_lora",
  "confidence": 0.0,
  "reason": "짧은 한국어 이유"
}
"""


DEFAULT_USER_PROMPT = """
현재 카메라 장면과 사용자 말을 보고 산소마스크 전달 모델을 실행해야 하는지 판단해라.

사용자 말:
"{user_text}"

JSON으로만 답해라.
"""


@dataclass
class VLMDecision:
    need_oxygen_mask: bool
    selected_task: str
    adapter_id: str
    confidence: float
    reason: str
    raw_text: str


def start_realsense_color(serial: str, width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)

    # depth는 사용하지 않는다. color stream만 연다.
    # bgr8로 받은 뒤 RGB로 변환한다.
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)

    # auto exposure 안정화용 warmup
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline


def capture_rgb_frame(pipeline: rs.pipeline) -> Image.Image:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        raise RuntimeError("RealSense color frame capture failed.")

    bgr = np.asanyarray(color_frame.get_data())
    rgb = bgr[:, :, ::-1]  # BGR -> RGB
    image = Image.fromarray(rgb)

    return image


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    모델이 실수로 JSON 앞뒤에 텍스트를 붙였을 때도 최대한 복구한다.
    """
    text = text.strip()

    # 1. 그대로 JSON 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` 제거
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 가장 바깥 JSON 객체 추출
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    return None


def normalize_decision(parsed: Optional[Dict[str, Any]], raw_text: str) -> VLMDecision:
    """
    JSON 파싱 실패 시 안전하게 idle 처리한다.
    """
    if parsed is None:
        return VLMDecision(
            need_oxygen_mask=False,
            selected_task="idle",
            adapter_id="idle_lora",
            confidence=0.0,
            reason="VLM JSON parsing failed.",
            raw_text=raw_text,
        )

    need = bool(parsed.get("need_oxygen_mask", False))

    selected_task = str(parsed.get("selected_task", "")).strip()
    adapter_id = str(parsed.get("adapter_id", "")).strip()

    if need:
        selected_task = "oxygen_mask_delivery"
        adapter_id = "oxygen_mask_delivery_lora"
    else:
        selected_task = "idle"
        adapter_id = "idle_lora"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."

    return VLMDecision(
        need_oxygen_mask=need,
        selected_task=selected_task,
        adapter_id=adapter_id,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
    )


def run_vlm_once(
    model,
    processor,
    image: Image.Image,
    user_text: str,
    max_new_tokens: int,
) -> VLMDecision:
    prompt = DEFAULT_USER_PROMPT.format(user_text=user_text)

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    parsed = extract_json_object(output_text)
    return normalize_decision(parsed, output_text)


def execute_oxygen_mask_delivery_stub(decision: VLMDecision) -> None:
    """
    아직 실제 VLA/정책 모델이 없으므로 실행되는 것처럼 보이는 stub.
    나중에 이 함수 내부를 XVLA/GR00T policy inference 호출로 교체하면 된다.
    """
    print("\n[ACTION PIPELINE]")
    print("[1/5] Selected adapter: oxygen_mask_delivery_lora")
    time.sleep(0.2)

    print("[2/5] Loading policy: oxygen_mask_delivery_policy")
    time.sleep(0.2)

    print("[3/5] Moving mobile base toward victim ... MOCK")
    time.sleep(0.2)

    print("[4/5] Executing SO-101 arm trajectory for oxygen mask delivery ... MOCK")
    time.sleep(0.2)

    print("[5/5] Oxygen mask delivery sequence completed ... MOCK")
    print(f"[REASON] {decision.reason}")
    print("[STATUS] DONE")


def execute_idle_stub(decision: VLMDecision) -> None:
    print("\n[ACTION PIPELINE]")
    print("[1/1] No oxygen-mask delivery required.")
    print(f"[REASON] {decision.reason}")
    print("[STATUS] IDLE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="332322071907")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[INFO] Loading VLM: {args.model}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model)

    print(f"[INFO] Starting RealSense color stream only.")
    print(f"[INFO] serial={args.serial}, resolution={args.width}x{args.height}, fps={args.fps}")

    pipeline = start_realsense_color(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print("\n[READY]")
    print("사용자 말을 입력하면 현재 RealSense RGB 프레임 1장을 캡처해서 판단합니다.")
    print("예시: 숨쉬기가 힘들어")
    print("종료: q / exit / quit")

    try:
        while True:
            user_text = input("\nUSER> ").strip()

            if user_text.lower() in {"q", "exit", "quit"}:
                break

            if not user_text:
                continue

            print("[INFO] Capturing RGB frame...")
            image = capture_rgb_frame(pipeline)

            print("[INFO] Running VLM decision...")
            decision = run_vlm_once(
                model=model,
                processor=processor,
                image=image,
                user_text=user_text,
                max_new_tokens=args.max_new_tokens,
            )

            print("\n[VLM RAW OUTPUT]")
            print(decision.raw_text)

            print("\n[VLM DECISION]")
            print(json.dumps(
                {
                    "need_oxygen_mask": decision.need_oxygen_mask,
                    "selected_task": decision.selected_task,
                    "adapter_id": decision.adapter_id,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
                ensure_ascii=False,
                indent=2,
            ))

            if (
                decision.need_oxygen_mask
                and decision.adapter_id == "oxygen_mask_delivery_lora"
                and decision.confidence >= args.confidence_threshold
            ):
                execute_oxygen_mask_delivery_stub(decision)
            else:
                execute_idle_stub(decision)

    finally:
        print("\n[INFO] Stopping RealSense pipeline.")
        pipeline.stop()


if __name__ == "__main__":
    main()