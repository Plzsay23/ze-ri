# qwen3vl_realsense_serial_chat.py

import argparse
import time
from typing import Tuple

import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.
카메라 RGB 장면과 RealSense depth 정보를 함께 보고 현재 상황을 판단한다.
사람, 화재, 연기, 장애물, 구조 도구, 접근 가능성을 판단하고 로봇이 해야 할 행동을 제안한다.
답변은 간결하고 실행 가능하게 작성한다.
"""


def start_realsense(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    # auto exposure 안정화용 warmup
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, depth_scale


def capture_aligned_rgb_depth(
    pipeline,
    align,
) -> Tuple[Image.Image, np.ndarray]:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError("Failed to capture aligned color/depth frames.")

    color_np = np.asanyarray(color_frame.get_data())  # RGB
    depth_np = np.asanyarray(depth_frame.get_data())  # uint16 depth raw

    image = Image.fromarray(color_np)

    return image, depth_np


def summarize_depth(depth_np: np.ndarray, depth_scale: float) -> str:
    """
    depth_np는 uint16 raw depth.
    meter = raw * depth_scale
    0은 invalid depth라 제외.
    """
    h, w = depth_np.shape

    depth_m = depth_np.astype(np.float32) * depth_scale
    valid = depth_m[depth_m > 0]

    if valid.size == 0:
        return "Depth information is unavailable."

    center_x = w // 2
    center_y = h // 2

    # 중앙 80x80 ROI
    roi_size = 80
    x1 = max(0, center_x - roi_size // 2)
    x2 = min(w, center_x + roi_size // 2)
    y1 = max(0, center_y - roi_size // 2)
    y2 = min(h, center_y + roi_size // 2)

    center_roi = depth_m[y1:y2, x1:x2]
    center_valid = center_roi[center_roi > 0]

    center_median = float(np.median(center_valid)) if center_valid.size > 0 else None
    nearest = float(np.percentile(valid, 1))
    median = float(np.median(valid))

    if center_median is None:
        center_text = "center_distance_m: unknown"
    else:
        center_text = f"center_distance_m: {center_median:.2f}"

    return (
        f"Depth summary:\n"
        f"- image_size: {w}x{h}\n"
        f"- {center_text}\n"
        f"- nearest_visible_depth_m_approx: {nearest:.2f}\n"
        f"- median_visible_depth_m: {median:.2f}\n"
    )


def run_qwen_vl(
    model,
    processor,
    image: Image.Image,
    user_prompt: str,
    depth_text: str,
    max_new_tokens: int,
) -> str:
    final_prompt = f"""
사용자 명령:
{user_prompt}

RealSense depth 정보:
{depth_text}

위 RGB 이미지와 depth 정보를 함께 보고 판단해라.
"""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_prompt},
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

    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="332322071907")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[INFO] Loading model: {args.model}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model)

    print(f"[INFO] Starting RealSense serial={args.serial}")

    pipeline, align, depth_scale = start_realsense(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print("[INFO] Ready.")
    print("질문을 입력하면 RealSense RGB+Depth 1프레임을 캡처해서 Qwen3-VL에 넣습니다.")
    print("종료: q / exit / quit")

    try:
        while True:
            user_prompt = input("\nUSER> ").strip()

            if user_prompt.lower() in {"q", "exit", "quit"}:
                break

            if not user_prompt:
                continue

            image, depth_np = capture_aligned_rgb_depth(pipeline, align)
            depth_text = summarize_depth(depth_np, depth_scale)

            print("\n[DEPTH]")
            print(depth_text)

            answer = run_qwen_vl(
                model=model,
                processor=processor,
                image=image,
                user_prompt=user_prompt,
                depth_text=depth_text,
                max_new_tokens=args.max_new_tokens,
            )

            print(f"\nQWEN> {answer}")

    finally:
        print("\n[INFO] Stopping RealSense pipeline.")
        pipeline.stop()


if __name__ == "__main__":
    main()