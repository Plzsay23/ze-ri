# qwen3vl_live_camera_chat.py

import argparse
import time
from typing import Union

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_SYSTEM_PROMPT = """
너는 재난 상황 초동 조치를 위한 모바일 매니퓰레이터 Ze-Ri의 VLM 에이전트다.
카메라 장면과 사용자 명령을 바탕으로 현재 상황을 판단한다.
답변은 간결하게 하되, 로봇이 해야 할 행동을 명확히 제시한다.
"""


def parse_camera_source(camera: str) -> Union[int, str]:
    """
    "0"이면 OpenCV index 0으로 처리.
    "/dev/video0"이면 device path로 처리.
    """
    if camera.isdigit():
        return int(camera)
    return camera


def open_camera(camera: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    source = parse_camera_source(camera)

    if isinstance(source, str) and source.startswith("/dev/video"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def capture_frame(cap: cv2.VideoCapture, warmup: int = 5) -> Image.Image:
    frame = None

    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)

    ret, frame = cap.read()

    if not ret or frame is None:
        raise RuntimeError("Camera frame read failed.")

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def run_qwen_vl(
    model,
    processor,
    image: Image.Image,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
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
    )

    inputs = inputs.to(model.device)

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
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--camera", default="/dev/video0")
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

    print(f"[INFO] Opening camera: {args.camera}")

    cap = open_camera(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print("[INFO] Ready.")
    print("질문을 입력하면 현재 카메라 프레임 1장을 캡처해서 VLM에 넣습니다.")
    print("종료: exit / quit / q")

    try:
        while True:
            user_prompt = input("\nUSER> ").strip()

            if user_prompt.lower() in {"exit", "quit", "q"}:
                break

            if not user_prompt:
                continue

            image = capture_frame(cap)

            answer = run_qwen_vl(
                model=model,
                processor=processor,
                image=image,
                user_prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
            )

            print(f"\nQWEN> {answer}")

    finally:
        cap.release()


if __name__ == "__main__":
    main()