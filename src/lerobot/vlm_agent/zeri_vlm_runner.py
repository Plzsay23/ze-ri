# zeri_vlm_runner.py
import json

from typing import Any, Dict

import torch
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from .zeri_vlm_constants import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from .zeri_vlm_decision import extract_json_object, normalize_decision
    from .zeri_vlm_types import VLMDecision
except ImportError:
    from zeri_vlm_constants import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from zeri_vlm_decision import extract_json_object, normalize_decision
    from zeri_vlm_types import VLMDecision


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
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
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
