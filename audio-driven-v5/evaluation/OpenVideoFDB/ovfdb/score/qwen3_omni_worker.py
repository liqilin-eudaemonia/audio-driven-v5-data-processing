"""Long-running Qwen3-Omni judge. One JSON object per stdin line, one per stdout line.

The OpenVideoFDB scorer does not ship this model. A separate Python that provides
``Qwen3OmniMoeForConditionalGeneration`` runs this file. The model directory comes
from ``QWEN3_OMNI_PATH`` and is loaded on the first request.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

_STATE = None


def _path() -> Path:
    raw = os.environ.get("QWEN3_OMNI_PATH", "").strip()
    if not raw:
        raise RuntimeError("QWEN3_OMNI_PATH is not set")
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"QWEN3_OMNI_PATH is not a directory: {path}")
    return path


def _load():
    global _STATE
    if _STATE is not None:
        return _STATE
    import torch
    import transformers as tf

    path = _path()
    device = os.environ.get("QWEN3_OMNI_DEVICE", "cuda:0").strip() or "cuda:0"
    cls = getattr(tf, "Qwen3OmniMoeForConditionalGeneration", None)
    if cls is None:
        raise RuntimeError("this Python has no Qwen3OmniMoeForConditionalGeneration")
    processor = tf.AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = cls.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True,
    ).eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    _STATE = (processor, model)
    return _STATE


def _move(inputs, model):
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(model.device)
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.to(model.dtype)
        moved[key] = value
    return moved


def complete(system: str, user: str, images: list[str]) -> dict:
    from PIL import Image

    processor, model = _load()
    pictures = [Image.open(path).convert("RGB") for path in images]
    content = [{"type": "image", "image": picture} for picture in pictures]
    content.append({"type": "text", "text": user})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
    template_args = {"add_generation_prompt": True, "tokenize": False}
    try:
        text = processor.apply_chat_template(messages, enable_thinking=False, **template_args)
    except TypeError:
        text = processor.apply_chat_template(messages, **template_args)
    kwargs = {"text": text, "return_tensors": "pt", "padding": True}
    if pictures:
        kwargs["images"] = pictures
    inputs = _move(processor(**kwargs), model)
    import torch

    prompt_tokens = int(inputs["input_ids"].shape[1])
    options = {
        "thinker_do_sample": False,
        "thinker_num_beams": 1,
        "thinker_max_new_tokens": 1024,
        "thinker_return_dict_in_generate": True,
    }
    with torch.inference_mode():
        output = model.generate(**inputs, **options, return_audio=False, use_audio_in_video=False)
    if isinstance(output, tuple):
        output = output[0]
    sequences = output.sequences if hasattr(output, "sequences") else output
    new_ids = sequences[0, prompt_tokens:]
    reply = processor.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": reply, "prompt_tokens": prompt_tokens, "completion_tokens": int(new_ids.shape[0])}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            result = complete(request.get("system") or "", request.get("user") or "", list(request.get("images") or []))
            result["id"] = request.get("id")
        except Exception as exc:  # noqa: BLE001 - the parent scores one axis and must see the failure
            traceback.print_exc(file=sys.stderr)
            result = {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
