"""Exercise real installed processors on CPU, without loading model tensors."""
import json
from listen_labeling.client import prompt
from listen_labeling.schema import CLASSES
from .backends import hf_inputs, context_limit, grammar
from .media import minicpm_native, prepare, timestamped_images


def check_processor(config, system, samples):
    if system["auxiliary"]:
        raise ValueError("processor-check currently targets the nine event annotator configurations")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(system["model_path"], trust_remote_code=True, local_files_only=True)
    results = []
    for row in samples:
        media_cfg = dict(config["media"], **system.get("media_override", {}))
        media = prepare(row, system["branch"], media_cfg)
        for stage in ("discovery", "audit"):
            text = prompt(system["branch"], row["duration_s"], stage)
            if system["backend"] == "minicpm":
                if system["representation"] == "native":
                    from minicpmo.utils import concat_images
                    content, representation = minicpm_native(media, concat_images)
                else:
                    content, representation = [], {"representation": "separate"}
                    for part in timestamped_images(media) if media["images"] else []:
                        content.append(part.get("image", part.get("text")))
                    if media["audio"] is not None:
                        content.append(media["audio"])
                import numpy as np
                from PIL import Image
                images, audios, parts = [], [], [text]
                for part in content:
                    if isinstance(part, Image.Image):
                        images.append(part); parts.append("<image>./</image>")
                    elif isinstance(part, np.ndarray):
                        audios.append(part); parts.append("<audio>./</audio>")
                    else:
                        parts.append(part)
                message = ("" if system["branch"] == "omni" else "\n").join(parts)
                template = processor.tokenizer.apply_chat_template([{"role": "user", "content": message}],
                    tokenize=False, add_generation_prompt=True, use_tts_template=bool(audios), enable_thinking=False)
                inputs = processor([template], [images], [audios], None, return_tensors="pt", max_length=None,
                                   max_slice_nums=system.get("max_slice_nums", 1), use_image_id=False)
                representation["processor_shapes"] = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
            else:
                inputs, template, representation = hf_inputs(processor, system, media, text, media_cfg["fps"])
            tokens = int(inputs["input_ids"].shape[-1])
            if tokens + config["generation"]["max_new_tokens"] > context_limit(system, config["generation"]):
                raise ValueError("Processor output exceeds context budget")
            allowed = grammar(processor.tokenizer, system["branch"], row["duration_s"], stage,
                              config["generation"]["guided_json"])
            if allowed:
                import torch
                prefix = inputs["input_ids"][0]
                if system["backend"] == "minicpm":
                    prefix = prefix[:0]  # MiniCPM generates from inputs_embeds, initially no token IDs.
                if not allowed["prefix_allowed_tokens_fn"](0, prefix):
                    raise ValueError("JSON constraint permits no first token")
                keys = (["visual", "audio"] if system["branch"] == "omni" else [system["branch"]]) if stage == "discovery" else CLASSES[system["branch"]]
                example = dict(reviewed={key: [[0, row["duration_s"]]] for key in keys},
                               unknown={key: [] for key in keys}, events=[], notes="")
                for token in processor.tokenizer.encode(json.dumps(example), add_special_tokens=False):
                    if token not in allowed["prefix_allowed_tokens_fn"](0, prefix):
                        raise ValueError("JSON constraint rejects a valid schema example")
                    prefix = torch.cat([prefix, torch.tensor([token], dtype=prefix.dtype)])
                if processor.tokenizer.eos_token_id not in allowed["prefix_allowed_tokens_fn"](0, prefix):
                    raise ValueError("JSON constraint does not allow termination after a valid object")
            results.append({"sample_id": row["sample_id"], "stage": stage, "prompt_tokens": tokens,
                            "representation": representation, "input_evidence": media["meta"],
                            "grammar_initialized": bool(allowed)})
    return {"system": system["name"], "results": results, "model_weights_loaded": False,
            "inference_performed": False, "passed": True}
