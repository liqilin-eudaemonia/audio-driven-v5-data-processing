"""Lazy local backends. Importing this module never loads pretrained weights."""
import inspect
import json
from pathlib import Path
import time

import numpy as np

from listen_labeling.guided import response_format
from .media import minicpm_native, timestamped_images, uniform_video


def context_limit(system, generation):
    config = json.loads((Path(system["model_path"]) / "config.json").read_text())
    text = config.get("thinker_config", config).get("text_config", config)
    available = text.get("max_position_embeddings", generation["context_limit"])
    return min(available, generation["context_limit"])


def grammar(tokenizer, branch, duration, stage, enabled):
    if not enabled:
        return {}
    from .decoding import prefix_constraint
    schema = response_format(branch, duration, stage)["json_schema"]["schema"]
    return {"prefix_allowed_tokens_fn": prefix_constraint(tokenizer, schema)}


def hf_inputs(processor, system, media, text, fps):
    """Processor-only function, also usable for CPU adapter smoke checks."""
    branch, backend = system["branch"], system["backend"]
    native = branch == "omni" and system["representation"] == "native"
    parts = [] if backend == "captioner" else [{"type": "text", "text": text}]
    kwargs = {"return_tensors": "pt", "padding": True}
    representation = {"representation": system["representation"]}
    if native:
        video, representation = uniform_video(media, fps)
        parts.append({"type": "video", "video": "listener://already-decoded"})
        # Pass arrays directly: no decoding of embedded MP4 audio and no second crop.
        kwargs.update(videos=[video], audio=[media["audio"]], fps=fps,
                      use_audio_in_video=True, do_sample_frames=False)
    else:
        if branch in ("visual", "omni"):
            parts.extend(timestamped_images(media))
            kwargs["images"] = media["images"]
        if branch in ("audio", "omni"):
            parts.append({"type": "audio", "audio": "listener://independent-wav",
                          "audio_url": "listener://independent-wav"})
            # Transformers 4.x used `audios`; 5.x renamed it to `audio`.
            # An ignored keyword can otherwise yield a text-only "successful" request.
            audio_key = "audio"
            if backend == "qwen2_audio" and "audios" in inspect.signature(processor.__call__).parameters:
                audio_key = "audios"
            kwargs[audio_key] = [media["audio"]]
            kwargs["sampling_rate"] = 16000
        if backend == "qwen3_omni":
            kwargs["use_audio_in_video"] = False
    messages = [{"role": "user", "content": parts}]
    template_options = {} if backend == "captioner" else {"enable_thinking": False}
    template = processor.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False, **template_options)
    inputs = processor(text=template, **kwargs)
    if branch in ("audio", "omni") and "input_features" not in inputs:
        raise ValueError("Processor dropped listener audio; refusing a text-only request")
    if branch in ("visual", "omni") and not ({"pixel_values", "pixel_values_videos"} & inputs.keys()):
        raise ValueError("Processor dropped listener frames")
    representation["processor_shapes"] = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
    if "video_second_per_grid" in inputs:
        value = inputs["video_second_per_grid"]
        representation["video_second_per_grid"] = value.tolist() if hasattr(value, "tolist") else value
    return inputs, template, representation


def finish_reason(ids, tokenizer, maximum):
    count = len(ids)
    eos = tokenizer.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    # End tokens can differ from tokenizer.eos_token_id (handled by caller).
    if count and int(ids[-1]) in eos:
        return "stop"
    return "length" if count >= maximum else "stopped_without_eos"


class TransformersBackend:
    def __init__(self, system, device, generation):
        import torch
        import transformers as tf
        self.torch, self.system, self.generation = torch, system, generation
        backend = system["backend"]
        path = system["model_path"]
        classes = {"qwen3_omni": "Qwen3OmniMoeForConditionalGeneration",
                   "captioner": "Qwen3OmniMoeForConditionalGeneration",
                   "qwen3_vl": "Qwen3VLForConditionalGeneration",
                   "qwen2_audio": "Qwen2AudioForConditionalGeneration",
                   "joy": "AutoModelForImageTextToText"}
        model_class = getattr(tf, classes[backend])
        self.processor = tf.AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = model_class.from_pretrained(path, torch_dtype=torch.bfloat16,
            device_map={"": device}, attn_implementation=system.get("attention", "sdpa"),
            trust_remote_code=True, local_files_only=True).eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

    def infer(self, media, text, stage, fps):
        inputs, template, representation = hf_inputs(self.processor, self.system, media, text, fps)
        tokens = int(inputs["input_ids"].shape[1])
        maximum = self.generation["max_new_tokens"]
        if tokens + maximum > context_limit(self.system, self.generation):
            raise ValueError("Input plus output budget exceeds context; no silent truncation")
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        backend = self.system["backend"]
        guided = self.system.get("guided_json", self.generation["guided_json"]) and backend != "captioner"
        options = grammar(self.processor.tokenizer, self.system["branch"], media["duration_s"], stage, guided)
        options.update(do_sample=False, num_beams=1, max_new_tokens=maximum,
                       max_time=self.generation["max_generation_seconds"], return_dict_in_generate=True)
        with self.torch.inference_mode():
            if backend in ("qwen3_omni", "captioner"):
                options = {"thinker_" + k: v for k, v in options.items()}
                output = self.model.generate(**inputs, **options, return_audio=False,
                    use_audio_in_video=self.system["branch"] == "omni" and self.system["representation"] == "native")
            else:
                output = self.model.generate(**inputs, **options)
        if isinstance(output, tuple):
            output = output[0]
        sequences = output.sequences if hasattr(output, "sequences") else output
        ids = sequences[0, tokens:].detach().cpu().tolist()
        result = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        finish = finish_reason(ids, self.processor.tokenizer, maximum)
        config = self.model.thinker.generation_config if hasattr(self.model, "thinker") else self.model.generation_config
        eos = config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        if ids and ids[-1] in eos:
            finish = "stop"
        return {"text": result, "finish_reason": finish, "generated_token_ids": ids,
                "prompt_tokens": tokens, "completion_tokens": len(ids), "rendered_prompt": template,
                "representation": representation, "guided_json": guided}


class MiniCPMBackend:
    def __init__(self, system, device, generation):
        import torch
        from transformers import AutoModel
        self.system, self.generation = system, generation
        self.model = AutoModel.from_pretrained(system["model_path"], trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=system.get("attention", "sdpa"),
            init_vision=True, init_audio=True, init_tts=False).eval().to(device)
        self.model.prepare_processor()

    def infer(self, media, text, stage, fps):
        contents = [text]
        representation = {"representation": self.system["representation"]}
        if self.system["representation"] == "native":
            from minicpmo.utils import concat_images
            native, representation = minicpm_native(media, concat_images)
            contents.extend(native)
        else:
            for part in timestamped_images(media) if media["images"] else []:
                contents.append(part.get("image", part.get("text")))
            if media["audio"] is not None:
                contents.append(media["audio"])
        maximum = self.generation["max_new_tokens"]
        tokenizer = self.model.processor.tokenizer
        options = grammar(tokenizer, self.system["branch"], media["duration_s"], stage,
                          self.generation["guided_json"])
        captured = {}
        original = self.model.generate

        def generate_checked(*args, **kwargs):
            tokens = int(kwargs["input_ids"].shape[-1])
            if tokens + maximum > context_limit(self.system, self.generation):
                raise ValueError("Input plus output budget exceeds context; no silent truncation")
            captured["prompt_tokens"] = tokens
            # MiniCPM.chat filters generation kwargs: inject at generate, after that filter.
            kwargs.update(options, max_time=self.generation["max_generation_seconds"])
            result = original(*args, **kwargs)
            captured["ids"] = result[1].sequences[0].detach().cpu().tolist()
            return result

        self.model.generate = generate_checked
        try:
            answer, template = self.model.chat(msgs=[{"role": "user", "content": contents}],
                max_new_tokens=maximum, max_inp_length=None, do_sample=False, num_beams=1,
                enable_thinking=False, generate_audio=False, return_prompt=True,
                omni_mode=self.system["branch"] == "omni", merge_audio_from_same_content=False,
                max_slice_nums=self.system.get("max_slice_nums", 1), use_image_id=False)
        finally:
            self.model.generate = original
        ids = captured["ids"]
        end_tokens = {tokenizer.convert_tokens_to_ids(t) for t in self.model.terminators}
        finish = "stop" if ids and ids[-1] in end_tokens else finish_reason(ids, tokenizer, maximum)
        return {"text": answer, "finish_reason": finish, "generated_token_ids": ids,
                "prompt_tokens": captured["prompt_tokens"], "completion_tokens": len(ids),
                "rendered_prompt": template, "representation": representation,
                "guided_json": self.generation["guided_json"]}


class ASRBackend:
    def __init__(self, system, device, generation):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(system["model_path"], dtype=torch.bfloat16,
            device_map=device, max_inference_batch_size=1, max_new_tokens=generation["max_new_tokens"])

    def infer(self, media, text, stage, fps):
        item = self.model.transcribe(audio=(media["audio"], 16000), language=None)[0]
        return {"text": item.text, "language": item.language, "finish_reason": "not_exposed_by_asr",
                "word_timestamps": None, "auxiliary_only": True,
                "notes": "No forced aligner configured; empty transcript is not a negative vocalization label"}


class FaceBackend:
    def __init__(self, system, device, generation):
        import mediapipe as mp
        self.mp, self.path = mp, system["model_path"]

    def infer(self, media, text, stage, fps):
        mp = self.mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            output_facial_transformation_matrixes=True)
        records = []
        # Tracking state is reset for every clip, not shared across people.
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            for image, record in zip(media["images"], media["meta"]["video"]["frames"]):
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.asarray(image)), round(record["time_s"] * 1000))
                records.append({"time_s": record["time_s"], "face_detected": bool(result.face_landmarks),
                    "transformation_matrices": [m.tolist() for m in result.facial_transformation_matrixes],
                    "landmarks": [[[p.x, p.y, p.z] for p in face] for face in result.face_landmarks]})
        return {"text": "", "frames": records, "auxiliary_only": True,
                "notes": "Geometry evidence only; missing face is unknown, no automatic head_nod conversion"}


def load_backend(system, device, generation):
    return {"minicpm": MiniCPMBackend, "asr": ASRBackend,
            "mediapipe": FaceBackend}.get(system["backend"], TransformersBackend)(system, device, generation)
