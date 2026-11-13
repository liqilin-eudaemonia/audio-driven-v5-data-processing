"""Native Qwen2.5-Omni perception, reply and planning-prefix preparation."""
import hashlib
import math
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import Observation, TextPlan, read_audio
from .lora import inject_planning_lora, native_reply_mode
from .planner import AVPlanner, PlanningBatch

SYSTEM = (
    "You are planning the listener's response in a conversation. Use the observed speaker audio "
    "and the listener reference image. Write exactly three short lines: Reply:, Delivery:, Behavior:. "
    "Reply contains the actual words to say in the conversation's language. Delivery describes voice "
    "and pacing. Behavior describes visible listening and response actions. Do not explain your reasoning. "
    "Use [no main spoken reply] only if no response is appropriate. Do not invent extra speakers."
)

OBSERVATION_USER_PROMPT = (
    "Plan a context-specific listener response based only on the provided speaker audio "
    "and listener reference image."
)


def observation_prompt_contract():
    """Changing either prompt invalidates frozen hidden caches and their checkpoints."""
    return dict(policy="observation_no_time_v1",
                system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
                user_sha256=hashlib.sha256(OBSERVATION_USER_PROMPT.encode()).hexdigest())


def load_planner(config, device="cuda:0"):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(config.omni_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        config.omni_path, dtype=torch.bfloat16, device_map={"": device},
        local_files_only=True, attn_implementation="sdpa")
    if thinker.config.text_config.hidden_size != config.hidden_size:
        raise ValueError("Configured hidden dimension differs from checkpoint")
    inject_planning_lora(thinker, config.lora_rank, config.lora_alpha)
    if config.gradient_checkpointing:
        thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    planner = AVPlanner(thinker, config)
    # Keep trainable parameters FP32; frozen base stays BF16.
    for name in ("video_flow", "audio_flow"):
        getattr(planner, name).to(device)
    planner.mask_token.data = planner.mask_token.data.to(device)
    planner.modality_embedding.data = planner.modality_embedding.data.to(device)
    planner.train()
    return planner, processor


def move_inputs(inputs, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def prepare_observation(processor, obs: Observation, config):
    audio = read_audio(obs.speaker_audio, end_s=obs.speaker_end_s, start_s=obs.speaker_start_s)
    with Image.open(obs.listener_image) as image:
        image = image.convert("RGB")
    messages = [dict(role="system", content=[dict(type="text", text=SYSTEM)]), dict(role="user", content=[
        dict(type="image"), dict(type="audio"), dict(type="text", text=OBSERVATION_USER_PROMPT)])]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return dict(processor(text=[prompt], images=[image], audio=[audio], sampling_rate=16000,
                          return_tensors="pt", padding=True, truncation=False,
                          images_kwargs={"max_pixels": config.vision_max_pixels,
                                         "min_pixels": 28 * 28 * 4}))


@torch.no_grad()
def generate_reply(planner, processor, prepared, max_new_tokens=256):
    model = planner.thinker
    inputs = move_inputs(prepared, next(model.parameters()).device)
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    with native_reply_mode(model), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        model.rope_deltas = None
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             use_audio_in_video=False, eos_token_id=eos,
                             pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
    reply_ids = ids[0, inputs["input_ids"].shape[1]:]
    ended = bool((reply_ids == eos).any())
    raw = processor.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    if not ended:
        raise ValueError(f"Native plan did not end within token budget: {raw}")
    return TextPlan.parse(raw), raw


@torch.no_grad()
def encode_prefix(planner, processor, prepared, text_plan):
    """Frozen multimodal embeddings, not cached trainable Thinker hidden states."""
    model = planner.thinker
    device = next(model.parameters()).device
    inputs = move_inputs(prepared, device)
    plan_ids = processor.tokenizer.encode(text_plan.serialize() + "<|im_end|>", add_special_tokens=False)
    plan_ids = torch.tensor([plan_ids], device=device)
    p = inputs["input_ids"].shape[1]
    ids = torch.cat([inputs["input_ids"], plan_ids], 1)
    valid = torch.cat([inputs["attention_mask"].bool(), torch.ones_like(plan_ids, dtype=torch.bool)], 1)
    with native_reply_mode(model):
        embeds = model.get_input_embeddings()(ids)
        audio = model.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
        image = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        image_mask, _, audio_mask = model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=image)
        if int(audio_mask.sum()) != audio.numel():
            raise ValueError("Audio placeholder count mismatch")
        embeds = embeds.masked_scatter(audio_mask, audio.to(embeds))
        embeds = embeds.masked_scatter(image_mask, image.to(embeds))
        positions, _ = model.get_rope_index(ids, image_grid_thw=inputs["image_grid_thw"],
            attention_mask=valid.long(), use_audio_in_video=False,
            audio_seqlens=inputs["feature_attention_mask"].sum(-1))
    text_mask = torch.zeros_like(valid)
    text_mask[:, p:] = True
    return dict(prefix=embeds.detach(), prefix_positions=positions, prefix_valid=valid,
                text_mask=text_mask)


def make_layout(prefix, duration_s, spatial_grid, config):
    """Output layout determined by external duration and reference-image spatial grid."""
    h, w = spatial_grid
    temporal_patch = 2  # Qwen2.5-Omni native video patch size, checked at extraction.
    groups = math.ceil(math.ceil(duration_s * config.vision_fps) / temporal_patch)
    rows = []
    mods = []
    for t in range(groups):
        timestamp = (t * temporal_patch + 0.5) / config.vision_fps
        for y in range(h):
            for x in range(w):
                rows.append([timestamp, y, x])
                mods.append(0)
    for t in range(math.ceil(duration_s * config.audio_hz)):
        rows.append([(t + 0.5) / config.audio_hz, 0, 0])
        mods.append(1)
    device = prefix["prefix"].device
    shape = (1, len(rows), config.hidden_size)
    return PlanningBatch(**prefix,
        targets=torch.zeros(shape, device=device, dtype=prefix["prefix"].dtype),
        coordinates=torch.tensor([rows], device=device, dtype=torch.float32),
        modalities=torch.tensor([mods], device=device, dtype=torch.long),
        valid=torch.ones((1, len(rows)), device=device, dtype=torch.bool))


def sampled_frames(path, duration, fps, max_pixels):
    import av
    count = math.ceil(duration * fps)
    frames, next_index = [], 0
    last = None
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            timestamp = float(frame.time or 0)
            if next_index < count and timestamp + 1e-6 >= next_index / fps:
                image = frame.to_image()
                scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
                image = image.resize((max(28, round(image.width * scale)), max(28, round(image.height * scale))))
                last = np.asarray(image)
                while next_index < count and timestamp + 1e-6 >= next_index / fps:
                    frames.append(last)
                    next_index += 1
            if next_index == count:
                break
    if last is None:
        raise ValueError("Video contains no decodable frames")
    # Tail extension is storage only; the teacher valid mask excludes unavailable targets.
    while len(frames) < count:
        frames.append(last)
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames)


@torch.no_grad()
def extract_targets(planner, processor, layout, video_path, audio_path, actual_duration_s, budget_s):
    if actual_duration_s > budget_s + 1e-3:
        raise ValueError("GT exceeds output budget; refusing truncation")
    cfg, model = planner.config, planner.thinker
    device = next(model.parameters()).device
    frames = sampled_frames(video_path, budget_s, cfg.vision_fps, cfg.vision_max_pixels)
    visual_coords = layout.coordinates[0, layout.modalities[0] == 0]
    patch = processor.image_processor.patch_size * processor.image_processor.merge_size
    size = (int(visual_coords[:, 2].max() + 1) * patch,
            int(visual_coords[:, 1].max() + 1) * patch)
    frames = np.stack([np.asarray(Image.fromarray(frame).resize(size)) for frame in frames])
    video = processor.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False,
                                      do_resize=False)
    wave = read_audio(audio_path)
    if len(wave) > round(budget_s * 16000) + 2:
        raise ValueError("Target audio exceeds output budget")
    wave = np.pad(wave[:round(budget_s * 16000)], (0, max(0, round(budget_s * 16000) - len(wave))))
    audio = processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt",
                                         return_attention_mask=True, padding="max_length")
    with native_reply_mode(model):
        vi = move_inputs(dict(video), device)
        vf = model.get_video_features(vi["pixel_values_videos"], vi["video_grid_thw"])
        af = model.get_audio_features(audio["input_features"].to(device, torch.bfloat16),
                                     audio["attention_mask"].to(device))
    nv = int((layout.modalities == 0).sum())
    na = int((layout.modalities == 1).sum())
    if vf.shape != (nv, cfg.hidden_size):
        raise ValueError(f"Target/source spatial grid mismatch: {vf.shape}, expected {(nv, cfg.hidden_size)}")
    af = F.adaptive_avg_pool1d(af.float().T[None], na)[0].T.to(vf.dtype)
    values = torch.cat([vf, af], 0)[None]
    valid = layout.coordinates[..., 0] < actual_duration_s
    values = values * valid[..., None]
    batch = replace(layout, targets=values.detach(), valid=valid)
    batch.validate()
    return batch
