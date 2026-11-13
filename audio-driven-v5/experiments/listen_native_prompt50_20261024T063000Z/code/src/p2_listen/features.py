"""Real model adapters, loaded only by explicit cache/generation commands."""
import gc
import math
from pathlib import Path
import torch
from .config import extractor_config
from .contracts import ROOT, path_at, sha, input_identity, read
from .cache import save_part, load_part, cpu, tree_to
from .layout import make_layout, aggregate_audio, geometry
from p3_listen.prompts import OMNI, GEMMA, FixedPrompt, contract as prompts


def unload():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def weights_identity(c, kind):
    cfg = extractor_config(c)
    p = Path(cfg.omni_path if kind == "omni" else cfg.ltx_path if kind == "ltx" else cfg.gemma_path)
    files = sorted(p.glob("*.json")) + sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(p)
    return {str(f): dict(size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns,
                        sha256=sha(f) if f.suffix == ".json" else None) for f in files}


def identity(c, part, obs=None, target=None, label=None):
    # Only relevant inputs participate; changing event labels cannot invalidate H.
    code = {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("layout.py"), ROOT / "src/p3_listen/features.py"]}
    if part == "text":
        return dict(part=part, prompts=prompts(), negative_prompt="", code=code,
                    ltx=weights_identity(c,"ltx"), gemma=weights_identity(c,"gemma"))
    value = dict(part=part, input=input_identity(obs), code=code)
    if part in ("input", "teacher"):
        value.update(extractor=extractor_config(c).to_dict(), omni=weights_identity(c,"omni"), prompts=prompts())
    if part == "teacher":
        value["target"] = target
    elif part == "vae":
        value.update(target=target, ltx=weights_identity(c,"ltx"), mask_policy="causal-prefix-v1")
    elif part == "events":
        value.update(label=label, grid=[c["extractor"]["vision_fps"], c["extractor"]["audio_hz"]],
                     annotation_code=sha(Path(__file__).with_name("annotations.py")))
    return value


def folder(c, part, sid=None):
    root = path_at(c["output"]) / "cache" / part
    return root if sid is None else root / sid


def make_observation(obs):
    from omni_av.data import Observation
    return Observation(obs["sample_id"], obs["speaker_audio"], obs["listener_image"],
                       0., obs["requested_duration_s"], obs["requested_duration_s"])


def video_inputs(processor, frames):
    from omni_av.omni import move_inputs
    return dict(processor.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False, do_resize=False))


def image_grid(obs, processor, cfg):
    import numpy as np
    from PIL import Image
    with Image.open(obs["listener_image"]) as im:
        im = im.convert("RGB")
        scale = min(1., math.sqrt(cfg.vision_max_pixels / (im.width * im.height)))
        patch = processor.image_processor.patch_size * processor.image_processor.merge_size
        size = (max(patch, round(im.width * scale / patch) * patch),
                max(patch, round(im.height * scale / patch) * patch))
        frame = np.asarray(im.resize(size))
    return frame, size, (size[1] // patch, size[0] // patch)


@torch.no_grad()
def extract_input(c, obs, runtime, processor):
    from p3_listen.features import prefix
    from omni_av_p2.data import select_hidden_layers
    cfg = runtime.config
    pref = prefix(runtime, processor, make_observation(obs))
    outputs = runtime.thinker.model(inputs_embeds=pref["prefix"], position_ids=pref["prefix_positions"],
        attention_mask=pref["prefix_valid"].long(), output_hidden_states=True, use_cache=False, return_dict=True)
    h = select_hidden_layers(outputs.hidden_states, cfg.hidden_layers, runtime.thinker.config.text_config.num_hidden_layers)
    _, size, grid = image_grid(obs, processor, cfg)
    layout = make_layout(obs["requested_duration_s"], grid, cfg.vision_fps, cfg.audio_hz)
    condition = dict(hidden=h, hidden_valid=pref["prefix_valid"], layer_ids=torch.tensor(cfg.hidden_layers, device=h.device))
    for mod in ("video", "audio"):
        for key in ("_coordinates", "_valid"):
            condition[mod + key] = layout[mod + key]
    return cpu(dict(condition=condition, layout=layout, teacher_size=list(size)))


def sample_frames(path, times, size):
    import av
    import numpy as np
    from PIL import Image
    with av.open(str(path)) as video:
        decoded = [(float(f.time), f.to_ndarray(format="rgb24")) for f in video.decode(video=0)]
    if not decoded:
        raise ValueError("Empty target video")
    return np.stack([np.asarray(Image.fromarray(min(decoded, key=lambda x:abs(x[0]-t))[1]).resize(size)) for t in times])


@torch.no_grad()
def extract_teacher(c, obs, target, inputs, runtime, processor):
    import numpy as np
    from omni_av.data import read_audio
    from omni_av.omni import move_inputs
    model = runtime.thinker
    device = next(model.parameters()).device
    layout = inputs["layout"]
    frames_n = layout["video_intervals"].shape[1] * 2
    frames = sample_frames(target["target_video"], [i / 8 for i in range(frames_n)], inputs["teacher_size"])
    vi = move_inputs(video_inputs(processor, frames), device)
    video = model.get_video_features(vi["pixel_values_videos"], vi["video_grid_thw"])[None]
    wave = read_audio(target["target_audio"], end_s=obs["requested_duration_s"])
    ai = processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding="max_length")
    audio = model.get_audio_features(ai["input_features"].to(device,torch.bfloat16), ai["attention_mask"].to(device))
    if audio.ndim != 2:
        raise ValueError("Unexpected native audio feature shape")
    if not math.isclose(processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate, .01):
        raise ValueError("Unsupported native audio timing")
    audio, audio_valid = aggregate_audio(audio[None], layout["audio_intervals"])
    result = dict(video=video, audio=audio,
                  video_valid=layout["video_target_valid"].to(device),
                  audio_valid=layout["audio_target_valid"].to(device) & audio_valid)
    for mod in ("video", "audio"):
        if result[mod].shape != (*inputs["condition"][mod+"_valid"].shape, runtime.config.target_dim):
            raise ValueError("Teacher/query shape mismatch")
    return cpu(result)


def load_input(c, obs, device="cpu"):
    from omni_av_p2.connector import ConditionInputs
    p = load_part(folder(c,"input",obs["sample_id"]), identity(c,"input",obs), device)
    x = ConditionInputs(**p["condition"])
    x.validate()
    return x, p["layout"]


def load_training_sample(c, obs, target, device="cpu"):
    x, layout = load_input(c, obs, device)
    teacher = load_part(folder(c,"teacher",obs["sample_id"]), identity(c,"teacher",obs,target), device)
    av = load_part(folder(c,"vae",obs["sample_id"]), identity(c,"vae",obs,target), device)
    label = read(path_at(c["output"]) / "labels/canonical" / (obs["sample_id"] + ".json"))
    events = load_part(folder(c,"events",obs["sample_id"]), identity(c,"events",obs,label=label),device)
    return x, layout, teacher, av, events


def load_text(c, device="cpu"):
    return load_part(folder(c,"text"), identity(c,"text"),device)


def cache_parts(c, observations, targets, parts, device="cuda:0"):
    from .annotations import validate, rasterize
    cfg = extractor_config(c)
    if any(p in parts for p in ("input","teacher")):
        from omni_av_p2.data import load_omni
        runtime, processor = load_omni(cfg,device)
        try:
            for obs in observations:
                ident = identity(c,"input",obs)
                data = load_part(folder(c,"input",obs["sample_id"]),ident,missing_ok=True)
                if data is None:
                    data = extract_input(c,obs,runtime,processor)
                    save_part(folder(c,"input",obs["sample_id"]),ident,data)
                if "teacher" in parts:
                    t = targets[obs["sample_id"]]
                    ident = identity(c,"teacher",obs,t)
                    if load_part(folder(c,"teacher",obs["sample_id"]),ident,missing_ok=True) is None:
                        save_part(folder(c,"teacher",obs["sample_id"]),ident,extract_teacher(c,obs,t,data,runtime,processor))
        finally:
            del runtime, processor
            unload()
    if "text" in parts:
        ident = identity(c,"text")
        if load_part(folder(c,"text"),ident,missing_ok=True) is None:
            from omni_av.ltx_runtime import NativeTextConditioner
            encoder = NativeTextConditioner(cfg,device)
            try:
                save_part(folder(c,"text"),ident,dict(positive=encoder.encode(FixedPrompt(GEMMA)),negative=encoder.encode(FixedPrompt(""))))
            finally:
                encoder.close()
                unload()
    if "vae" in parts:
        from .runtime import encode_vae
        for obs in observations:
            t = targets[obs["sample_id"]]
            ident = identity(c,"vae",obs,t)
            if load_part(folder(c,"vae",obs["sample_id"]),ident,missing_ok=True) is None:
                save_part(folder(c,"vae",obs["sample_id"]),ident,encode_vae(c,obs,t,device))
    if "events" in parts:
        for obs in observations:
            label = read(path_at(c["output"]) / "labels/canonical" / (obs["sample_id"] + ".json"))
            validate(label,obs)
            _,layout = load_input(c,obs)
            save_part(folder(c,"events",obs["sample_id"]),identity(c,"events",obs,label=label),rasterize(label,layout))
