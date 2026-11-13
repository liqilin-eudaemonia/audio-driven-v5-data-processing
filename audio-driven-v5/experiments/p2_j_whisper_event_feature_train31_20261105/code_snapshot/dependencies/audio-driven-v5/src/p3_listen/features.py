"""Frozen perception embeddings and native VAE/text caches with separate inference layouts."""
import gc
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import digest, read, resolve, sha, write, write_once
from .media import paths, selection, verify_media
from .prompts import FixedPrompt, contract, observation_messages


def unload():
    gc.collect()
    torch.cuda.empty_cache()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending.pt")
    torch.save(value, tmp)
    tmp.replace(path)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to(v, device) for k, v in value.items()}
    return value


def observation(root, row, speaker_audio=None):
    from omni_av.data import Observation
    p = paths(root, row)
    return Observation(row["sample_id"], str(speaker_audio or p["speaker_audio"]), str(p["listener_image"]),
                       0., row["duration_s"], row["duration_s"])


def prepare_prefix(cfg, processor, obs):
    """CPU-only preprocessing, suitable for a bounded producer pool."""
    from omni_av.data import read_audio
    wave = read_audio(obs.speaker_audio, end_s=obs.speaker_end_s)
    if len(wave) != round(obs.output_duration_s * 16000):
        raise ValueError("Speaker audio must cover the exact window")
    with Image.open(obs.listener_image) as im:
        image = im.convert("RGB")
    messages = observation_messages()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return dict(processor(text=[prompt], images=[image], audio=[wave], sampling_rate=16000,
        return_tensors="pt", padding=True, truncation=False,
        images_kwargs={"max_pixels": cfg.vision_max_pixels, "min_pixels": 28 * 28 * 4}))


@torch.no_grad()
def prefix(runtime, processor, obs, prepared=None):
    from omni_av.omni import move_inputs
    model = runtime.thinker
    device = next(model.parameters()).device
    inputs = move_inputs(prepare_prefix(runtime.config,processor,obs) if prepared is None else prepared, device)
    ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(ids)
    audio = model.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
    image = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    image_mask, _, audio_mask = model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=image)
    if audio_mask.sum() != audio.numel() or image_mask.sum() != image.numel():
        raise ValueError("Multimodal placeholder mismatch")
    embeds = embeds.masked_scatter(audio_mask, audio.to(embeds)).masked_scatter(image_mask, image.to(embeds))
    positions, _ = model.get_rope_index(ids, image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"], use_audio_in_video=False,
        audio_seqlens=inputs["feature_attention_mask"].sum(-1))
    return dict(prefix=embeds.detach(), prefix_positions=positions, prefix_valid=inputs["attention_mask"].bool())


def visual_inputs(processor, frames, cfg):
    return dict(processor.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False,
                max_pixels=cfg.vision_max_pixels, min_pixels=28 * 28 * 4))


def audio_inputs(processor, wave):
    return processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt",
                                       return_attention_mask=True, padding="max_length")


@torch.no_grad()
def inference_layout(runtime, processor, obs):
    """Target coordinates depend ONLY on duration/reference geometry, including silence slots."""
    from omni_av_p3.model import Layout
    cfg, model = runtime.config, runtime.thinker
    pref = prefix(runtime, processor, obs)
    with Image.open(obs.listener_image) as im:
        image = im.convert("RGB")
        scale = min(1., math.sqrt(cfg.vision_max_pixels / (image.width * image.height)))
        image = image.resize((max(28, round(image.width * scale)), max(28, round(image.height * scale))))
        frame = np.asarray(image)
    n = math.ceil(obs.output_duration_s * cfg.vision_fps)
    frames = np.repeat(frame[None], n + n % 2, axis=0)
    vi = visual_inputs(processor, frames, cfg)
    tg, gh, gw = map(int, vi["video_grid_thw"][0].tolist())
    merge = processor.image_processor.merge_size
    patch_t = processor.video_processor.temporal_patch_size
    if patch_t != 2 or gh % merge or gw % merge:
        raise ValueError("Unexpected native video grid")
    # The audio tower output length is a function of sample count, never VAD.
    ai = audio_inputs(processor, np.zeros(round(obs.output_duration_s * 16000), dtype=np.float32))
    _, length = model.audio_tower._get_feat_extract_output_lengths(ai["attention_mask"].sum(-1))
    na = int(length.item())
    hop = processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate
    if not math.isclose(hop, .01):
        raise ValueError("Unexpected audio feature hop")
    coords = [[(t * patch_t + .5) / cfg.vision_fps, y, x]
              for t in range(tg) for y in range(gh // merge) for x in range(gw // merge)]
    nv = len(coords)
    coords += [[(4 * t + 2) * hop, 0, 0] for t in range(na)]
    device = pref["prefix"].device
    layout = Layout(**pref, coordinates=torch.tensor([coords], device=device, dtype=torch.float32),
                    modalities=torch.tensor([[0] * nv + [1] * na], device=device),
                    valid=torch.ones(1, nv + na, dtype=torch.bool, device=device))
    layout.validate()
    return layout


@torch.no_grad()
def extract_native(runtime, processor, obs, p):
    from omni_av.data import read_audio
    from omni_av.omni import sampled_frames, move_inputs
    cfg, model = runtime.config, runtime.thinker
    device = next(model.parameters()).device
    layout = inference_layout(runtime, processor, obs)
    frames = sampled_frames(p["target_video"], obs.output_duration_s, cfg.vision_fps, cfg.vision_max_pixels)
    vi = move_inputs(visual_inputs(processor, frames, cfg), device)
    ai = audio_inputs(processor, read_audio(p["target_audio"], end_s=obs.output_duration_s))
    video = model.get_video_features(vi["pixel_values_videos"], vi["video_grid_thw"])
    audio = model.get_audio_features(ai["input_features"].to(device, torch.bfloat16), ai["attention_mask"].to(device))
    targets = torch.cat([video, audio])[None].detach()
    if (len(video) != int((layout.modalities == 0).sum()) or len(audio) != int((layout.modalities == 1).sum())
            or targets.shape != (*layout.valid.shape, cfg.hidden_size) or not torch.isfinite(targets).all()):
        raise ValueError("Native GT shape differs from independently constructed inference layout")
    return cpu(vars(layout)), targets.cpu()


def cache_dir(root, row):
    return Path(root) / "cache" / row["split"] / row["sample_id"]


def native_contract(cfg):
    from dataclasses import asdict
    from omni_av_p3.model import Config
    c = Config(**cfg["planner"])
    return dict(planner=asdict(c), prompts=contract(),
                model_config=sha(Path(c.omni_path) / "config.json"),
                model_index=sha(Path(c.omni_path) / "model.safetensors.index.json"))


def validate_part(folder, part, expected):
    marker = Path(folder) / (part + ".json")
    if not marker.exists():
        return False
    data = read(marker)
    if data["contract"] != expected:
        raise ValueError(f"Stale {part} cache: {folder}")
    for name, value in data["files"].items():
        if sha(Path(folder) / name) != value:
            raise ValueError(f"Cache checksum mismatch: {name}")
    return True


def seal(folder, part, expected, files):
    write(Path(folder) / (part + ".json"), dict(contract=expected, files={f: sha(Path(folder) / f) for f in files}))


def load_native(root, row, cfg, targets=False, device="cpu"):
    from omni_av_p3.model import Layout
    folder = cache_dir(root, row)
    expected = dict(row=digest(row), native=digest(native_contract(cfg)), media=sha(paths(root, row)["marker"]))
    # Prediction loads layout only; no GT feature file is opened or hashed.
    marker = read(folder / "native.json")
    if marker["contract"] != expected:
        raise ValueError("Native cache contract mismatch")
    def load(name):
        if sha(folder / name) != marker["files"][name]:
            raise ValueError("Native cache checksum mismatch")
        return torch.load(folder / name, map_location=device, weights_only=True)
    layout = Layout(**load("layout.pt"))
    layout.validate()
    if not targets:
        return layout
    value = load("targets.pt")
    if value.shape != (*layout.valid.shape, layout.prefix.shape[-1]) or not torch.isfinite(value).all():
        raise ValueError("Invalid native target")
    return layout, value


def renderer_contract(cfg):
    from dataclasses import asdict
    from omni_av_p2.config import Config
    c = Config(**cfg["renderer"])
    # Full huge weights are not rehashed per sample; stat + config tracked, code bound at train time.
    stat = Path(c.ltx_path).stat()
    return dict(config=c.to_dict(), weights_size=stat.st_size, weights_mtime_ns=stat.st_mtime_ns)


def load_av(root, row, cfg, device):
    folder = cache_dir(root, row)
    expected = dict(row=digest(row), renderer=digest(renderer_contract(cfg)), media=sha(paths(root, row)["marker"]))
    if not validate_part(folder, "vae", expected):
        raise FileNotFoundError("VAE cache missing")
    av = torch.load(folder / "av.pt", map_location=device, weights_only=True)
    v, a = av["latents"]["latents"], av["audio_latents"]["latents"]
    if tuple(v.shape) != (1, 128, (row["frames"] - 1) // 8 + 1, row["height"] // 32, row["width"] // 32):
        raise ValueError("VAE video geometry mismatch")
    if a.ndim != 4 or a.shape[1] != 8 or a.shape[-1] != 16 or not torch.isfinite(v).all() or not torch.isfinite(a).all():
        raise ValueError("Invalid AV latents")
    return av


def load_text(root, cfg, device, negative=False):
    folder = Path(root) / "cache" / "text"
    expected = dict(prompts=contract(), renderer=renderer_contract(cfg))
    if not validate_part(folder, "text", expected):
        raise FileNotFoundError("Fixed text cache missing")
    return torch.load(folder / ("negative.pt" if negative else "fixed.pt"), map_location=device, weights_only=True)


def cache(cfg, args):
    from omni_av_p3.data import load_native as load_model
    from omni_av_p3.model import Config as PlannerConfig
    from omni_av_p2.config import Config as RendererConfig
    torch.set_num_threads(2)
    _, rows = selection(cfg, args)
    root, device = resolve(cfg["output"]), "cuda:0"
    for row in rows:
        verify_media(root, row)
    if args.part in ("native", "all"):
        contract_value = native_contract(cfg)
        write_once(root / "cache/native_contract.json", contract_value)
        runtime, processor = load_model(PlannerConfig(**cfg["planner"]), device)
        for row in rows:
            folder = cache_dir(root, row)
            expected = dict(row=digest(row), native=digest(contract_value), media=sha(paths(root, row)["marker"]))
            if not validate_part(folder, "native", expected):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    layout, targets = extract_native(runtime, processor, observation(root, row), paths(root, row))
                save(folder / "layout.pt", layout); save(folder / "targets.pt", targets)
                seal(folder, "native", expected, ["layout.pt", "targets.pt"])
                print(dict(native=row["sample_id"], shape=list(targets.shape)), flush=True)
        del runtime, processor
        unload()
    if args.part in ("text", "all"):
        from omni_av.ltx_runtime import NativeTextConditioner
        folder = root / "cache/text"
        expected = dict(prompts=contract(), renderer=renderer_contract(cfg))
        if not validate_part(folder, "text", expected):
            encoder = NativeTextConditioner(RendererConfig(**cfg["renderer"]), device)
            for name, prompt in (("fixed.pt", FixedPrompt()), ("negative.pt", FixedPrompt(""))):
                save(folder / name, cpu(encoder.encode(prompt)))
            encoder.close(); del encoder
            seal(folder, "text", expected, ["fixed.pt", "negative.pt"])
            print(dict(text="fixed and empty CFG encoded"), flush=True)
    if args.part in ("vae", "all"):
        cache_vae(cfg, rows, root, device)


@torch.no_grad()
def cache_vae(cfg, rows, root, device):
    import av
    import soundfile as sf
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_core.types import Audio
    from ltx_trainer.model_loader import load_video_vae_encoder, load_audio_vae_encoder
    from omni_av_p2.config import Config
    c = Config(**cfg["renderer"])
    ve = load_video_vae_encoder(c.ltx_path, device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    ae = load_audio_vae_encoder(c.ltx_path, device=device, dtype=torch.float32).requires_grad_(False).eval()
    ap = AudioProcessor(target_sample_rate=ae.sample_rate, mel_bins=ae.mel_bins,
                        mel_hop_length=ae.mel_hop_length, n_fft=ae.n_fft).to(device)
    for row in rows:
        folder, p = cache_dir(root, row), paths(root, row)
        expected = dict(row=digest(row), renderer=digest(renderer_contract(cfg)), media=sha(p["marker"]))
        if validate_part(folder, "vae", expected):
            continue
        with av.open(str(p["target_video"])) as video:
            pixels = np.stack([f.to_ndarray(format="rgb24") for f in video.decode(video=0)])
        if len(pixels) != row["frames"]:
            raise ValueError("Decoded frame count mismatch")
        value = torch.from_numpy(pixels).permute(3, 0, 1, 2)[None].to(device, torch.bfloat16) / 127.5 - 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = ve(value)
            first = ve(value[:, :, :1])
        # First-frame condition is independently encoded, never an encoding of the GT tail.
        z[:, :, :1] = first
        wave, rate = sf.read(str(p["target_audio"]), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(wave.T.copy())[None].to(device)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        az = ae(ap.waveform_to_mel(Audio(waveform=waveform, sampling_rate=rate)).float())
        scalar = lambda x: torch.tensor([x])
        result = dict(latents=dict(latents=z.cpu(), num_frames=scalar(z.shape[2]), height=scalar(z.shape[3]),
                                  width=scalar(z.shape[4]), fps=scalar(row["fps"])),
                      audio_latents=dict(latents=az.cpu().to(torch.bfloat16), num_time_steps=scalar(az.shape[2]),
                                         duration=scalar(row["duration_s"])))
        if not torch.isfinite(z).all() or not torch.isfinite(az).all():
            raise ValueError("Nonfinite VAE cache")
        save(folder / "av.pt", result)
        save(folder / "reference.pt", first.cpu())
        seal(folder, "vae", expected, ["av.pt", "reference.pt"])
        print(dict(vae=row["sample_id"], video=list(z.shape), audio=list(az.shape)), flush=True)
        del value, z, first, az, result
    del ve, ae, ap
    unload()
