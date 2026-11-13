"""Source-only inference. Targets/annotations are deliberately not imported."""
from pathlib import Path
import subprocess
import torch
from .contracts import observation, digest, read, sha, atomic_json, path_at
from .config import extractor_config, source_identity
from .features import (extract_input, load_input, load_text, make_observation, unload,weights_identity)
from .cache import tree_to, cpu, atomic_torch
from .layout import geometry
from .runtime import guided_sample, append_negative


def generate(c, obs, output, *, checkpoint=None, seed=137, mode="normal", swap_audio=None, release=None, device="cuda:0"):
    cfg=extractor_config(c)
    obs=observation(obs)
    if mode not in ("normal","text_only","remove_u","audio_swap"):
        raise ValueError("Unknown inference mode")
    if mode!="text_only" and checkpoint is None: raise ValueError("Checkpoint required")
    if (mode=="audio_swap") != (swap_audio is not None): raise ValueError("audio_swap requires exactly one replacement audio")
    if obs["experiment_split"]=="test":
        if release is None: raise ValueError("Test requires a locked release JSON")
        r=read(release)
        if (r.get("config_sha256")!=digest(c) or (sha(checkpoint) if checkpoint else None) not in r.get("checkpoint_sha256",[]) or
            obs["sample_id"] not in r.get("sample_ids",[]) or seed not in r.get("seeds",[])):
            raise ValueError("Test release does not cover this generation")
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(sample_id=obs["sample_id"],group_id=obs["group_id"],sample_seed=seed,mode=mode,
                  config_sha256=digest(c),checkpoint_sha256=sha(checkpoint) if checkpoint else None,
                  observation_sha256=digest(obs),gt_access=False)
    try:
        from omni_av_p2.ltx import load_renderer,append_conditions,unpack_latents
        from omni_av_p2.inference import reference_latents
        from omni_av.ltx_runtime import with_text
        from omni_av_p3_staged.media import decode_and_save
        if swap_audio:
            obs=dict(obs,speaker_audio=str(Path(swap_audio).resolve()),media_hashes=dict(obs["media_hashes"],speaker_audio=sha(swap_audio)))
            from .dataset import verify_media
            verify_media(obs)
            metadata["swapped_observation_sha256"]=digest(obs)
        u=None
        if mode!="text_only":
            from .training import checked_checkpoint
            from .model import ListeningExtractor
            saved=checked_checkpoint(checkpoint)
            if saved["identity"]["config"]!=c or saved["identity"]["source"]!=source_identity(c):
                raise ValueError("Generation/checkpoint config or source mismatch")
            if saved["identity"].get("assets")!={k:weights_identity(c,k) for k in ("omni","ltx","gemma")}:
                raise ValueError("Pretrained assets changed since training")
            model=ListeningExtractor(cfg,c["events"]["head_width"]).to(device).eval().requires_grad_(False)
            model.load_state_dict(saved["model"])
            metadata.update(arm=saved["identity"]["arm"],train_seed=saved["identity"]["seed"],step=saved["step"],s0=saved["identity"]["s0"])
            del saved
            if mode=="audio_swap":
                from omni_av_p2.data import load_omni
                from omni_av_p2.connector import ConditionInputs
                runtime,processor=load_omni(cfg,device)
                data=extract_input(c,obs,runtime,processor)
                condition=ConditionInputs(**tree_to(data["condition"],device));layout=tree_to(data["layout"],device)
                del runtime,processor
                unload()
            else:
                try:
                    condition,layout=load_input(c,obs,device)
                except FileNotFoundError:
                    from omni_av_p2.data import load_omni
                    from omni_av_p2.connector import ConditionInputs
                    runtime,processor=load_omni(cfg,device)
                    data=extract_input(c,obs,runtime,processor)
                    condition=ConditionInputs(**tree_to(data["condition"],device));layout=tree_to(data["layout"],device)
                    del runtime,processor
                    unload()
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                result=model(condition,layout,event_mode="probe_only")
            atomic_torch(out/"probe.pt",cpu(dict(logits=result["logits"],layout=layout)))
            u=result["u"] if mode!="remove_u" else None
            del model,result,condition,layout
            unload()
        g=geometry(obs)
        av=reference_latents(cfg,make_observation(obs),g,device)
        text=load_text(c,device)
        renderer,strategy,sampler=load_renderer(cfg,device)
        torch.manual_seed(seed)
        prepared=strategy.prepare_training_inputs(with_text(av,text["positive"]),sampler)
        if u is not None: prepared=append_conditions(prepared,u)
        negative=append_negative(text["negative"],u)
        torch.manual_seed(seed)
        gen=c["generation"]
        sampled=guided_sample(renderer,prepared,negative,gen["steps"],gen["video_cfg"],gen["audio_cfg"])
        video,audio=unpack_latents(sampled,av)
        del renderer,prepared,sampled,av,u,text
        unload()
        decoded=decode_and_save(cfg,video,audio,g["fps"],out/"padded.mp4",device,raw_audio_path=out/"raw_audio.pt")
        # Decode full native geometry first; expose only the real input support.
        subprocess.run(["ffmpeg","-v","error","-i",str(out/"padded.mp4"),"-t",str(obs["requested_duration_s"]),
                        "-c:v","libx264","-crf","18","-c:a","aac",str(out/"generated.mp4")],check=True)
        import soundfile as sf
        raw=torch.load(out/"raw_audio.pt",map_location="cpu",weights_only=True)
        wave=raw["wave"];count=round(obs["requested_duration_s"]*raw["rate"])
        wave=torch.nn.functional.pad(wave,(0,max(0,count-wave.shape[-1])))[...,:count]
        sf.write(out/"listener.wav",wave.T.numpy(),raw["rate"],subtype="FLOAT")
        metadata.update(status="success",geometry=g,decoded=decoded,events_scored=False)
        atomic_json(out/"completed.json",metadata)
        return metadata
    except Exception as e:
        metadata.update(status="failed",error_type=type(e).__name__,error=str(e))
        atomic_json(out/"failed.json",metadata)
        raise
