"""Batched frozen teacher extraction and strict, separately versioned storage."""
from pathlib import Path
import math

import torch

from p2_listen.contracts import ROOT, read, sha, digest, atomic_json
from p2_listen.cache import cpu, save_part, load_part


def code_identity():
    return {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def open_source(source):
    from p2_j_cache.runtime import CacheReader
    reader = CacheReader(source)
    complete = read(Path(source)/'COMPLETE.json')
    if not complete['success'] or complete['samples'] != len(reader.ids('train')):
        raise ValueError('Source cache not completely accepted')
    return reader


def prepare_work(source, work, ids=None):
    reader = open_source(source); m = reader.manifest
    from p2_listen.config import extractor_config
    from p2_plan_cache.runtime import dependencies
    cfg = extractor_config(m['config'])
    chosen = reader.ids('train') if ids is None else ids
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(reader.ids('train')):
        raise ValueError('Duplicate or out-of-train teacher sample')
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema='p2-j-feature-teacher-v1', source=str(Path(source).resolve()),
        source_manifest_sha256=sha(Path(source)/'manifest.json'), source_complete_sha256=sha(Path(source)/'COMPLETE.json'),
        sample_ids=chosen, ids_sha256=digest(chosen), samples=len(chosen), model=cfg.omni_path,
        dependencies=dependencies(m['config'],'input_plan'), code=code_identity(), target_dim=cfg.target_dim,
        video_policy='final_512_crop_then_nearest_25fps_training_frame_at_8fps_then_teacher_resize',
        audio_policy='frozen_listener_official_union_recipe_then_mono_resample16000_round_grid',
        mask_policy='full_temporal_support_padding_only_silence_valid', video_execution='per_clip_exact_native', audio_execution='per_clip_exact_native', teacher_frozen=True,
        no_lora=True, no_intermediate_wav_or_video=True)
    atomic_json(work/'protocol.json',protocol,immutable=True)
    jobs = [chosen[i:i+128] for i in range(0,len(chosen),128)]
    atomic_json(work/'jobs.json',jobs,immutable=True)
    return protocol


class Reader:
    def __init__(self, work):
        from p2_plan_cache.runtime import dependencies
        self.work = Path(work); self.protocol = read(self.work/'protocol.json')
        p = self.protocol; self.source = open_source(p['source'])
        if (p['schema'] != 'p2-j-feature-teacher-v1' or p['code'] != code_identity()
                or p['source_manifest_sha256'] != sha(Path(p['source'])/'manifest.json')
                or p['source_complete_sha256'] != sha(Path(p['source'])/'COMPLETE.json')
                or p['dependencies'] != dependencies(self.source.manifest['config'],'input_plan')):
            raise ValueError('Teacher source/code/assets changed')
        self.protocol_sha = sha(self.work/'protocol.json'); self.ids = set(p['sample_ids'])

    def item(self, sid):
        from p2_j_cache.data import sample
        if sid not in self.ids: raise ValueError('Sample outside teacher manifest')
        return sample(self.source.work,self.source.manifest,sid)

    def identity(self, item):
        sid = item['sample_id']
        return dict(schema=self.protocol['schema'], protocol_sha256=self.protocol_sha,
                    sample_id=sid, source_digest=self.source.manifest['entries'][sid],
                    source=item['source'], audio_recipe=item['audio_recipe'],
                    video_sampling_override=self.source.manifest.get('video_sampling_overrides',{}).get(sid))

    def load(self, sid, device='cpu', missing_ok=False):
        return load_part(self.work/'cache'/sid, self.identity(self.item(sid)), device, missing_ok)

    def training_sample(self, sid, variant='guided', device='cpu'):
        value = self.source.training_sample(sid,variant,device)
        teacher = self.load(sid,device)
        validate_geometry(teacher,value['layout'],self.protocol['target_dim'])
        value['teacher'] = {k:teacher[k] for k in ('video','audio','video_valid','audio_valid')}
        return value


def validate_geometry(value, layout, dim):
    for mod in ('video','audio'):
        if value[mod].shape != (*layout[mod+'_valid'].shape,dim) or not torch.isfinite(value[mod]).all():
            raise ValueError('Teacher/query feature mismatch: '+mod)
        mask = value[mod+'_valid']
        if mask.dtype != torch.bool or mask.shape != layout[mod+'_valid'].shape:
            raise ValueError('Teacher mask mismatch')
        if (mask & ~layout[mod+'_target_valid'].to(mask.device)).any():
            raise ValueError('Teacher covers padded tail')


def load_model(reader):
    from p2_listen.config import extractor_config
    from omni_av_p2.data import load_omni
    cfg = extractor_config(reader.source.manifest['config'])
    runtime, processor = load_omni(cfg,'cuda:0')
    if any(p.requires_grad for p in runtime.thinker.parameters()):
        raise ValueError('Teacher must be frozen')
    patch = processor.image_processor.patch_size * processor.image_processor.merge_size
    side = max(patch,round(min(512,math.sqrt(cfg.vision_max_pixels))/patch)*patch)
    return runtime,processor,(side,side),(side//patch,side//patch)


@torch.inference_mode()
def extract_batch(runtime, prepared):
    from p2_listen.layout import aggregate_audio
    model = runtime.thinker; device = next(model.parameters()).device
    # Native video batching changed teacher values beyond the pilot tolerance.
    # Keep both towers' forwards independent; scheduling and writes overlap.
    videos = [model.get_video_features(p['video']['pixel_values_videos'].to(device),
                                      p['video']['video_grid_thw'].to(device)) for p in prepared]
    audios = [model.get_audio_features(p['audio']['input_features'].to(device,torch.bfloat16),
                                      p['audio']['attention_mask'].to(device)) for p in prepared]
    result=[]
    for p,v,a in zip(prepared,videos,audios):
        layout = p['layout']; a,valid = aggregate_audio(a[None],layout['audio_intervals'])
        value = dict(video=v[None],audio=a,video_valid=layout['video_target_valid'].to(device),
                     audio_valid=layout['audio_target_valid'].to(device)&valid,audit=p['audit'])
        validate_geometry(value,layout,runtime.config.target_dim)
        result.append(cpu(value))
    return result


def persist(reader,item,value):
    sid=item['sample_id']; identity=reader.identity(item); folder=reader.work/'cache'/sid
    save_part(folder,identity,value)
    actual=load_part(folder,identity)
    for mod in ('video','audio','video_valid','audio_valid'):
        if not torch.equal(actual[mod],value[mod]):raise ValueError('Teacher tensor write mismatch')
    if actual['audit']!=value['audit']:raise ValueError('Teacher media audit write mismatch')
    return dict(sample_id=sid,bytes=(folder/'payload.pt').stat().st_size,
                payload_sha256=sha(folder/'payload.pt'),identity=digest(identity))
