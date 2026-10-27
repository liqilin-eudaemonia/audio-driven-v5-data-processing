"""Overlap media export with dynamically balanced, prefetched GPU cache jobs.

The frozen model functions, prompts, row contracts and output tensor formats are
unchanged. Each GPU keeps both encoders resident and decodes each window once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import config,resolve,read,write,digest,sha,source_contract


def sample_pixels(pixels,timestamps,duration,fps,max_pixels):
    import numpy as np
    from PIL import Image
    count=math.ceil(duration*fps);frames=[];next_index=0;last=None
    for pixels_i,timestamp in zip(pixels,timestamps):
        if next_index<count and timestamp+1e-6>=next_index/fps:
            image=Image.fromarray(pixels_i)
            scale=min(1.,math.sqrt(max_pixels/(image.width*image.height)))
            image=image.resize((max(28,round(image.width*scale)),max(28,round(image.height*scale))))
            last=np.asarray(image)
            while next_index<count and timestamp+1e-6>=next_index/fps:
                frames.append(last);next_index+=1
        if next_index==count:break
    if last is None:raise ValueError('No decoded frames')
    while len(frames)<count:frames.append(last)
    if len(frames)%2:frames.append(frames[-1])
    return np.stack(frames)


class Encoders:
    def __init__(self,cfg,root,device='cuda:0'):
        import torch
        from p3_listen import features
        from omni_av_p3.data import load_native
        from omni_av_p3.model import Config as PlannerConfig
        from omni_av_p2.config import Config as RendererConfig
        from ltx_trainer.model_loader import load_video_vae_encoder,load_audio_vae_encoder
        from ltx_core.model.audio_vae import AudioProcessor
        from omni_av import omni,data
        torch.set_num_threads(2)
        self.cfg,self.root,self.device=cfg,root,device
        self.runtime,self.processor=load_native(PlannerConfig(**cfg['planner']),device)
        c=RendererConfig(**cfg['renderer'])
        self.ve=load_video_vae_encoder(c.ltx_path,device=device,dtype=torch.bfloat16).requires_grad_(False).eval()
        self.ae=load_audio_vae_encoder(c.ltx_path,device=device,dtype=torch.float32).requires_grad_(False).eval()
        self.ap=AudioProcessor(target_sample_rate=self.ae.sample_rate,mel_bins=self.ae.mel_bins,
            mel_hop_length=self.ae.mel_hop_length,n_fft=self.ae.n_fft).to(device)
        self.native=features.native_contract(cfg);self.renderer=features.renderer_contract(cfg)
        self.original_frames=omni.sampled_frames;self.original_audio=data.read_audio
        self.current=None
        def cached_frames(path,duration,fps,max_pixels):
            item=self.current
            if item is not None and str(path)==str(item['paths']['target_video']):
                if (duration,fps,max_pixels)!=(item['row']['duration_s'],self.runtime.config.vision_fps,self.runtime.config.vision_max_pixels):
                    raise ValueError('Prefetched video parameters differ')
                return item['sampled']
            return self.original_frames(path,duration,fps,max_pixels)
        def cached_audio(path,end_s=None,start_s=0):
            item=self.current
            if item is not None and str(path) in item['audio16']:
                if start_s!=0 or end_s!=item['row']['duration_s']:raise ValueError('Prefetched audio parameters differ')
                return item['audio16'][str(path)]
            return self.original_audio(path,end_s,start_s)
        omni.sampled_frames=cached_frames;data.read_audio=cached_audio

    def prepare(self,row,force=False):
        import av
        import numpy as np
        import soundfile as sf
        from p3_listen import features
        p,_=features.verify_media(self.root,row)
        folder=features.cache_dir(self.root,row)
        expected_native=dict(row=digest(row),native=digest(self.native),media=sha(p['marker']))
        expected_vae=dict(row=digest(row),renderer=digest(self.renderer),media=sha(p['marker']))
        need_native=force or not features.validate_part(folder,'native',expected_native)
        need_vae=force or not features.validate_part(folder,'vae',expected_vae)
        item=dict(row=row,paths=p,folder=folder,expected_native=expected_native,expected_vae=expected_vae,
                  need_native=need_native,need_vae=need_vae,audio16={})
        if not (need_native or need_vae):return item
        with av.open(str(p['target_video'])) as video:
            video.streams.video[0].thread_count=2
            decoded=list(video.decode(video=0))
            timestamps=[float(f.time or 0) for f in decoded]
            pixels=np.stack([f.to_ndarray(format='rgb24') for f in decoded])
        if len(pixels)!=row['frames']:raise ValueError('Decoded frame count mismatch')
        item['pixels']=pixels
        if need_native:
            item['sampled']=sample_pixels(pixels,timestamps,row['duration_s'],self.runtime.config.vision_fps,self.runtime.config.vision_max_pixels)
            item['audio16']={str(p[k]):self.original_audio(p[k],end_s=row['duration_s']) for k in ('speaker_audio','target_audio')}
        if need_vae:
            item['wave'],item['rate']=sf.read(str(p['target_audio']),dtype='float32',always_2d=True)
        return item

    def compute(self,item,save=True):
        import torch
        from p3_listen import features
        from ltx_core.types import Audio
        self.current=item;row=item['row'];folder=item['folder'];result={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if item['need_native']:
                layout,targets=features.extract_native(self.runtime,self.processor,features.observation(self.root,row),item['paths'])
                result.update(layout=layout,targets=targets)
                if save:
                    features.save(folder/'layout.pt',layout);features.save(folder/'targets.pt',targets)
                    features.seal(folder,'native',item['expected_native'],['layout.pt','targets.pt'])
        with torch.inference_mode():
            if item['need_vae']:
                value=torch.from_numpy(item['pixels']).permute(3,0,1,2)[None].to(self.device,torch.bfloat16)/127.5-1
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=self.ve(value);first=self.ve(value[:,:,:1])
                z[:,:,:1]=first
                waveform=torch.from_numpy(item['wave'].T.copy())[None].to(self.device)
                if waveform.shape[1]==1:waveform=waveform.repeat(1,2,1)
                az=self.ae(self.ap.waveform_to_mel(Audio(waveform=waveform,sampling_rate=item['rate'])).float())
                scalar=lambda x:torch.tensor([x])
                av=dict(latents=dict(latents=z.cpu(),num_frames=scalar(z.shape[2]),height=scalar(z.shape[3]),
                     width=scalar(z.shape[4]),fps=scalar(row['fps'])),
                     audio_latents=dict(latents=az.cpu().to(torch.bfloat16),num_time_steps=scalar(az.shape[2]),duration=scalar(row['duration_s'])))
                if not torch.isfinite(z).all() or not torch.isfinite(az).all():raise ValueError('Nonfinite VAE cache')
                result.update(av=av,reference=first.cpu())
                if save:
                    features.save(folder/'av.pt',av);features.save(folder/'reference.pt',result['reference'])
                    features.seal(folder,'vae',item['expected_vae'],['av.pt','reference.pt'])
        self.current=None
        return result


def prefetched(pool,function,rows,depth=3):
    source=iter(rows);pending=[]
    for _ in range(depth):
        row=next(source,None)
        if row is not None:pending.append(pool.submit(function,row))
    while pending:
        value=pending.pop(0).result()
        yield value
        row=next(source,None)
        if row is not None:pending.append(pool.submit(function,row))


@contextmanager
def job_lock(path):
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False;return
        yield True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/p3_listen_300h.json')
    p.add_argument('--job-size',type=int,default=32)
    p.add_argument('--limit',type=int,default=0,help='Engineering only; cannot seal production')
    p.add_argument('--benchmark-batch',action='store_true',help='Rank zero measures worst-case VAE batches once, then resumes caching')
    a=p.parse_args();cfg=config(a.config);root=resolve(cfg['output'])
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    visible=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8)))).split(',')
    os.environ['CUDA_VISIBLE_DEVICES']=visible[local]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
    from p3_listen import features
    class FixedPromptWarning(logging.Filter):
        def filter(self,record):
            return not record.getMessage().startswith('System prompt modified, audio output may not work as expected.')
    logging.getLogger('transformers.models.qwen2_5_omni.processing_qwen2_5_omni').addFilter(FixedPromptWarning())
    manifest=read(root/'windows.json');rows=[r for r in manifest['windows'] if r['split'] in ('train','dev')]
    if a.limit:rows=rows[:a.limit]
    contract=dict(manifest_sha256=sha(root/'windows.json'),config_sha256=digest(cfg),source=source_contract(cfg),
                  implementation_sha256=sha(__file__),job_size=a.job_size,limit=a.limit)
    queue=root/'stream_jobs'/digest(contract)[:20];queue.mkdir(parents=True,exist_ok=True)
    jobs=[rows[i:i+a.job_size] for i in range(0,len(rows),a.job_size)]
    pending=list(range(rank,len(jobs),world))+[i for i in range(len(jobs)) if i%world!=rank]
    enc=Encoders(cfg,root)
    if a.benchmark_batch and rank==0:
        from benchmark_vae_batches import benchmark
        benchmark(enc,rows,root/'cluster_checks/vae_batch_benchmark.json')
    begin=time.monotonic();completed=0;visited=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending:
            remaining=[];made_progress=False
            for i in pending:
                marker=queue/f'{i:06d}.json';batch=jobs[i]
                if marker.exists():
                    if read(marker)['sample_ids_sha256']!=digest([r['sample_id'] for r in batch]):raise ValueError('Cache job identity changed')
                    continue
                # Jobs may be consumed while other media windows are still being exported.
                if not all(features.paths(root,r)['marker'].exists() for r in batch):
                    remaining.append(i);continue
                with job_lock(queue/f'{i:06d}.lock') as acquired:
                    if not acquired:
                        remaining.append(i);continue
                    if marker.exists():continue
                    for item in prefetched(pool,enc.prepare,batch):
                        enc.compute(item);completed+=1
                    write(marker,dict(sample_ids_sha256=digest([r['sample_id'] for r in batch]),count=len(batch),rank=rank))
                    made_progress=True
                    report=dict(rank=rank,completed=completed,seconds=time.monotonic()-begin,job=i,limit=a.limit,
                                total_windows=len(rows),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                    write(root/f'stream_progress_{rank:02d}.json',report);print(json.dumps(report),flush=True)
            pending=remaining
            if pending and not made_progress:time.sleep(5)
    # Dynamic processing is followed by exactly the original deterministic full validation shards.
    selected=rows[rank::world];h=hashlib.sha256()
    for row in selected:
        features.verify_media(root,row)
        features.load_native(root,row,cfg,targets=True);features.load_av(root,row,cfg,'cpu')
        folder=features.cache_dir(root,row)
        h.update(digest([row['sample_id'],sha(folder/'native.json'),sha(folder/'vae.json')]).encode())
    write(root/'cache_shards'/f'rank_{rank:02d}.json',dict(rank=rank,world_size=world,
        manifest_sha256=sha(root/'windows.json'),config_sha256=digest(cfg),count=len(selected),
        sample_ids_sha256=digest([r['sample_id'] for r in selected]),marker_digest=h.hexdigest(),
        verified_file_hashes=True,source=source_contract(cfg),limit=a.limit,stream_contract=contract))


if __name__=='__main__':main()
