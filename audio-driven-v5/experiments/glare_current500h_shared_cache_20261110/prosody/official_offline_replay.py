"""Replay authors' offline functions on exactly the7 flagged audio inputs."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import soundfile as sf
import torch
import prepare_glare_prosody_cache as cache
from glare_prosody_audit import parse_reply


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>30*1024**3
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    path=ROOT/'related/visual_reaction_annotation/prosody_extract/batch_emotion_fluctuation_multi_gpu.py'
    spec=importlib.util.spec_from_file_location('official_glare_offline_prosody',path)
    vendor=importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    manifest=cache.read(ROOT/'data_previews/listen_benchmark_review/glare_prosody_null_review_20261110/data.json')
    chunks=[];metadata=[];pairs=[]
    for item in manifest['items']:
        wave,rate=sf.read(item['audio_file'],dtype='float32');assert rate==16000
        assert hashlib.sha256(wave.tobytes()).hexdigest()==item['waveform_sha256']
        normalized=vendor.normalize_waveform_loudness(wave,target_dbfs=-20.,peak_limit=.98)
        parts=vendor.split_chunks(normalized,16000,2.4)
        pairs.append(dict(sample_id=item['sample_id'],duration_s=len(wave)/16000,
            full_chunks=len(parts),dropped_tail_s=(len(wave)-sum(len(p) for p in parts))/16000,
            official_skips_pair=not parts))
        for i,p in enumerate(parts):
            chunks.append(p);metadata.append(dict(sample_id=item['sample_id'],start_s=i*2.4,end_s=(i+1)*2.4,
                input_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
    prompt=vendor.read_prompt(path.parent/'input_prompt.txt')
    analyzer=vendor.Qwen2AudioAnalyzer(cache.MODEL,prompt,64,'cuda:0')
    adaptations=[]
    if 'audios' in inspect.signature(analyzer.processor.__call__).parameters:
        class KeywordCompatibility:
            def __init__(self,original):self.original=original
            def __getattr__(self,name):return getattr(self.original,name)
            def __call__(self,**kwargs):
                if 'audio' in kwargs:kwargs['audios']=kwargs.pop('audio')
                return self.original(**kwargs)
        analyzer.processor=KeywordCompatibility(analyzer.processor)
        adaptations.append('Installed Transformers expects audios; API keyword only: audio -> audios')
    raw_rows=[];original=vendor.normalize_result
    generation={};original_generate=analyzer.model.generate
    def generate(*args,**kwargs):
        assert kwargs.get('input_features') is not None and kwargs.get('feature_attention_mask') is not None
        output=original_generate(*args,**kwargs)
        new=output[:,kwargs['input_ids'].shape[1]:].tolist()
        eos=analyzer.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        generation['rows']=[dict(tokens=len(r),truncated=len(r)>=64 and r[-1] not in eos) for r in new]
        generation['input_features_shape']=list(kwargs['input_features'].shape)
        generation['valid_mel_frames']=kwargs['feature_attention_mask'].sum(-1).tolist()
        return output
    analyzer.model.generate=generate
    def normalize(text,max_time):
        official=original(text,max_time)
        index=len(raw_rows)
        raw_rows.append(dict(**metadata[index],raw_text=text,official_normalized=official,
            audit=parse_reply(text,max_time,truncated=generation['rows'][index]['truncated']),
            generation=generation['rows'][index]))
        return official
    vendor.normalize_result=normalize
    started=time.time();outputs=analyzer.infer_chunks(chunks,16000,2.4)
    result=dict(complete=True,at=time.time(),elapsed_s=time.time()-started,official_source=str(path),
        source_sha256=cache.sha(path),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        actual_official_functions=['normalize_waveform_loudness','split_chunks','Qwen2AudioAnalyzer.infer_chunks','normalize_result'],
        configuration=dict(chunk_seconds=2.4,max_new_tokens=64,normalize_loudness=True,target_dbfs=-20.,
            peak_limit=.98,batch_size_cap=16,actual_batch_size=len(chunks),low_rms_gate=False,gpu=7),
        compatibility_adaptations=adaptations,raw_capture_does_not_change_generation=True,
        pairs=pairs,chunks=raw_rows,generation=generation,training_cache_modified=False,
        caution='Official split_chunks keeps full2.4s chunks only. Short pairs/tails are skipped here for faithful diagnostic replay, not approved as our production policy.')
    cache.save(cache.WORK/'OFFICIAL_OFFLINE_REPLAY.json',result)
    print(json.dumps(dict(chunks=len(chunks),official_null=sum(x=='NULL' for x in outputs),
        audit_accepted=sum(r['audit']['accepted'] for r in raw_rows),skipped_pairs=sum(p['official_skips_pair'] for p in pairs))),flush=True)


if __name__=='__main__':main()
