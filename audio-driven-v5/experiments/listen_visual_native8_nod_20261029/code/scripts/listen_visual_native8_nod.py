#!/usr/bin/env python3
"""Confirmed visual annotation: native video8 + independently sourced nod candidates.

`vlm` starts GPU inference explicitly. `export` only reads existing model results.
CPU nod extraction continues to use run_glare_nodding_pilot.py; see the handoff.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
CONFIG=ROOT/'configs/listen_visual_native8_nod_v1.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def selected(manifest):
    data=json.loads(manifest.read_text())
    rows=data if isinstance(data,list) else [{'sample_id':sid} for sid in data['ids']]
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(Path(s).name!=s for s in ids):raise ValueError('Invalid or duplicate sample IDs')
    return rows

@contextmanager
def lock(out,name):
    path=out/'locks'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def export(args):
    from listen_benchmark.simple_timeline import validate_timeline
    from listen_benchmark.visual_native8 import observation_prompt
    cfg=json.loads(CONFIG.read_text());rows=selected(args.manifest);out=[]
    for row in rows:
        sid=row['sample_id'];vp=args.vlm_results/f'{sid}.json';np=args.nod_results/f'{sid}.json'
        raw=json.loads(vp.read_text());nod=json.loads(np.read_text())
        assert raw['sample_id']==sid and nod['id']==sid and nod['status']=='ok'
        duration=raw['duration_s'];assert abs(duration-nod['duration'])<1e-6
        records=[r for r in raw['records'] if r['arm']=='video8']
        assert records and len({r['view'] for r in records})==len(records)
        for r in records:
            assert r['prompt']==observation_prompt(r['end']-r['start']), 'Unexpected prompt; refuse silently mixing versions'
            assert r['native']['do_resize'] is False and r['native']['do_sample_frames'] is False
            assert r['media']['video']['requested_fps']==8 and r['native']['audio_input'] is False
        full=next(r for r in records if r['view']=='full')
        assert full['status']=='valid' and full['finish_reason']=='stop'
        validate_timeline(full['parsed'],duration)
        assert all(e['modality']=='visual' for e in full['parsed']['events'])
        # Pose metadata was validated against the cache crop during nod extraction.
        media=json.loads((Path(nod['video']).parent/'media.json').read_text())
        vm=full['media']['video']
        assert media['sample_id']==sid and media['video_transform']==vm['video_transform']
        assert abs(media['source_start_s']-vm['source_time_origin_s'])<=1/48000
        assert nod['fps']==cfg['nodding']['input_fps']
        # Reject accidental use of a different recall profile.
        nod_protocol=json.loads((args.nod_results.parent/'protocol.json').read_text())
        expected={k:v for k,v in cfg['nodding'].items() if k not in ('input_fps','profile')}
        assert nod_protocol['profiles']['recall']==expected
        events=[]
        for i,event in enumerate(nod['profiles']['recall']['events']):
            assert 0<=event['start']<event['end']<=duration
            events.append(dict(event,event_id=f'nod-{i+1:04d}',type='nodding',modality='visual',
                               description='Head nodding',description_source='detector_label_not_vlm_quote',
                               source='GLARE_nodding_v2_recall',status='candidate'))
        result={'schema':'listen-visual-native8-nod-v1','sample_id':sid,'duration_s':duration,
                'config_sha256':sha(CONFIG),'sources':{'vlm_file':str(vp.resolve()),'vlm_sha256':sha(vp),
                    'nod_file':str(np.resolve()),'nod_sha256':sha(np),'nod_protocol_sha256':sha(args.nod_results.parent/'protocol.json')},
                'vlm_observations':copy.deepcopy(records),'vlm_full_events':copy.deepcopy(full['parsed']['events']),
                'nod_candidates':events,'disabled_detector_types':list(cfg['supplement_disabled']),
                'status':'candidate_bundle_not_final_training_labels','training_admitted':False}
        save(args.output/'results'/f'{sid}.json',result);out.append(result)
    with (args.output/'bundles.jsonl').open('w') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'samples':len(out),'vlm_full_events':sum(len(r['vlm_full_events']) for r in out),
            'nod_candidates':sum(len(r['nod_candidates']) for r in out),'automatic_headshake_supplements':0,
            'invalid_windows_preserved':sum(x['status']!='valid' for r in out for x in r['vlm_observations']),
            'config_sha256':sha(CONFIG),'manifest_sha256':sha(args.manifest),'production_modified':False}
    save(args.output/'summary.json',report);print(json.dumps(report))

def vlm(args):
    from listen_benchmark.visual_native8 import prepare_media,prepare_input,MM_SIZE
    from listen_benchmark.full_raw import views
    from listen_benchmark.simple_timeline import response_schema,validate_timeline
    cfg=json.loads(CONFIG.read_text());rows=selected(args.manifest)
    if not 0<=args.rank<args.shards:raise ValueError('Invalid shard rank')
    if any('source' not in r for r in rows):raise ValueError('vlm needs full source rows, not an IDs-only manifest')
    run_cfg={'config_sha256':sha(CONFIG),'manifest_sha256':sha(args.manifest),'shards':args.shards,
             'runner_sha256':sha(Path(__file__)),'native_code_sha256':sha(ROOT/'src/listen_benchmark/visual_native8.py')}
    with lock(args.output,'protocol.lock'):
        p=args.output/'protocol.json'
        if p.exists():assert json.loads(p.read_text())==run_cfg,'Changed run inputs/config; choose a new output'
        else:save(p,run_cfg)
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    torch.set_num_threads(2)
    processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
    with lock(args.output,'model-init-'+socket.gethostname()):
        llm=LLM(model=cfg['model'],dtype='bfloat16',gpu_memory_utilization=args.memory,
                max_model_len=cfg['max_model_len'],max_num_seqs=cfg['max_num_seqs'],
                max_num_batched_tokens=cfg['max_num_batched_tokens'],enable_chunked_prefill=True,
                limit_mm_per_prompt={'image':0,'video':1},skip_mm_profiling=True,
                mm_processor_kwargs={'size':MM_SIZE},enforce_eager=True,seed=cfg['seed'])
    for row in rows[args.rank::args.shards]:
        sid=row['sample_id'];target=args.output/'results'/f'{sid}.json'
        with lock(args.output,'sample-'+sid):
            if target.exists():continue
            with lock(args.output,'decode-'+socket.gethostname()):whole=prepare_media(row['source'])
            partial=args.output/'partial'/f'{sid}.json'
            records=json.loads(partial.read_text())['records'] if partial.exists() else []
            completed={r['view'] for r in records}
            for view,start,end in views(whole[8]['duration_s']):
                if view in completed:continue
                entry,evidence=prepare_input(processor,whole,start,end,view)
                params=SamplingParams(temperature=cfg['temperature'],max_tokens=cfg['max_tokens'],
                    structured_outputs=StructuredOutputsParams(json=response_schema(end-start)))
                tick=time.monotonic();response=llm.generate([entry],params,use_tqdm=False)[0];answer=response.outputs[0]
                rec=dict(arm='video8',view=view,start=start,end=end,raw=answer.text,finish_reason=answer.finish_reason,
                         prompt_tokens=len(response.prompt_token_ids),completion_tokens=len(answer.token_ids),
                         elapsed_s=time.monotonic()-tick,**evidence)
                try:
                    rec['parsed']=json.loads(answer.text);validate_timeline(rec['parsed'],end-start)
                    assert all(e['modality']=='visual' for e in rec['parsed']['events']) and answer.finish_reason=='stop'
                    rec['status']='valid'
                except Exception as exc:rec.update(status='invalid',error=str(exc))
                records.append(rec);save(partial,{'sample_id':sid,'records':records})
            save(target,{'sample_id':sid,'duration_s':whole[8]['duration_s'],'records':records,'config_sha256':sha(CONFIG)})
            print(sid,'complete',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('vlm','export'):
        a=sub.add_parser(name);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='vlm':
            a.add_argument('--rank',type=int,default=0);a.add_argument('--shards',type=int,default=1);a.add_argument('--memory',type=float,default=.2)
        else:
            a.add_argument('--vlm-results',type=Path,required=True);a.add_argument('--nod-results',type=Path,required=True)
    args=p.parse_args();{'vlm':vlm,'export':export}[args.command](args)

if __name__=='__main__':main()
