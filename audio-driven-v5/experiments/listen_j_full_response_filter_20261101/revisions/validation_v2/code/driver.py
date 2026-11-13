"""Resumable caption-only Qwen judgment of full train/dev reference manifests."""
import argparse
import collections
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
SOURCE=ROOT/'experiments/listen_j_full_response_sft_20261101'
DEFAULT=ROOT/'experiments/listen_j_full_response_filter_20261101'
MODEL='/private/ckpts/Qwen3-30B-A3B-Instruct-2507'
GPU_UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
SPLITS=('dev','train')
BLOCK=2048


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def static_visual_evidence(caption):
    # Frozen from the previous 100-caption judgment pilot.
    evidence=[]
    for sentence in re.split(r'(?<=[.!?])\s+',caption):
        if re.search(r'\b(background|backdrop)\b',sentence,re.I) and not re.search(
                r'\b(audio|hum|sound|sounds|noise|music|vocalizations)\b',sentence,re.I):
            evidence.append(dict(rule='visual_background_sentence',quote=sentence))
        if re.search(r'\bhair\b[^.!?]*\b(frames|framing|styled|pulled back)\b',sentence,re.I):
            evidence.append(dict(rule='static_hair_description',quote=sentence))
    return evidence


def schema():
    entry=dict(type='object',properties=dict(category=dict(type='string',enum=['A','B','C','D']),
        quote=dict(type='string'),reason=dict(type='string')),
        required=['category','quote','reason'],additionalProperties=False)
    return dict(type='object',properties=dict(keep=dict(type='boolean'),
        defects=dict(type='array',maxItems=1,items=entry)),required=['keep','defects'],additionalProperties=False)


def parse_answer(raw,finish,caption):
    if finish!='stop':
        raise ValueError('generation did not end normally: '+str(finish))
    obj=json.loads(raw)
    if type(obj.get('keep')) is not bool or not isinstance(obj.get('defects'),list):
        raise ValueError('invalid decision types')
    if obj['keep']!=(len(obj['defects'])==0) or len(obj['defects'])>1:
        raise ValueError('decision/defect mismatch')
    for d in obj['defects']:
        if d.get('category') not in ('A','B','C','D') or not d.get('quote') or d['quote'] not in caption or not d.get('reason'):
            raise ValueError('invalid category or evidence is not an exact source substring')
    return obj


def prepare(work):
    work.mkdir(exist_ok=False)
    summary=read(SOURCE/'summary.json')
    complete=read(SOURCE/'COMPLETE.json')
    assert complete['status']=='complete' and complete['summary_sha256']==sha(SOURCE/'summary.json')
    assert summary['scope']=='full'
    sources={}
    for split in SPLITS:
        info=summary['manifests'][split]
        path=SOURCE/info['file']
        assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256']
        sources[split]=dict(path=str(path),**info,count=summary['splits'][split]['clips'])
    prompt=(ROOT/'experiments/p2_planning_caption_clean100_20261101/v3_judge/prompt.txt').read_text()
    prompt+='\nFor efficient batch screening, if rejecting report exactly ONE decisive defect. Use a short exact source quote and a brief reason (prefer at most 15 words). Do not enumerate all defects. The keep/reject criteria above are unchanged.\n'
    (work/'system.txt').write_text(prompt)
    (work/'code').mkdir()
    shutil.copy2(__file__,work/'code/driver.py')
    cfg=dict(schema='full-response-caption-judge-v1',source=str(SOURCE),sources=sources,splits=list(SPLITS),
        total=sum(x['count'] for x in sources.values()),model=MODEL,physical_gpu=7,gpu_uuid=GPU_UUID,
        block_size=BLOCK,max_num_seqs=1024,max_num_batched_tokens=32768,max_model_len=8192,
        gpu_memory_utilization=.92,max_tokens=512,retry_max_tokens=1024,temperature=0,
        source_summary_sha256=sha(SOURCE/'summary.json'),system_sha256=sha(work/'system.txt'),
        code_sha256=sha(work/'code/driver.py'),model_config_sha256=sha(Path(MODEL)/'config.json'),
        input='target.text only; no audio/image decoding',
        outputs='model judgment plus frozen static-visual veto; original reference lines copied without editing',
        started_at=time.time())
    assert cfg['total']==1095039
    save(work/'config.json',cfg)
    save(work/'status.json',dict(stage='prepared',processed=0,total=cfg['total'],at=time.time()))
    print(json.dumps(dict(work=str(work),total=cfg['total'],splits={k:v['count'] for k,v in sources.items()})),flush=True)


def completed(work,cfg):
    all_meta=[]
    starts={}
    for split in SPLITS:
        metas=sorted((work/'batches'/split).glob('*.meta.json'))
        offset=0
        count=0
        for index,path in enumerate(metas):
            m=read(path)
            assert m['split']==split and m['block']==index and m['start_offset']==offset
            assert (work/m['result_file']).is_file()
            count+=m['count']
            offset=m['end_offset']
            all_meta.append(m)
        assert count<=cfg['sources'][split]['count']
        starts[split]=dict(block=len(metas),offset=offset,count=count)
    return all_meta,starts


def aggregate(metas,cfg):
    states=collections.Counter()
    per_split={s:collections.Counter() for s in SPLITS}
    for m in metas:
        states.update(m['counts'])
        per_split[m['split']].update(m['counts'])
    processed=sum(states.values())
    return dict(processed=processed,total=cfg['total'],counts=dict(states),
        splits={s:dict(v) for s,v in per_split.items()})


def worker(work,max_blocks=0):
    cfg=read(work/'config.json')
    assert sha(__file__)==cfg['code_sha256'] and sha(work/'system.txt')==cfg['system_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    query=['nvidia-smi','-i','7','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    info=[s.strip() for s in subprocess.check_output(query,text=True).split(',')]
    if info[0]!=cfg['gpu_uuid'] or int(info[1])>1024:
        raise RuntimeError('Assigned GPU7 is not free; do not overlap another task')
    lock=(work/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metas,starts=completed(work,cfg)
    if sum(m['count'] for m in metas)==cfg['total']:
        return
    for split,source in cfg['sources'].items():
        assert Path(source['path']).stat().st_size==source['bytes']
    prompt=(work/'system.txt').read_text()
    (work/'runs').mkdir(exist_ok=True)
    run_id=f'{int(time.time())}-{os.getpid()}'
    stop=threading.Event()
    stage=['loading']
    def monitor():
        with (work/'runs'/f'{run_id}.resources.jsonl').open('w',buffering=1) as f:
            while not stop.is_set():
                try:
                    info=[x.strip() for x in subprocess.check_output(query,text=True,timeout=5).split(',')]
                    r=dict(at=time.time(),stage=stage[0],memory_mib=int(info[1]),utilization=int(info[2]))
                except Exception as e:
                    r=dict(at=time.time(),stage=stage[0],error=str(e))
                f.write(json.dumps(r)+'\n')
                save(work/'telemetry.json',r)
                stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    save(work/'status.json',dict(stage='loading',run_id=run_id,at=time.time(),**aggregate(metas,cfg)))
    begin=time.monotonic()
    try:
        from importlib.metadata import version
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
        graph_sizes=[1,2,4,8,16,32,64,128,256,512,1024]
        llm=LLM(model=cfg['model'],dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=cfg['gpu_memory_utilization'],max_model_len=cfg['max_model_len'],
            max_num_seqs=cfg['max_num_seqs'],max_num_batched_tokens=cfg['max_num_batched_tokens'],
            enable_prefix_caching=True,enable_chunked_prefill=True,seed=20261101,
            compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=graph_sizes,
                max_cudagraph_capture_size=graph_sizes[-1]),
            structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
        load_seconds=time.monotonic()-begin
        save(work/'runs'/f'{run_id}.json',dict(run_id=run_id,host=socket.gethostname(),pid=os.getpid(),
            load_seconds=load_seconds,vllm=version('vllm'),torch=version('torch'),config_sha256=sha(work/'config.json')))
        params=SamplingParams(temperature=0,max_tokens=cfg['max_tokens'],
            structured_outputs=StructuredOutputsParams(json=schema()))
        retry_params=SamplingParams(temperature=0,max_tokens=cfg['retry_max_tokens'],
            structured_outputs=StructuredOutputsParams(json=schema()))

        def batches():
            for split in SPLITS:
                source=cfg['sources'][split]
                with open(source['path'],'rb') as f:
                    f.seek(starts[split]['offset'])
                    index=starts[split]['block']
                    while True:
                        rows=[]
                        start=f.tell()
                        for _ in range(cfg['block_size']):
                            offset=f.tell()
                            raw=f.readline()
                            if not raw:
                                break
                            r=json.loads(raw)
                            assert r['split']==split and r['target']['format']=='original_caption_verbatim'
                            text=r['target']['text']
                            assert isinstance(text,str) and text.strip()
                            rows.append(dict(sample_id=r['sample_id'],split=split,duration_s=r['duration_s'],
                                in_current_500h_trial=r['provenance']['in_current_500h_trial'],
                                listener_waveform_nonzero=r['provenance']['processed_listener_waveform_nonzero'],
                                source_offset=offset,source_bytes=len(raw),source_line_sha256=hashlib.sha256(raw).hexdigest(),
                                caption_sha256=hashlib.sha256(text.encode()).hexdigest(),caption=text,
                                rule_evidence=static_visual_evidence(text)))
                        if not rows:
                            break
                        prompts=[tokenizer.apply_chat_template([dict(role='system',content=prompt),
                            dict(role='user',content=json.dumps(dict(caption=r['caption']),ensure_ascii=False))],
                            tokenize=False,add_generation_prompt=True) for r in rows]
                        tokens=tokenizer(prompts,add_special_tokens=False)['input_ids']
                        yield dict(split=split,block=index,start_offset=start,end_offset=f.tell(),rows=rows,tokens=tokens)
                        index+=1

        iterator=batches()
        def get_next():
            return next(iterator,None)
        processed_run=0
        model_seconds=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(get_next)
            while True:
                batch=future.result()
                if batch is None:
                    break
                if (work/'STOP').exists():
                    save(work/'status.json',dict(stage='stopped_at_batch_boundary',at=time.time(),**aggregate(metas,cfg)))
                    return
                future=pool.submit(get_next)
                stage[0]='generation'
                t=time.monotonic()
                results=[dict(**{k:v for k,v in r.items() if k!='caption'},attempts=[]) for r in batch['rows']]
                indices=[i for i,tokens in enumerate(batch['tokens']) if len(tokens)+cfg['max_tokens']<=cfg['max_model_len']]
                for i in set(range(len(results)))-set(indices):
                    results[i]['error']='context_overflow_no_truncation'
                generated=llm.generate([dict(prompt_token_ids=batch['tokens'][i]) for i in indices],params,use_tqdm=False)
                retry=[]
                for i,response in zip(indices,generated):
                    ans=response.outputs[0]
                    results[i]['attempts'].append(dict(raw=ans.text,finish_reason=ans.finish_reason,tokens=len(ans.token_ids)))
                    try:
                        results[i]['model_judgment']=parse_answer(ans.text,ans.finish_reason,batch['rows'][i]['caption'])
                    except Exception as e:
                        results[i]['error']=str(e)
                        retry.append(i)
                if retry:
                    stage[0]='retry'
                    retry_prompts=[]
                    retry_indices=[]
                    for i in retry:
                        messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(dict(caption=batch['rows'][i]['caption']),ensure_ascii=False)),
                            dict(role='user',content='The previous response failed structural validation: '+results[i]['error']+'. Judge the original caption again. Copy a short exact source quote; use one decisive defect. Do not rewrite the caption.')]
                        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                        if len(ids)+cfg['retry_max_tokens']<=cfg['max_model_len']:
                            retry_prompts.append(dict(prompt_token_ids=ids))
                            retry_indices.append(i)
                    retried=llm.generate(retry_prompts,retry_params,use_tqdm=False) if retry_prompts else []
                    for i,response in zip(retry_indices,retried):
                        ans=response.outputs[0]
                        results[i]['attempts'].append(dict(raw=ans.text,finish_reason=ans.finish_reason,tokens=len(ans.token_ids)))
                        try:
                            results[i]['model_judgment']=parse_answer(ans.text,ans.finish_reason,batch['rows'][i]['caption'])
                            results[i].pop('error',None)
                        except Exception as e:
                            results[i]['error']=str(e)
                elapsed=time.monotonic()-t
                model_seconds+=elapsed
                for r in results:
                    j=r.get('model_judgment')
                    r['decision']='needs_review' if j is None else ('keep' if j['keep'] and not r['rule_evidence'] else 'reject')
                stage[0]='committing'
                directory=work/'batches'/batch['split']
                directory.mkdir(parents=True,exist_ok=True)
                name=f"{batch['block']:06d}"
                output=directory/(name+'.jsonl')
                temp=directory/(name+f'.{os.getpid()}.tmp')
                h=hashlib.sha256()
                with temp.open('wb') as f:
                    for r in results:
                        data=(json.dumps(r,ensure_ascii=False)+'\n').encode()
                        f.write(data)
                        h.update(data)
                temp.replace(output)
                metric=dict(split=batch['split'],block=batch['block'],start_offset=batch['start_offset'],end_offset=batch['end_offset'],
                    count=len(results),counts=dict(collections.Counter(r['decision'] for r in results)),
                    result_file=str(output.relative_to(work)),result_sha256=h.hexdigest(),
                    run_id=run_id,seconds=elapsed,clips_per_second=len(results)/elapsed,
                    output_tokens=sum(a['tokens'] for r in results for a in r['attempts']),
                    retried=len(retry),at=time.time())
                save(directory/(name+'.meta.json'),metric)
                metas.append(metric)
                processed_run+=len(results)
                state=aggregate(metas,cfg)
                state.update(stage='running',at=time.time(),run_id=run_id,last_batch=metric,
                    run_samples=processed_run,run_model_seconds=model_seconds,
                    recent_clips_per_second=sum(m['count'] for m in metas[-10:])/sum(m['seconds'] for m in metas[-10:]),
                    elapsed_seconds=time.monotonic()-begin)
                save(work/'status.json',state)
                print(json.dumps(metric),flush=True)
                if max_blocks and processed_run>=max_blocks*cfg['block_size']:
                    state['stage']='pilot_complete'
                    save(work/'status.json',state)
                    return
        state=aggregate(metas,cfg)
        assert state['processed']==cfg['total']
        save(work/'status.json',dict(**state,stage='judgment_complete_pending_export',at=time.time()))
    finally:
        stop.set()
        thread.join(timeout=6)
        lock.close()


def finalize(work):
    cfg=read(work/'config.json')
    metas,_=completed(work,cfg)
    assert sum(m['count'] for m in metas)==cfg['total']
    destination=work/'filtered'
    if (work/'COMPLETE.json').exists():
        assert destination.is_dir()
        return
    temp=work/'filtered.partial'
    temp.mkdir(exist_ok=True)
    for folder in ['rejected','needs_review']:
        (temp/folder).mkdir(exist_ok=True)
    dbpath=temp/'index.sqlite'
    if dbpath.exists():
        dbpath.unlink()
    db=sqlite3.connect(dbpath)
    db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
    seen=set()
    totals={}
    groups=collections.defaultdict(lambda:dict(clips=0,hours=0.0))
    save(work/'status.json',dict(stage='exporting_and_verifying',at=time.time(),**aggregate(metas,cfg)))
    for split in SPLITS:
        source=cfg['sources'][split]
        source_hash=hashlib.sha256()
        counts=collections.Counter()
        hours=collections.Counter()
        with contextlib.ExitStack() as stack:
            src=stack.enter_context(open(source['path'],'rb'))
            outputs={name:stack.enter_context((temp/path).open('wb')) for name,path in
                [('keep',split+'.jsonl'),('reject','rejected/'+split+'.jsonl'),('needs_review','needs_review/'+split+'.jsonl')]}
            for meta in [m for m in metas if m['split']==split]:
                result_path=work/meta['result_file']
                assert sha(result_path)==meta['result_sha256']
                insert=[]
                with result_path.open() as f:
                    for text in f:
                        r=json.loads(text)
                        assert src.tell()==r['source_offset']
                        raw=src.readline()
                        source_hash.update(raw)
                        assert len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_line_sha256']
                        original=json.loads(raw)
                        sid=original['sample_id']
                        assert sid==r['sample_id'] and original['split']==r['split']==split and sid not in seen
                        assert hashlib.sha256(original['target']['text'].encode()).hexdigest()==r['caption_sha256']
                        seen.add(sid)
                        decision=r['decision']
                        offset=outputs[decision].tell()
                        outputs[decision].write(raw)
                        counts[decision]+=1
                        hours[decision]+=original['duration_s']/3600
                        for group in ['all','listener_nonzero' if r['listener_waveform_nonzero'] else 'listener_zero',
                                'in_500h' if r['in_current_500h_trial'] else 'outside_500h']:
                            g=groups[f'{split}:{decision}:{group}']
                            g['clips']+=1
                            g['hours']+=original['duration_s']/3600
                        if decision=='keep':
                            audio=original['inputs']['speaker_audio']
                            frame=original['inputs']['listener_first_frame']
                            insert.append((sid,split,original['pair'],frame['participant_id'],original['group_id'],
                                audio['start_s'],audio['end_s'],original['duration_s'],original['face']['short_side_p10_px'],
                                split+'.jsonl',offset,len(raw)))
                db.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',insert)
                db.commit()
            assert not src.read(1) and source_hash.hexdigest()==source['sha256']
        assert sum(counts.values())==source['count']
        totals[split]=dict(counts=dict(counts),hours=dict(hours))
    assert len(seen)==cfg['total']
    assert db.execute('SELECT count(*) FROM samples').fetchone()[0]==sum(v['counts'].get('keep',0) for v in totals.values())
    db.close()
    shutil.copy2(Path(cfg['source'])/'crop_config.json',temp/'crop_config.json')
    manifest={}
    for path in sorted(temp.rglob('*.jsonl')):
        manifest[str(path.relative_to(temp))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    summary=dict(schema=cfg['schema'],status='all_train_dev_judged_and_exported',total=cfg['total'],splits=totals,
        groups=dict(groups),manifests=manifest,source=cfg['source'],test_processed=False,
        original_rows_copied_verbatim=True,model_judgment_not_human_ground_truth=True,
        unresolved=sum(v['counts'].get('needs_review',0) for v in totals.values()),completed_at=time.time())
    save(temp/'summary.json',summary)
    if destination.exists():
        raise FileExistsError('Existing published export without completion marker; inspect before replacing')
    temp.replace(destination)
    save(work/'COMPLETE.json',dict(status='complete',processed=cfg['total'],export=str(destination),
        summary_sha256=sha(destination/'summary.json'),completed_at=time.time(),unresolved=summary['unresolved']))
    save(work/'status.json',dict(stage='complete',at=time.time(),**aggregate(metas,cfg),summary=str(destination/'summary.json')))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def supervise(work):
    lock=(work/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for attempt in range(1,4):
        if (work/'STOP').exists() or (work/'COMPLETE.json').exists():
            return
        command=[sys.executable,str(work/'code/driver.py'),'worker','--work',str(work)]
        with (work/f'worker-{attempt:02d}-{int(time.time())}.log').open('a') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            save(work/'process.json',dict(supervisor_pid=os.getpid(),worker_pid=p.pid,host=socket.gethostname(),
                attempt=attempt,command=command,started_at=time.time(),log=str(Path(log.name))))
            code=p.wait()
        if (work/'STOP').exists():
            return
        if code==0:
            finalize(work)
            return
        save(work/'last_failure.json',dict(at=time.time(),attempt=attempt,exit_code=code))
        if attempt<3:
            time.sleep(20)
    cfg=read(work/'config.json')
    metas,_=completed(work,cfg)
    save(work/'status.json',dict(stage='failed_after_retries',at=time.time(),**aggregate(metas,cfg)))
    raise RuntimeError('Worker failed three times; details retained in logs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','worker','finalize','supervise'])
    p.add_argument('--work',type=Path,default=DEFAULT)
    p.add_argument('--max-blocks',type=int,default=0)
    a=p.parse_args()
    if a.action=='worker':
        worker(a.work,a.max_blocks)
    else:
        {'prepare':prepare,'finalize':finalize,'supervise':supervise}[a.action](a.work)
