"""One GPU, bounded asynchronous preparation, shared shard claims, raw-only outputs."""
import argparse, dataclasses, fcntl, hashlib, json, os, queue, socket, sys, threading, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import prepare, views
from p2_listen.contracts import atomic_json


def recover(path):
    """A committed newline is a record; remove only an interrupted trailing write."""
    done={};counts=Counter()
    if not path.exists():return done,counts
    with path.open('r+b') as f:
        while True:
            pos=f.tell();line=f.readline()
            if not line:break
            if not line.endswith(b'\n'):
                f.truncate(pos);break
            r=json.loads(line)
            if r.get('record_kind')=='result':done[r['request_id']]=r['status']
    counts.update(done.values())
    return done,counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--branch',choices=['visual','audio','asr'],required=True)
    p.add_argument('--worker',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--prepare-workers',type=int,default=16);p.add_argument('--memory',type=float,default=.90)
    a=p.parse_args();run=Path(a.run).resolve();config=json.loads((run/'config.json').read_text())
    branch=a.branch;worker=a.worker;statefile=run/'workers'/(worker+'.json')
    state=dict(worker=worker,pid=os.getpid(),host=socket.gethostname(),branch=branch,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               batch=a.batch,prepare_workers=a.prepare_workers,started_at=time.time(),completed_this_process=0)
    def status(**kw):state.update(kw,heartbeat=time.time());atomic_json(statefile,state)
    status(status='loading_model')
    import torch
    torch.set_num_threads(4)
    if branch=='asr':
        sys.path.insert(0,config['asr_dependency_path'])
        from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
        model=Qwen3ASRModel.from_pretrained(config['models']['asr'],dtype=torch.bfloat16,device_map='cuda:0',
                                          max_inference_batch_size=128,max_new_tokens=256)
        aligner=Qwen3ForcedAligner.from_pretrained(config['models']['aligner'],dtype=torch.bfloat16,device_map='cuda:0')
    else:
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from listen_benchmark.prompts import annotation_prompt
        from listen_benchmark.simple_timeline import response_schema,validate_timeline
        from listen_labeling.schema import parse
        from listen_benchmark.media import timestamped_images
        processor=AutoProcessor.from_pretrained(config['models'][branch],trust_remote_code=True,local_files_only=True)
        model=LLM(model=config['models'][branch],trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=a.memory,
            max_model_len=49152 if branch=='visual' else 8192,max_num_seqs=128,
            max_num_batched_tokens=32768 if branch=='visual' else 16384,
            enable_chunked_prefill=True,skip_mm_profiling=True,enforce_eager=True,
            limit_mm_per_prompt={'image':128,'video':0} if branch=='visual' else {'image':0,'audio':1,'video':0})
    status(status='waiting_for_verified_index')
    while not (run/'queue/index.json').exists():
        if (run/'STOP').exists():return
        time.sleep(2)
    shards=sorted((run/'queue/shards').glob('*.jsonl'));shards=shards[a.rank%len(shards):]+shards[:a.rank%len(shards)]
    from listen_benchmark.windows import slice_media
    workroot=run/'raw'/branch
    for name in ['locks','state','results']:(workroot/name).mkdir(parents=True,exist_ok=True)
    def prep(row):
        try:return row,prepare(row,branch),None
        except Exception as exc:return row,None,dict(type=type(exc).__name__,message=str(exc))
    def infer(batch, write):
        began=time.monotonic()
        if branch=='asr':
            transcripts=model.transcribe(audio=[(b['media']['audio'],16000) for b in batch],language=None)
            if len(transcripts)!=len(batch):raise ValueError('ASR result count mismatch')
            for b,t in zip(batch,transcripts):
                b['transcript']=dict(text=t.text,language=t.language)
                write(dict(b['record'],record_kind='asr_transcript',raw_response=b['transcript']))
            chosen=[i for i,t in enumerate(transcripts) if t.text.strip()]
            aligned={}
            if chosen:
                try:
                    outputs=aligner.align(audio=[(batch[i]['media']['audio'],16000) for i in chosen],
                        text=[transcripts[i].text for i in chosen],language=[transcripts[i].language for i in chosen])
                    if len(outputs)!=len(chosen):raise ValueError('Aligner result count mismatch')
                    aligned=dict(zip(chosen,outputs))
                except Exception as exc:
                    for i in chosen:aligned[i]=exc
            for i,(b,t) in enumerate(zip(batch,transcripts)):
                r=dict(b['record'],raw_response=dict(text=t.text,language=t.language),status='complete')
                if isinstance(aligned.get(i),Exception):r.update(status='alignment_failed',error=repr(aligned[i]))
                else:r['raw_response']['words']=[dataclasses.asdict(x) for x in aligned[i]] if i in aligned else []
                r.update(batch_size=len(batch),batch_seconds=time.monotonic()-began);write(r)
            return
        payloads=[];params=[]
        for b in batch:
            media=b['media'];text=annotation_prompt(dict(branch=branch,prompt_variant=f'backchannel_{branch}_simple_v2'),media['duration_s'],'discovery')
            parts=[dict(type='text',text=text)];mm={};options={}
            if branch=='visual':parts.extend(timestamped_images(media));mm['image']=media['images']
            else:
                parts.append(dict(type='audio',audio='listener://independent',audio_url='listener://independent'))
                mm['audio']=[media['audio']];options['use_audio_in_video']=False
            rendered=processor.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            b['record']['prompt']=text;b['record']['rendered_prompt']=rendered
            payloads.append(dict(prompt=rendered,multi_modal_data=mm,mm_processor_kwargs=options))
            params.append(SamplingParams(temperature=0,max_tokens=4096,structured_outputs=StructuredOutputsParams(json=response_schema(media['duration_s']))))
        outputs=model.generate(payloads,params,use_tqdm=False)
        if len(outputs)!=len(batch):raise ValueError('Engine output count mismatch')
        elapsed=time.monotonic()-began
        for b,o in zip(batch,outputs):
            item=o.outputs[0]
            r=dict(b['record'],raw_response=dict(text=item.text,finish_reason=item.finish_reason,
                   generated_token_ids=list(item.token_ids),prompt_tokens=len(o.prompt_token_ids),completion_tokens=len(item.token_ids)),
                   status='schema_pass',batch_size=len(batch),batch_seconds=elapsed)
            try:
                if item.finish_reason!='stop':raise ValueError('Incomplete/non-stop generation')
                r['parsed']=validate_timeline(parse(item.text),b['media']['duration_s'])
                if any(e['modality']!=branch for e in r['parsed']['events']):raise ValueError('Wrong modality in response')
            except Exception as exc:r.update(status='validation_failed',error=repr(exc))
            write(r)
    def safe_infer(batch,write):
        try:infer(batch,write)
        except Exception as exc:
            # Keep the failed batch attempt; isolate bad inputs without losing others.
            write(dict(record_kind='batch_error',request_ids=[b['record']['request_id'] for b in batch],
                       error=repr(exc),traceback=traceback.format_exc()[-4000:],utc=time.time()))
            if 'out of memory' in str(exc).lower():torch.cuda.empty_cache()
            if any(x in str(exc).lower() for x in ['enginecore', 'engine core', 'engine is dead', 'connection reset']):raise
            if len(batch)>1:
                mid=len(batch)//2;safe_infer(batch[:mid],write);safe_infer(batch[mid:],write)
            else:write(dict(batch[0]['record'],status='inference_failed',error=repr(exc),
                            raw_transcript=batch[0].get('transcript')))
    for path in shards:
        if (run/'STOP').exists():break
        shard=path.stem;sp=workroot/'state'/(shard+'.json')
        if sp.exists() and json.loads(sp.read_text()).get('done'):continue
        with (workroot/'locks'/(shard+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if sp.exists() and json.loads(sp.read_text()).get('done'):continue
            dest=workroot/'results'/(shard+'.jsonl');done,counts=recover(dest)
            parents=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            status(status='preparing',shard=shard)
            started=time.monotonic();out=dest.open('ab',buffering=1024*1024)
            def write(r):
                r.update(worker=worker,branch=branch,run_id=run.name,recorded_at=time.time())
                out.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())
                if r.get('record_kind')=='result':
                    rid=r['request_id']
                    if rid in done:counts[done[rid]]-=1
                    done[rid]=r['status'];counts[r['status']]+=1;state['completed_this_process']+=1
                # ASR text survives a later alignment exception/crash.
                if r.get('record_kind')=='asr_transcript':out.flush()
            prepared=queue.Queue(maxsize=max(16,a.prepare_workers*2));endmarker=object()
            remaining=[r for r in parents if r['sample_id']+'@input' not in done and
                       any(r['sample_id']+'@'+name not in done for name,_,_ in views(r['duration_s']))]
            def producer():
                try:
                    with ThreadPoolExecutor(max_workers=a.prepare_workers) as pool:
                        # Bounded submission: no million-item in-memory future list.
                        for offset in range(0,len(remaining),a.prepare_workers):
                            for item in pool.map(prep,remaining[offset:offset+a.prepare_workers]):prepared.put(item)
                except Exception as exc:prepared.put(exc)
                finally:prepared.put(endmarker)
            producer_thread=threading.Thread(target=producer,daemon=True);producer_thread.start()
            pending=[];finished=False;wait_seconds=0.;gpu_seconds=0.;batches=0
            while not finished or pending:
                control=run/'control.json';settings=json.loads(control.read_text()) if control.exists() else {}
                batchsize=max(1,min(128,settings.get(worker,{}).get('batch',a.batch)))
                while len(pending)<batchsize and not finished:
                    began=time.monotonic();item=prepared.get();wait_seconds+=time.monotonic()-began
                    if item is endmarker:finished=True;break
                    if isinstance(item,Exception):raise item
                    row,media,error=item;sid=row['sample_id']
                    record=dict(record_kind='result',sample_id=sid,source=row,human_verified=False,training_admitted=False)
                    if error:
                        write(dict(record,request_id=sid+'@input',status='preparation_failed',error=error,expected_views=len(views(row['duration_s']))));continue
                    for name,start,end in views(media['duration_s']):
                        rid=sid+'@'+name
                        if rid in done:continue
                        part=media if name=='full' else slice_media(media,start,end,8)
                        rr=dict(record,request_id=rid,view=name,window_start_s=start,window_end_s=end,
                            time_origin='window_start_seconds',parent_duration_s=media['duration_s'],input_evidence=part['meta'])
                        pending.append(dict(record=rr,media=part))
                if pending:
                    batch=pending[:batchsize];del pending[:batchsize]
                    status(status='inference',batch=len(batch),shard=shard,preparation_queue=prepared.qsize(),
                           cumulative_input_wait_s=wait_seconds,cumulative_inference_s=gpu_seconds)
                    began=time.monotonic();safe_infer(batch,write);gpu_seconds+=time.monotonic()-began;batches+=1
                out.flush();os.fsync(out.fileno())
                atomic_json(sp,dict(shard=shard,worker=worker,done=False,counts=counts,requests=len(done),heartbeat=time.time()))
                status(status='running',shard=shard,shard_counts=dict(counts),batches=batches,
                       cumulative_input_wait_s=wait_seconds,cumulative_inference_s=gpu_seconds,
                       requests_per_second=len(done)/max(.001,time.monotonic()-started))
                if (run/'STOP').exists():out.close();return
            producer_thread.join();out.close()
            atomic_json(sp,dict(shard=shard,worker=worker,done=True,counts=counts,requests=len(done),parents=len(parents),heartbeat=time.time()))
    status(status='finished_available_shards')

if __name__=='__main__':main()
