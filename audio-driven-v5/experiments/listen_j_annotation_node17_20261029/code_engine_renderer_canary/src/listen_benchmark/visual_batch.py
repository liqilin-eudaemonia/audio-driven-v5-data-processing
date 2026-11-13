"""Cross-clip batching with source-specific checkpoints and unchanged inputs."""
import json
import time
from collections import Counter
from .full_raw import views
from .simple_timeline import validate_timeline
from .visual_native8 import prepare_input


def annotate_group(rows_media, processor, llm, make_params, output, batch_size,
                   frame_budget, save, config_hash):
    states={};jobs=[]
    for row,whole in rows_media:
        sid=row['sample_id'];duration=whole[8]['duration_s'];windows=views(duration)
        path=output/'partial'/f'{sid}.json'
        checkpoint=json.loads(path.read_text()) if path.exists() else {'sample_id':sid,'records':[]}
        if checkpoint['sample_id']!=sid:raise ValueError('Checkpoint sample mismatch')
        records=checkpoint['records'];expected={v:(s,e) for v,s,e in windows}
        done={r['view'] for r in records}
        if len(done)!=len(records):raise ValueError('Duplicate checkpoint views')
        for r in records:
            if r['view'] not in expected or (r['start'],r['end'])!=expected[r['view']] or r['arm']!='video8':
                raise ValueError('Checkpoint window mismatch')
            if r.get('status') not in ('valid','invalid'):raise ValueError('Incomplete checkpoint')
        states[sid]={'sample_id':sid,'duration_s':duration,'records':records,'config_sha256':config_hash,
            'performance':{'resumed_windows':len(records),'new_windows':0,'batches':0,'batch_sizes':[],
                'prepare_s':0.,'generate_s':0.,'timing_scope':'current_attempt_allocated_across_group'}}
        for view,start,end in windows:
            if view not in done:jobs.append((sid,whole,view,start,end))
    jobs.sort(key=lambda j:-(j[4]-j[3]))
    cursor=0
    while cursor<len(jobs):
        prepared=[];batch=[];frames=0;params=[];engine_inputs=[]
        while cursor<len(jobs) and len(batch)<batch_size:
            sid,whole,view,start,end=job=jobs[cursor]
            # Same frame slicing as slice_media; count before creating tensors.
            import math
            count=len(whole[8]['images']) if view=='full' else min(len(whole[8]['images']),math.ceil(end*8-1e-8))-round(start*8)
            count+=count%2
            if batch and frames+count>frame_budget:break
            tick=time.monotonic()
            cached=whole.get('_prepared_inputs',{}).get(view)
            if cached is None:
                entry,evidence=prepare_input(processor,whole,start,end,view)
            else:
                if cached['interval']!=(start,end):raise ValueError('Precompiled input interval mismatch')
                entry,evidence=cached['input']
            states[sid]['performance']['prepare_s']+=time.monotonic()-tick
            prepared.append((entry,evidence));batch.append(job);params.append(make_params(end-start))
            engine_inputs.append(cached.get('engine_input') if cached is not None else None)
            frames+=count;cursor+=1
        tick=time.monotonic()
        if any(item is not None for item in engine_inputs):
            if not all(item is not None for item in engine_inputs):
                raise ValueError('Cannot mix rendered and unrendered inputs in one batch')
            from .engine_input_renderer import generate_engine_inputs
            responses=generate_engine_inputs(llm,engine_inputs,params)
        else:
            responses=llm.generate([x[0] for x in prepared],params,use_tqdm=False)
        elapsed=time.monotonic()-tick
        if len(responses)!=len(batch):raise RuntimeError('Incomplete inference batch')
        batch_id=f'{batch[0][0]}:{batch[0][2]}'
        for (sid,whole,view,start,end),(_,evidence),response in zip(batch,prepared,responses):
            answer=response.outputs[0]
            rec=dict(arm='video8',view=view,start=start,end=end,raw=answer.text,finish_reason=answer.finish_reason,
                prompt_tokens=len(response.prompt_token_ids),completion_tokens=len(answer.token_ids),
                elapsed_s=elapsed,elapsed_scope='shared_batch_wall_time',batch_id=batch_id,batch_size=len(batch),**evidence)
            try:
                rec['parsed']=json.loads(answer.text);validate_timeline(rec['parsed'],end-start)
                assert all(e['modality']=='visual' for e in rec['parsed']['events']) and answer.finish_reason=='stop'
                rec['status']='valid'
            except Exception as exc:rec.update(status='invalid',error=str(exc))
            states[sid]['records'].append(rec)
        for sid,count in Counter(j[0] for j in batch).items():
            state=states[sid];order={v:i for i,(v,_,_) in enumerate(views(state['duration_s']))}
            state['records'].sort(key=lambda r:order[r['view']])
            perf=state['performance'];perf['generate_s']+=elapsed*count/len(batch)
            perf['new_windows']+=count;perf['batches']+=1;perf['batch_sizes'].append(len(batch))
            save(output/'partial'/f'{sid}.json',{'sample_id':sid,'records':state['records']})
        del prepared,responses
    for state in states.values():
        assert {r['view'] for r in state['records']}=={v for v,_,_ in views(state['duration_s'])}
    return list(states.values())
