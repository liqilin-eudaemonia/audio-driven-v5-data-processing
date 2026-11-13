#!/usr/bin/env python3
"""Paired 8/16 fps timing on frozen inputs. Never writes production labels.

One worker, one GPU, serial original-video decoding. ABBA/BAAB blocks control
order; shared-GPU observations are explicitly not an isolated speed benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/listen_visual_input50_20261029'
DEFAULT_OUT = RUN / 'native_fps_decision/speed'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def select_samples(samples, n=12):
    # Deterministic random sampling within duration strata, unrelated to outputs.
    if n < 3 or n > len(samples):
        raise ValueError('sample count must be between 3 and cohort size')
    ordered = sorted(samples, key=lambda s: (s['source']['duration_s'], s['sample_id']))
    rng = random.Random(20261029)
    selected = []
    for i in range(3):
        bucket = ordered[len(ordered)*i//3:len(ordered)*(i+1)//3]
        count = n//3 + (i < n % 3)
        selected.extend(rng.sample(bucket, count))
    return selected


def gpu_snapshot(gpu):
    fields = 'index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory'
    def run(args):
        return subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=10).stdout.strip()
    return {'time': time.time(), 'device': run(['-i', str(gpu), '--query-gpu='+fields, '--format=csv,noheader,nounits']),
            'processes': run(['-i', str(gpu), '--query-compute-apps=pid,used_memory,process_name', '--format=csv,noheader,nounits'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--gpu', type=int, default=7)
    ap.add_argument('--memory', type=float, default=.20)
    ap.add_argument('--batch-size', type=int, default=3)
    ap.add_argument('--max-model-len', type=int, default=49152)
    ap.add_argument('--views', choices=['full','full-and-middle'], default='full-and-middle')
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--shared-gpu', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    samples = select_samples(json.loads((RUN/'samples.json').read_text()), args.samples)
    plan = {'schema':'listen-native-fps-speed-v1', 'samples':[s['sample_id'] for s in samples],
            'selection':'fixed-seed random within three duration strata, not chosen by label quality',
            'views':args.views,
            'batch_size':args.batch_size, 'max_model_len':args.max_model_len, 'passes_per_block':4, 'repetitions_per_arm':2,
            'order':'alternate ABBA / BAAB across identical request batches',
            'gpu':args.gpu, 'shared_gpu':args.shared_gpu, 'host':socket.gethostname(),
            'memory_fraction':args.memory, 'cpu_threads':2, 'source_readers':1,
            'separate_arm_timing':True, 'prefix_cache':False, 'multimodal_processor_cache_gb':0,
            'model':'/private/ckpts/Qwen3-VL-8B-Instruct', 'temperature':0, 'max_tokens':4096,
            'seed':20261029, 'production_modified':False,
            'timing_scope':'prepared-input model.generate wall time, excluding source decode and model load',
            'not_isolated_warning':'Other training remains active; paired timings can still be confounded by varying load.' if args.shared_gpu else None}
    if args.plan_only:
        write(args.out/'plan.json',plan); print(json.dumps(plan,ensure_ascii=False)); return
    if (args.out/'measurements.json').exists():
        raise RuntimeError('Refusing to mix a new run with old measurements; choose a new --out')
    args.out.mkdir(parents=True, exist_ok=True)
    initial = gpu_snapshot(args.gpu)
    values = [x.strip() for x in initial['device'].split(',')]
    if len(values) != 7:
        raise RuntimeError('Cannot validate target GPU')
    free = float(values[4])-float(values[3])
    if free < float(values[4])*args.memory + 4096:
        raise RuntimeError('Insufficient free GPU memory for declared allocation plus 4GiB margin')
    if initial['processes'] and not args.shared_gpu:
        raise RuntimeError('GPU occupied; explicitly declare shared-GPU timing or use an idle GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    write(args.out/'plan.json',plan)
    write(args.out/'status.json',{'state':'preparing','pid':os.getpid(),'started_at':time.time(),'plan':plan})
    spec = importlib.util.spec_from_file_location('frozen_native_input_run',RUN/'run.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(mod.MODEL, trust_remote_code=True, local_files_only=True)
    # prepared_inputs historically assumes image8/video8 media equality. Keep its
    # immutable implementation and select the two native outputs instead.
    mod.ARMS = ['image8','video8','video16']
    prepared, decode_records = [], []
    for s in samples:
        began=time.perf_counter()
        with mod.decode_slot(): whole=mod.prepare_media(s['source'])
        source = json.loads((RUN/'results'/(s['sample_id']+'.json')).read_text())
        for arm, fps in [('video8',8),('video16',16)]:
            old=next(r for r in source['records'] if r['arm']==arm and r['view']=='full')
            assert whole[fps]['meta']['video']['frames']==old['media']['video']['frames']
        decode_records.append({'sample_id':s['sample_id'],'seconds':time.perf_counter()-began,
                               'note':'shared single decode at16fps supplies both arms; not separate decode timings'})
        candidates=mod.views(whole[8]['duration_s'])
        full=next(v for v in candidates if v[0]=='full')
        windows=[v for v in candidates if v[0]!='full']
        selected=[full] + ([windows[len(windows)//2]] if args.views=='full-and-middle' and windows and full[2]-full[1]>3 else [])
        for view,start,end in selected:
            inputs,evidence=mod.prepared_inputs(processor,whole,start,end,view)
            prepared.append({'sample_id':s['sample_id'],'view':view,'duration':end-start,
                             'inputs':inputs[1:],'evidence':evidence[1:]})
        print('prepared',s['sample_id'],flush=True)
    write(args.out/'decode.json',decode_records)
    stop=threading.Event()
    def monitor():
        with (args.out/'resources.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                try: f.write(json.dumps(gpu_snapshot(args.gpu))+'\n')
                except Exception as e:f.write(json.dumps({'time':time.time(),'error':str(e)})+'\n')
                stop.wait(3)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    measurements=[]
    try:
        t=time.perf_counter()
        llm=LLM(model=mod.MODEL,dtype='bfloat16',gpu_memory_utilization=args.memory,
                max_model_len=args.max_model_len,max_num_seqs=args.batch_size,max_num_batched_tokens=4096,
                enable_chunked_prefill=True,limit_mm_per_prompt={'image':128,'video':1},
                skip_mm_profiling=True,mm_processor_kwargs={'size':mod.MM_SIZE},
                mm_processor_cache_gb=0,enable_prefix_caching=False,enforce_eager=True,seed=20261029)
        load_seconds=time.perf_counter()-t
        def params(item):
            return SamplingParams(temperature=0,max_tokens=4096,seed=20261029,
                structured_outputs=StructuredOutputsParams(json=mod.response_schema(item['duration'])))
        for arm_index in (0,1):
            warm=prepared[:args.batch_size]
            llm.generate([r['inputs'][arm_index] for r in warm],[params(r) for r in warm],use_tqdm=False)
        batches=[]
        for is_full in (True,False):
            items=[r for r in prepared if (r['view']=='full')==is_full]
            for i in range(0,len(items),args.batch_size):batches.append(items[i:i+args.batch_size])
        for bi,batch in enumerate(batches):
            order=[0,1,1,0] if bi%2==0 else [1,0,0,1]
            for pi,ai in enumerate(order):
                before=gpu_snapshot(args.gpu)
                began=time.perf_counter()
                output=llm.generate([r['inputs'][ai] for r in batch],[params(r) for r in batch],use_tqdm=False)
                elapsed=time.perf_counter()-began
                records=[]
                for item,response in zip(batch,output):
                    out=response.outputs[0]; valid=False;error=None
                    try:
                        mod.validate_timeline(json.loads(out.text),item['duration'])
                        valid=out.finish_reason=='stop'
                    except Exception as e:error=str(e)
                    records.append({'sample_id':item['sample_id'],'view':item['view'],'duration':item['duration'],
                        'input_tokens':len(response.prompt_token_ids),'output_tokens':len(out.token_ids),
                        'valid':valid,'error':error,'finish_reason':out.finish_reason,'raw':out.text,
                        'input_sha256':hashlib.sha256(json.dumps(item['evidence'][ai],sort_keys=True).encode()).hexdigest()})
                measurements.append({'batch':bi,'pass':pi,'arm':['video8','video16'][ai],
                    'elapsed_s':elapsed,'records':records,'gpu_before':before,'gpu_after':gpu_snapshot(args.gpu)})
                write(args.out/'measurements.json',measurements)
                write(args.out/'status.json',{'state':'running','completed_batches':len(measurements),
                     'total_batches':len(batches)*4,'shared_gpu':args.shared_gpu,'updated_at':time.time()})
                print('timed',bi,pi,['video8','video16'][ai],round(elapsed,3),flush=True)
        totals={}
        for arm in ('video8','video16'):
            ms=[m for m in measurements if m['arm']==arm]
            secs=sum(m['elapsed_s'] for m in ms);rs=[r for m in ms for r in m['records']]
            totals[arm]={'seconds':secs,'requests':len(rs),'valid_requests':sum(r['valid'] for r in rs),
                         'requests_per_minute':60*len(rs)/secs,'valid_requests_per_minute':60*sum(r['valid'] for r in rs)/secs,
                         'input_tokens':sum(r['input_tokens'] for r in rs),'output_tokens':sum(r['output_tokens'] for r in rs)}
        paired=[]
        for bi in range(len(batches)):
            ts={a:sum(m['elapsed_s'] for m in measurements if m['batch']==bi and m['arm']==a) for a in totals}
            paired.append({'batch':bi,'ratio_16_to_8':ts['video16']/ts['video8']})
        result={'state':'complete','plan':plan,'totals':totals,'paired_batches':paired,
                'ratio_16_to_8_wall_time':totals['video16']['seconds']/totals['video8']['seconds'],
                'model_load_seconds':load_seconds,'isolated_speed_claim_allowed':not args.shared_gpu,
                'end_to_end_speed_claim_allowed':False,'completed_at':time.time()}
        write(args.out/'summary.json',result);write(args.out/'status.json',result)
    except BaseException as e:
        write(args.out/'status.json',{'state':'failed','error':str(e),'completed_batches':len(measurements),'updated_at':time.time()})
        raise
    finally:stop.set();watcher.join(timeout=11)


if __name__=='__main__':main()
