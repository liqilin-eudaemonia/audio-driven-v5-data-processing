"""Audit known queue sources only; never scan the result tree."""
import json
import argparse
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path,digest
from listen_benchmark.full_raw import views
from listen_benchmark.visual_native8 import observation_prompt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--tag',default='rolling_canary_first100_audit')
    parser.add_argument('--worker',help='Only results with this recorded execution.worker_id')
    parser.add_argument('--renderer-workers',type=int,choices=[2,4],help='Restrict to a CPU renderer configuration')
    parser.add_argument('--decode-backend',choices=['thread','process','hybrid'])
    parser.add_argument('--annotation-groups',type=int,choices=[2,4],help='Restrict and validate rolling group capacity')
    parser.add_argument('--queue-snapshot',type=Path,help='Start from this canary-specific queue snapshot')
    args=parser.parse_args()
    if args.skip<0 or args.count<1 or Path(args.tag).name!=args.tag or args.tag in ('.','..'):
        parser.error('Invalid sample range or report tag')
    destination=WORK/'monitor_snapshots'/f'{args.tag}.json'
    if destination.exists():parser.error('Report already exists; choose a new tag')
    before=read(args.queue_snapshot or WORK/'monitor_snapshots/rolling_canary_before.json')
    first=before['snapshot']['vlm_queue_range'];current=read(WORK/'queues/vlm/state.json')
    if current['offset']<first['offset']:raise ValueError('Queue journal offset regressed')
    jobs=set(first['inflight'])
    with (WORK/'ready.jsonl').open('rb') as f:
        f.seek(first['offset'])
        for line in f.read(current['offset']-first['offset']).splitlines():jobs.add(json.loads(line)['job_id'])
    candidates=[];seen=set();source_jobs={}
    for job in sorted(jobs):
        spec=read(WORK/'jobs'/f'{job}.json');source_jobs[job]=spec
        for row in spec['rows']:
            sid=row['sample_id']
            if sid in seen:continue
            seen.add(sid);path=result_path(WORK,'vlm',sid)
            if not path.exists():continue
            d=read(path)
            eligible=d['execution'].get('rolling_inference') or (args.decode_backend=='process' and d['execution'].get('decode_backend')=='process')
            if d.get('completed_at',0)>=before['at'] and eligible:
                if args.worker and d['execution'].get('worker_id')!=args.worker:continue
                if args.renderer_workers is not None and d['execution'].get('renderer_workers')!=args.renderer_workers:continue
                if args.decode_backend is not None and d['execution'].get('decode_backend','thread')!=args.decode_backend:continue
                if args.annotation_groups is not None and d['execution'].get('rolling_annotation_groups')!=args.annotation_groups:continue
                candidates.append((d,row,job))
    candidates.sort(key=lambda x:(x[0]['completed_at'],x[0]['sample_id']))
    if len(candidates)<args.skip+args.count:raise SystemExit(f'Only {len(candidates)} rolling results available; keep waiting')
    chosen=candidates[args.skip:args.skip+args.count];errors=[];invalid=[];requests=valid=truncated=0
    config_hash=digest((ROOT/'configs/listen_visual_native8_nod_v1.json').read_bytes())
    for d,row,job in chosen:
        sid=row['sample_id']
        try:
            assert d['sample_id']==sid and d['source']==row['source'],'source mismatch'
            assert d['config_sha256']==config_hash and d['training_admitted'] is False,'configuration/admission mismatch'
            expected=views(d['duration_s'])
            assert [(r['view'],r['start'],r['end']) for r in d['records']]==expected,'missing/extra/reordered windows'
            ex=d['execution']
            assert ex['rolling_annotation_groups']==((args.annotation_groups or 2) if ex['rolling_inference'] else 1) and ex['batch_size']==512 and ex['tokens']==32768 and ex['frames']==16384,'execution limits changed'
            if ex.get('decode_backend')=='process':
                perf=d['input_performance']
                assert perf.get('decode_backend')=='process' and perf.get('decoder_cuda_visible_devices')=='','non-CPU decoder evidence'
                assert type(perf.get('decoder_pid')) is int and perf['decoder_pid']!=ex.get('worker_pid'),'decoder was not a separate process'
            if ex.get('decode_backend')=='hybrid':
                perf=d['input_performance'];hybrid=perf.get('hybrid_decode',{})
                assert perf.get('decode_backend')=='hybrid','hybrid path evidence missing'
                assert perf.get('decoder_cuda_visible_devices')==ex['worker_id'].split('-')[1],'decoder GPU assignment changed'
                assert type(perf.get('decoder_pid')) is int and perf['decoder_pid']!=ex.get('worker_pid'),'decoder was not a separate process'
                if perf.get('media_cache_hit'):
                    assert hybrid.get('reused_exact_media_cache') is True,'cache reuse provenance missing'
                elif hybrid.get('cuda_used'):
                    assert hybrid.get('source_codec')=='h264' and hybrid.get('source_pixel_format')=='yuv420p','unvalidated CUDA source format'
                    assert hybrid.get('pixel_normalization')=='nv12_to_yuv420p_before_original_rgb','pixel normalization missing'
                    assert not hybrid.get('cpu_fallback'),'conflicting hybrid decode provenance'
                else:
                    assert hybrid.get('cpu_reason') in ('format_outside_validated_cuda_path','hardware_attempt_failed'),'CPU selection provenance missing'
                    if hybrid.get('cpu_fallback'):assert hybrid.get('cuda_error'),'silent hardware fallback'
            for r in d['records']:
                requests+=1;valid+=r['status']=='valid';truncated+=r['finish_reason']=='length'
                if r['status']!='valid':invalid.append({'sample_id':sid,'view':r['view'],'error':r.get('error'),'finish_reason':r['finish_reason']})
                assert r['arm']=='video8' and r['status'] in ('valid','invalid') and isinstance(r['raw'],str),'raw result missing'
                assert r['prompt']==observation_prompt(r['end']-r['start']),'prompt changed'
                video=r['media']['video'];native=r['native']
                assert video['source']==row['source']['source'] and video['video_transform']==row['source']['video_transform'],'media provenance/crop mismatch'
                assert video['requested_fps']==8 and native['audio_input'] is False,'input modality/rate changed'
                assert native['do_resize'] is False and native['do_sample_frames'] is False,'resampling changed'
                frames=video['frames'];indices=native['metadata']['frames_indices']
                assert len(frames)==len(indices) and indices==[round(f['time_s']*video['source_fps']) for f in frames],'source metadata mutated'
                assert native['final_frame_padding_count']==len(frames)%2,'padding mismatch'
                assert all(f['size']==[512,512] and len(f['jpeg_sha256'])==64 for f in frames),'frame evidence missing'
        except Exception as exc:errors.append({'sample_id':sid,'error':repr(exc)})
    receipts=[]
    for job in sorted({x[2] for x in chosen}):
        spec=source_jobs[job];saved=sum(result_path(WORK,'vlm',r['sample_id']).exists() for r in spec['rows'])
        complete=(WORK/'vlm/completed'/f'{job}.json').exists()
        receipts.append({'job':job,'saved':saved,'total':len(spec['rows']),'receipt_exists':complete})
        if complete and saved!=len(spec['rows']):errors.append({'job':job,'error':'premature source receipt'})
    report={'at':time.time(),'known_sources_read':len(jobs),'selected_clips':len(chosen),'available_rolling_clips':len(candidates),
        'requests':requests,'valid_requests':valid,'truncated_requests':truncated,'errors':errors,'invalid_responses':invalid,
        'source_receipts':receipts,'sample_ids':[x[0]['sample_id'] for x in chosen],
        'passed_input_provenance_and_window_audit':not errors,
        'selection':{'skip':args.skip,'count':args.count,'worker':args.worker,'renderer_workers':args.renderer_workers,'decode_backend':args.decode_backend,'annotation_groups':args.annotation_groups,'queue_snapshot':str(args.queue_snapshot) if args.queue_snapshot else None,'order':'completed_at then sample_id'},
        'scope':'Selected rolling results; source/window/input metadata and raw preservation checks. Structural validity is not semantic accuracy; input media hashes are not independently redecoded by this audit.'}
    write(destination,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sample_ids','source_receipts','invalid_responses']},ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
