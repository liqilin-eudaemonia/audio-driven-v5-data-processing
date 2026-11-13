"""Paired fixed200 evaluation: legacy joint8000 vs all-LoRA6500, seeds137/238.

Reuse the existing 28-GPU progressive scheduler, native pinned model loaders,
shared sample claims, VAE decode, metric workers and global feature merge.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
from p2_combined_whisper_eval import link
WORK=ROOT/'experiments/p2_seamless200_pipeline_two_seeds_20261112'
PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
OLD=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
NEW=ROOT/'experiments/p2_vico43_all_lora6500_sft2752_20261111'
SEEDS=(137,238)
PIPELINES=('joint8000','all_lora6500')
LABELS={'joint8000':'旧联合8000＋SFT2752','all_lora6500':'全LoRA6500＋SFT2752'}
ARM=''


def stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def code_identity():
    paths=[SELF,ROOT/'scripts/p2_all_lora6428_seamless_eval28_20261109.py',
        ROOT/'src/p2_progressive_eval_20261108.py',ROOT/'src/p2_all_lora_followups_eval_20261109.py',
        ROOT/'scripts/p2_vico43_combined8000_sft2752.py',ROOT/'scripts/p2_combined_whisper_eval.py',
        ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',
        ROOT/'src/p2_testsets_eval/shared_items.py',ROOT/'src/p2_testsets_eval/cluster.py',
        ROOT/'src/p2_testsets_eval/metrics.py',ROOT/'scripts/p2_pipelined_testsets_eval.py',
        ROOT/'scripts/p2_pipelined_video_metrics.py',ROOT/'src/p2_testsets_eval/metric_pipeline.py']
    return {str(p):sha(p) for p in paths}


def validate_cluster(cluster):
    assert [n['name'] for n in cluster['nodes']]==['node12','node8','node4','node11']
    assert [n['gpus'] for n in cluster['nodes']]==[list(range(8))]*3+[list(range(4))]
    assert cluster['nodes'][3]['host']=='172.29.189.119'
    assert sum(len(n['gpus']) for n in cluster['nodes'])==28


def prepare(work=WORK):
    work=Path(work)
    if (work/'RUN.json').exists():
        p=read(work/'protocol.json');assert p['execution_code']==code_identity()
        validate_cluster(read(work/'cluster.json'))
        assert sha(work/'shared_items.json')==p['shared_items_sha256']
        return p
    work.mkdir(parents=True,exist_ok=True)
    if (work/'PREPARED.json').exists():
        for v in read(work/'PREPARED.json')['runs']:prepare(Path(v))
        return read(work/'PREPARED.json')
    from p2_testsets_eval.runtime import ready
    parent=read(PARENT/'protocol.json');old=read(OLD/'protocol.json');new=read(NEW/'protocol.json')
    assert parent['planner_adapter_identity']==old['planner_adapter_identity']==new['planner_adapter_identity']
    assert old['whisper_encoder']==new['whisper_encoder']==parent['whisper_encoder']
    assert old['feature_code']==new['feature_code']==parent['feature_code']
    assert sha(PARENT/'shared_items.json')==parent['shared_items_sha256']
    items=[x for x in read(PARENT/'shared_items.json') if x['dataset']=='seamless']
    spec=parent['datasets']['seamless'];refs=Path(spec['prepared'])/'references.jsonl'
    assert len(items)==len({x['key'] for x in items})==spec['count']==200
    assert sha(refs)==spec['references_sha256']==read(refs.parent/'seal.json')['references_sha256']
    assert spec['face_padding']==0
    for x in items:
        folder=PARENT/'conditions'/x['key']
        assert ready(folder/'qwen') and ready(folder/'text') and ready(PARENT/'whisper'/x['key'])
        assert read(folder/'qwen/plan.json')['adapter']==parent['planner_adapter_identity']
        assert not read(folder/'qwen/plan.json')['target_description_access']
        audit=read(folder/'input_audit.json')
        assert audit['source_only'] and not audit['listener_future_media_in_model_input']
        assert read(PARENT/'whisper'/x['key']/'metadata.json')['source']['input_audit_sha256']==sha(folder/'input_audit.json')
    shared=work/'shared_items.json';save(shared,items)
    oldarm=old['arms'][0];assert oldarm['step']==8000
    oldreceipt=read(Path(oldarm['checkpoint']).with_suffix('.export.json'))
    assert oldreceipt['all_model_tensors_bitwise_equal'] and oldreceipt['original_sha256']==oldarm['original_sha256']
    assert read(Path(oldarm['original_checkpoint']).with_suffix('.complete.json'))['sha256']==oldarm['original_sha256']
    assert sha(oldarm['checkpoint'])==oldarm['checkpoint_sha256']==oldreceipt['inference_sha256']
    oldstamp=stamp(oldarm['checkpoint'])
    newreceipt=read(NEW/'export/RECEIPT.json')
    assert newreceipt['step']==6500 and newreceipt['all_model_tensors_bitwise_equal']
    assert newreceipt['stamp']==stamp(newreceipt['checkpoint'])
    assert newreceipt['original_sha256']==new['arms'][0]['original_sha256']
    assert read(Path(newreceipt['original']).with_suffix('.complete.json'))['sha256']==newreceipt['original_sha256']
    c=copy.deepcopy(read(ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/cluster.json'))
    for n in c['nodes']:n['gpus']=list(range(4 if n['name']=='node11' else 8))
    c['training_script']=str(SELF);validate_cluster(c);save(work/'cluster.json',c)
    runs=[]
    # Complete both pipelines for a seed before moving to the other seed.
    for seed in SEEDS:
        for pipeline in PIPELINES:
            child=work/f'{pipeline}_seed{seed}';child.mkdir(exist_ok=True)
            source=old if pipeline=='joint8000' else new
            arm=copy.deepcopy(source['arms'][0]);arm['name']=f'{pipeline}_sft2752'
            p=copy.deepcopy(parent)
            for k in ['selection','selection_sha256','upstream_work','upstream_completion','source_media_identity','source_media_identity_sha256']:
                p.pop(k,None)
            p.update(schema='seamless-fixed200-two-pipeline-two-seed-v1',created_at=time.time(),
                arms=[arm],seed=seed,checkpoint_identity=source['checkpoint_identity'],datasets={'seamless':spec},
                workers=28,total_samples=200,total_outputs=200,cluster=str(child/'cluster.json'),
                metrics_cluster=str(child/'cluster.json'),worker_cli=str(SELF),
                shared_item_queue=True,shared_items_manifest=str(shared),shared_items_sha256=sha(shared),
                execution_code=code_identity(),authorized_scope='node12/node8/node4 GPU0-7; node11 GPU0-3 only; preserve all other tasks',
                metric_sampling='Original formulas; pooled full200 FID/FVD per seed, arithmetic mean of two completed seed scores',
                comparison='Same frozen200, SFT2752 plans/H/Gemma/Whisper and seed per paired pipeline; no best-of-N',
                conditioning_parent=str(PARENT),conditioning_parent_protocol_sha256=sha(PARENT/'protocol.json'),
                metrics_workers_per_gpu=2,metrics_inference_workers_per_gpu=2,
                metrics_gt_cache=str(work/'gt_cache'),
                metrics_use_released_inference_nodes=False)
            # Keep existing small200 decoded-GT cache; never predecode full6624.
            assert p['metrics_decoded_gt_cache']==parent['metrics_decoded_gt_cache']
            for name in ['conditions','whisper','negative']:link(PARENT/name,child/name)
            link(shared,child/'shared_items.json');link(work/'cluster.json',child/'cluster.json')
            if pipeline=='all_lora6500':link(NEW/'export',child/'export')
            data=(read(OLD/'adapter_session/protocol.json')['data_config'] if pipeline=='joint8000'
                  else read(NEW/'DATA_CONFIG.json'))
            save(child/'DATA_CONFIG.json',data)
            for rank,shard in enumerate(partition(items,28)):save(child/'shards'/f'{rank:02d}.json',shard)
            save(child/'RUN.json',dict(pipeline=pipeline,seed=seed,root=str(work),original=arm['original_checkpoint'],
                old_export_stamp=oldstamp if pipeline=='joint8000' else None))
            save(child/'protocol.json',p)
            save(child/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),outputs=200,seed=seed,
                shared200_conditions_verified=True,planner_step=2752,world=28,reserved_node11=[4,5,6,7]))
            runs.append(str(child))
    # Reuse immutable feature leaves with a new lock namespace for this task.
    from p2_combined_whisper_eval import reuse_gt_features
    reused=reuse_gt_features(work,[parent])
    report=dict(success=True,at=time.time(),runs=runs,outputs=800,seeds=list(SEEDS),
        fixed_pairs=200,shared_conditions_sha256=sha(shared),reused_gt_features=reused,
        pipelines=LABELS,reserved_node11=[4,5,6,7],scope='fixed diagnostic200 only; full test6624 not scheduled')
    save(work/'PREPARED.json',report);save(work/'status.json',dict(stage='prepared',at=time.time(),total_outputs=800))
    return report


def worker(work,stage,rank,slot=0):
    p=prepare(work);run=read(work/'RUN.json');cluster=read(work/'cluster.json')
    assert 0<=rank<28 and 0<=slot<4
    node=cluster['nodes'][rank//8];physical=rank%8
    assert physical in node['gpus'] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical)
    assert node['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    # Bound our process on shared cards without changing dtype/sampling.
    torch.cuda.set_per_process_memory_fraction(min(80*2**30,max(2**30,free-4*2**30))/total,0)
    from p2_testsets_eval import runtime as rt,whisper_runtime as wr
    from p2_testsets_eval.shared_items import all_items
    from p2_listen.config import extractor_config
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d);items=all_items(work,[],p);arm=p['arms'][0]
    lock=work/'locks'/f'{stage}_{rank}_{slot}.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as stream,ExitStack() as stack:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stack.enter_context(patch.object(rt,'data_config',lambda:(d,cfg)))
        stack.enter_context(patch.object(wr,'data_config',lambda:(d,cfg)))
        save(work/'devices'/f'{stage}_{rank}_{slot}.json',dict(pid=os.getpid(),at=time.time(),
            host=socket.gethostname(),physical_gpu=physical,node=node['name'],seed=p['seed'],stage=stage,status='started'))
        oldprogress=rt.progress
        stack.enter_context(patch.object(rt,'progress',lambda w,s,r,**kw:oldprogress(w,s,rank+28*slot,**kw)))
        if stage=='decode':rt.decode(work,items,rank,p,p['arms'])
        elif run['pipeline']=='joint8000':
            import p2_vico43_combined8000_sft2752 as historical
            with historical.sft_runtime():
                from p2_whisper_event_feature import checkpoint
                def checked(path):
                    assert str(path)==arm['checkpoint'] and stamp(path)==run['old_export_stamp']
                    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                    assert value['identity']==p['checkpoint_identity'] and value['step']==8000
                    return value
                with patch.object(checkpoint,'checked',checked):
                    wr.render(work,items,0,p,p['arms'])
        else:
            import p2_all_lora_followups_eval_20261109 as native
            import p2_ltx_events.inference as previous
            native.STEP=6500;native.TRAIN=ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109'
            native.WORK=work;native.ARM=arm['name']
            def loading(path,device='cuda:0'):
                result=native.load_model(work,path,device)
                assert len(result[0].text_targets)==1632
                save(work/'model_acceptance'/f'{rank:02d}.json',dict(success=True,at=time.time(),
                    strict_all_lora_load=True,targets=1632,original_sha256=arm['original_sha256']))
                return result
            with patch.object(previous,'load_model',loading):
                native.render(work,items,0,p)
        save(work/'devices'/f'{stage}_{rank}_{slot}.json',dict(pid=os.getpid(),at=time.time(),
            host=socket.gethostname(),physical_gpu=physical,node=node['name'],seed=p['seed'],stage=stage))


def final(work):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');arm=p['arms'][0]
    for x in read(work/'shared_items.json'):
        folder=work/'generated'/arm['name']/x['key'];assert ready(folder);m=read(folder/'metadata.json')
        assert m['step']==arm['step'] and m['seed']==p['seed']
        assert m['checkpoint_sha256']==arm['original_sha256'] and m['planning_adapter']==p['planner_adapter_identity']
        assert m['generation']==p['generation'] and not m['listener_future_input'] and not m['target_description_access']
    summary=read(work/'RESULTS.json')[arm['name']+'/seamless']
    assert summary['requested_pairs']==summary['audio']['requested_pairs']==200
    assert summary['audio']['pairs_with_errors']==0
    assert all(summary['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=200,seed=p['seed'],step=arm['step'],world=28))
    save(work/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),outputs=200))


def average_metrics(results):
    """Equal-seed mean only after both global fixed200 summaries are present."""
    assert set(results)==set(SEEDS)
    keys=set(results[137]['metrics']) & set(results[238]['metrics']);means={}
    for key in sorted(keys):
        values=[results[s]['metrics'][key] for s in SEEDS]
        means[key]=sum(values)/2 if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) else None
    return means


def summarize(work):
    report={};lines=['# 固定Seamless200：两条pipeline、双seed对比','',
        '每个seed独立完成全200条评分并全量汇总FID/FVD，再将seed137与238的最终分数等权平均；不把两个seed合并成400条计算FID，也不平均分片FID。','']
    for pipeline in PIPELINES:
        results={}
        for seed in SEEDS:
            child=work/f'{pipeline}_seed{seed}';p=read(child/'protocol.json')
            assert read(child/'FINAL_AUDIT.json')['success']
            results[seed]=read(child/'RESULTS.json')[p['arms'][0]['name']+'/seamless']
        report[pipeline]=dict(label=LABELS[pipeline],seed_results=results,mean=average_metrics(results))
    keys=['psnr','ssim','lpips','fid','fvd','emo_acc','emo_div_generated','emo_div_gt','delta_resp','utmos']
    lines += ['| pipeline | seed | '+' | '.join(keys)+' |','|---|---|'+'---:|'*len(keys)]
    for pipeline in PIPELINES:
        block=report[pipeline]
        for seed in [137,238,'mean']:
            metrics=block['mean'] if seed=='mean' else block['seed_results'][seed]['metrics']
            lines.append('| '+LABELS[pipeline]+' | '+str(seed)+' | '+' | '.join('N/A' if metrics.get(k) is None else f'{metrics[k]:.6f}' for k in keys)+' |')
    a,b=[report[k]['mean'] for k in PIPELINES]
    primary={k:('tie' if a[k]==b[k] else PIPELINES[0] if a[k]<b[k] else PIPELINES[1]) for k in ['fid','fvd']}
    verdict=(next(iter(primary.values())) if len(set(primary.values()))==1 else 'tradeoff')
    report['comparison']=dict(primary_distribution_metrics=primary,verdict=verdict,
        rule='FID/FVD lower is better; mixed winners are reported as tradeoff, no invented composite score',
        confidence='two fixed seeds only; no significance claim')
    lines += ['',f"FID/FVD平均值结论：{LABELS.get(verdict,verdict)}。其他指标保留各自取舍，不能仅凭两次采样宣称统计显著。",'',
        '图像和音频覆盖数、GT参考、N/A及逐条结果见各seed完整JSON。两条pipeline训练权重／训练过程不同，本轮比较完整pipeline，不作为严格单变量消融。','']
    save(work/'RESULTS.json',report);(work/'RESULTS.md').write_text('\n'.join(lines))
    save(work/'COMPLETE.json',dict(success=True,at=time.time(),outputs=800,seeds=list(SEEDS),scope='Seamless fixed200'))
    return report


def queue(work):
    global ARM
    prepare(work)
    with (work/'controller.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'CONTROLLER.json',dict(pid=os.getpid(),host=socket.gethostname(),at=time.time()))
        import p2_all_lora6428_seamless_eval28_20261109 as existing
        for child in map(Path,read(work/'PREPARED.json')['runs']):
            if (work/'STOP').exists():return
            if (child/'FINAL_AUDIT.json').exists():continue
            ARM=read(child/'protocol.json')['arms'][0]['name']
            save(work/'status.json',dict(stage='running',at=time.time(),run=child.name,total_outputs=800))
            stop_event=threading.Event()
            def propagate_stop():
                while not stop_event.wait(2):
                    if (work/'STOP').exists():save(child/'STOP',dict(reason='two-seed root STOP',at=time.time()));return
            thread=threading.Thread(target=propagate_stop,daemon=True);thread.start()
            try:
                with patch.object(existing,'E',sys.modules[__name__]),patch.object(existing,'SELF',SELF):existing.queue(child)
            finally:stop_event.set();thread.join(timeout=3)
            if (child/'STOP').exists():return
        summarize(work);save(work/'status.json',dict(stage='complete',at=time.time(),outputs=800))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','queue','worker','summarize'])
    parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--stage',choices=['render','decode']);parser.add_argument('--rank',type=int)
    parser.add_argument('--slot',type=int,default=0);args=parser.parse_args();work=args.work.resolve()
    try:
        if args.action=='worker':worker(work,args.stage,args.rank,args.slot)
        elif args.action=='prepare':prepare(work)
        elif args.action=='summarize':summarize(work)
        else:queue(work)
    except Exception as exc:
        save(work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
