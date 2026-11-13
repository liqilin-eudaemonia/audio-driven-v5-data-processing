"""Queued Whisper6000/8000 supplement on the same sealed Seamless200 benchmark."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
DEFAULT=ROOT/'experiments/p2_seamless200_whisper_20261105'
BASE=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
TRAIN=ROOT/'experiments/p2_j_qwen_whisper_av_tuned_20261104'


def link(dst,src):
    if dst.is_symlink():
        if dst.resolve()!=src.resolve():raise ValueError('Different preexisting symlink '+str(dst))
        return
    if dst.exists():raise FileExistsError(dst)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src,target_is_directory=src.is_dir())


def export_checkpoint(work,step):
    import torch
    from p2_whisper_av.checkpoint import checked
    from p2_listen.cache import atomic_torch
    original=TRAIN/f'step_{step:06d}.pt';marker=read(original.with_suffix('.complete.json'))
    dest=work/'checkpoints'/original.name;audit=dest.with_suffix('.export.json')
    if audit.exists():
        saved=read(audit)
        if saved['original_sha256']!=marker['sha256']:raise ValueError('Changed training checkpoint')
        value=checked(dest)
        return value,saved
    value=checked(original)
    from p2_whisper_av.config import source_identity
    if value['identity']['source']!=source_identity():raise ValueError('Whisper training code identity changed')
    model_only={k:value[k] for k in ['schema','step','identity','model']}
    atomic_torch(dest,model_only)
    checksum=sha(dest);save(dest.with_suffix('.complete.json'),dict(step=step,sha256=checksum))
    reread=checked(dest)
    if reread['identity']!=value['identity'] or set(reread['model'])!=set(value['model']):
        raise ValueError('Model-only export metadata mismatch')
    for name,tensor in value['model'].items():
        if not torch.equal(tensor,reread['model'][name]):raise ValueError('Export changed model '+name)
    report=dict(original_checkpoint=str(original),original_sha256=marker['sha256'],
        inference_checkpoint=str(dest),inference_sha256=checksum,step=step,
        optimizer_omitted=True,all_model_tensors_bitwise_equal=True,tensors=len(value['model']))
    save(audit,report)
    return reread,report


def prepare(work):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4)
    work.mkdir(parents=True,exist_ok=True)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        for name,expected in p['execution_code'].items():
            if sha(name)!=expected:raise ValueError('Queued execution code changed: '+name)
        return p
    save(work/'status.json',dict(stage='cpu_preparing',at=time.time()))
    cfg=read(BASE/'cluster.json');extra=read(BASE/'metrics_cluster.json')['nodes'][0]
    assert extra['name']=='node8' and extra['gpus']==[0,1,2,3]
    cfg['nodes'].append(extra);save(work/'cluster.json',cfg)
    p=read(BASE/'base_standard/protocol.json');items=[]
    for rank in range(p['workers']):items+=read(BASE/'base_standard/shards'/f'{rank:02d}.json')
    assert len(items)==200 and len({x['key'] for x in items})==200
    for spec in p['datasets'].values():
        assert sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
    arms=[];identity=None
    for step in [6000,8000]:
        value,audit=export_checkpoint(work,step)
        if identity is None:identity=value['identity']
        if identity!=value['identity']:raise ValueError('Whisper checkpoint architectures differ')
        arms.append(dict(name=f'whisper_av_{step}',kind='whisper_av',step=step,
            checkpoint=audit['inference_checkpoint'],checkpoint_sha256=audit['inference_sha256'],
            original_checkpoint=audit['original_checkpoint'],original_sha256=audit['original_sha256'],
            planner=p['planner']))
        del value
    from p2_whisper_av.cache import model_contract,feature_code_identity
    encoder=model_contract(identity['experiment']['whisper']['model_path'])
    training_encoder=read(Path(identity['experiment']['whisper_cache'])/'protocol.json')['encoder']
    if encoder!=training_encoder:raise ValueError('Whisper encoder differs from training cache')
    p.update(schema='seamless200-whisper-supplement-v1',arms=arms,workers=28,total_outputs=400,
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),
        metrics_decoded_gt_cache=str(BASE/'decoded_gt_cache'),metrics_gt_cache=str(BASE/'gt_cache'),
        metrics_use_released_inference_nodes=False,metrics_workers_per_gpu=4,
        whisper_encoder=encoder,whisper_batch=16,feature_code=feature_code_identity(),checkpoint_identity=identity,
        worker_cli=str(Path(__file__).resolve()),conditioning='unchanged base Qwen H/Gemma plus speaker-only Whisper',
        sampling='original ten-arm p2_listen.runtime.guided_sample; FP32 video noise; native audio noise; unchanged Euler CFG',
        authorized_scope='Wait until authorized28 GPUs idle; do not kill other tasks; node8 GPU0..3 only',
        execution_order='speaker Whisper cache -> both checkpoints render -> native decode -> unchanged metrics -> merge -> publish',
        comparison='6000 vs AV6000 matched steps; 8000 progression reference, not AV8927 equal-step comparison')
    p['execution_code']={str(path):sha(path) for path in [Path(__file__).resolve(),
        ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/runtime.py',
        ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'scripts/publish_p2_seamless200_whisper.py']}
    link(work/'conditions',BASE/'base_standard/conditions')
    for rank in range(28):link(work/'negative'/f'{rank:02d}',BASE/'base_standard/negative'/f'{rank%24:02d}')
    for rank,shard in enumerate(partition(items,28)):save(work/'shards'/f'{rank:02d}.json',shard)
    for x in items:
        from p2_testsets_eval.runtime import ready
        folder=work/'conditions'/x['key']
        assert ready(folder/'qwen') and ready(folder/'text')
        assert read(folder/'qwen/plan.json')['adapter'] is None
        assert (folder/'first_frame.png').is_file()
    save(work/'protocol.json',p)
    save(work/'CPU_PRECHECK.json',dict(passed=True,samples=200,outputs=400,workers=28,
        references=p['datasets'],checkpoint_exports='all tensors bitwise equal',
        code_identity_matched=True,base_conditions_reused=200,gt_cache_reused=True,
        gpu_end_to_end='pending idle GPUs; not yet run',at=time.time()))
    return p


def idle_snapshot(cfg):
    sys.path.insert(0,str(ROOT/'scripts'));from listen_cluster import transport
    def one(node):
        try:
            report=subprocess.check_output(transport(cfg,node,['nvidia-smi',
                '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),
                text=True,timeout=25)
            values={int(parts[0]):dict(used_mib=float(parts[1]),total_mib=float(parts[2]),util=float(parts[3]))
                    for line in report.splitlines() if (parts:=line.split(','))}
            rows=[dict(gpu=g,**values[g]) for g in node['gpus']]
            return dict(node=node['name'],gpus=rows,idle=all(r['used_mib']<2048 and r['util']<10 for r in rows))
        except Exception as e:return dict(node=node['name'],idle=False,error=repr(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(one,cfg['nodes']))
    return dict(at=time.time(),all_idle=len(reports)==4 and all(r['idle'] for r in reports),nodes=reports)


def queue(work):
    work.mkdir(parents=True,exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=prepare(work)
        from p2_testsets_eval.cluster import run_cluster,bind,state,stopped
        if (work/'COMPLETE.json').exists():
            publish(work)
            return
        # Require two successive quiet polls, and recheck before each new GPU stage.
        def wait_idle():
            quiet=0
            while quiet<2:
                if stopped(work):state(work,'stopped');return False
                snap=idle_snapshot(read(p['cluster']));save(work/'resources.json',snap)
                quiet=quiet+1 if snap['all_idle'] else 0
                state(work,'waiting_authorized28_gpu_release',resources=snap,consecutive_idle_polls=quiet)
                if quiet<2:time.sleep(30)
            return True
        try:
            for stage in ['cache','render','decode']:
                if (work/'stages'/(stage+'.json')).exists():continue
                if not wait_idle():return
                run_cluster(work,stage)
            # Smoke metrics use two actual outputs from the completed generation.
            save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',arms=[a['name'] for a in p['arms']],
                 samples=200,outputs=400,at=time.time()))
            bind(work)
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time()))
            state(work,'generation_complete_starting_metrics')
            from p2_testsets_eval.metrics import queue as score
            score(work)
            # Preserve ten-arm results; publish the supplement as separate linked report.
            if (work/'COMPLETE.json').exists():publish(work)
        except Exception as e:
            state(work,'failed',error=repr(e));raise


def publish(work):
    p=read(work/'protocol.json');results=read(work/'RESULTS.json')
    report=['# Whisper speaker cross-attention补测','',
        '同一Seamless200、原Qwen规划文本与H、512×512/25fps、seed137、30步Euler、CFG3/7。',
        '6000步与AV6000对照；8000步是训练进展参考。原十组结果保持不变。','',
        '| 配置 | PSNR↑ | SSIM↑ | FID↓ | FVD↓ | LPIPS↓ | Emo-Acc↑ | ΔResp↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,s in results.items():
        m=s['metrics'];report.append('| '+key+' | '+' | '.join(str(m.get(k)) for k in ['psnr','ssim','fid','fvd','lpips','emo_acc','delta_resp'])+' |')
    (work/'RESULTS.md').write_text('\n'.join(report)+'\n')
    preview=ROOT/'data_previews/listen_benchmark_review/p2_seamless200_20261105'
    link(preview/'whisper_supplement',work)
    from p2_testsets_eval.protocol import METRIC_PYTHON
    from p2_testsets_eval.cluster import environment
    with (work/'publish.log').open('a') as log:
        subprocess.run([str(METRIC_PYTHON),str(ROOT/'scripts/publish_p2_seamless200_whisper.py'),
            '--work',str(work)],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    save(work/'PUBLISHED.json',dict(results=str(work/'RESULTS.json'),at=time.time(),
        url='http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/whisper_supplement/RESULTS.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','queue','worker','resources','publish'])
    parser.add_argument('--work',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['cache','render','decode']);parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--limit',type=int);parser.add_argument('--arms',nargs='+')
    a=parser.parse_args()
    if a.action=='worker':
        from p2_testsets_eval.whisper_runtime import worker
        worker(a.work,a.stage,a.rank,a.limit,a.arms)
    elif a.action=='resources':print(json.dumps(idle_snapshot(read(a.work/'cluster.json')),indent=2))
    elif a.action=='prepare':prepare(a.work)
    elif a.action=='publish':publish(a.work)
    else:queue(a.work)

if __name__=='__main__':main()
