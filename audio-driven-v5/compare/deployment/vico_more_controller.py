"""One-shot approved baseline workflow, with receipts and resource ownership guards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from vico_more_common import ROOT, WORK, read, save, sha, remote, launch_node11

PY=ROOT/'compare/.envs/vico_more/bin/python'
MPY=ROOT/'compare/.envs/vico_eval/bin/python'
SCRIPTS=ROOT/'compare/deployment'
FREEZE=1794096000
DELIVERY=1794110400


def execute_cpu(script,args):
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    subprocess.run([str(PY),str(SCRIPTS/script),*args],cwd=ROOT,env=env,check=True)


def ensure_job(name,script,args,gpu=7,python=PY):
    """Restart only this experiment's failed resumable stage, at most twice."""
    for attempt in range(3):
        key=name+(f'_retry{attempt}' if attempt else '')
        d=WORK/'jobs'/key
        if (d/'exit.json').exists():
            result=read(d/'exit.json')
            if result['returncode']==0:return 'done'
            continue
        if (d/'running.json').exists():
            runtime=read(d/'running.json')
            # Check the actual remote worker too: losing its supervisor must not
            # start a second copy of a still-running training/inference process.
            code=('from pathlib import Path; '
                f'w=Path("/proc",{str(runtime["pid"])!r},"cmdline"); '
                f's=Path("/proc",{str(runtime["supervisor_pid"])!r},"cmdline"); '
                f'worker=w.exists() and {script.encode()!r} in w.read_bytes(); '
                f'supervisor=s.exists() and b"run.lock" in s.read_bytes() and {str(d).encode()!r} in s.read_bytes(); '
                'print("alive" if worker or supervisor else "dead")')
            answer=remote(['python','-c',code]).strip()
            if answer=='alive':return 'running'
            save(d/'exit.json',dict(returncode=-999,reason='Both owned worker and supervisor exited; resume from receipts'))
            continue
        # A prior launched supervisor may be importing the shared filesystem.
        if (d/'submitted.json').exists() and time.time()-read(d/'submitted.json')['at']<120:
            return 'starting'
        assert gpu==7,'User allocated GPU7 only'
        needs_gpu=not (script=='vico_more_motion.py' and args[-1]=='score')
        if needs_gpu:
            observation=remote(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']).strip().split(',')
            from vico_more_common import GPU_UUID
            assert observation[0].strip()==GPU_UUID
            free=float(observation[1]);save(WORK/'GPU_RESOURCE_OBSERVATION.json',dict(at=time.time(),uuid=GPU_UUID,free_mib=free,
                required_free_gib=32,evict_existing_processes=False))
            if free<32*1024:return 'waiting for GPU7 memory headroom'
        pid=launch_node11(key,python,SCRIPTS/script,args,gpu=gpu).strip()
        save(d/'submitted.json',dict(at=time.time(),supervisor_pid=pid,gpu=gpu,args=args))
        return 'starting'
    raise RuntimeError(f'{name} failed three times; inspect owned job logs')


def released_gpu():
    parent=read(ROOT/'experiments/p2_current500h_av_only31_20261107/status.json')
    if not parent['complete']:return None
    # A completed status alone is insufficient: require the physical GPU's
    # compute processes to have actually exited. Never kill or evict a process.
    code='''import subprocess,json
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
busy={a.split(',')[0].strip() for a in apps if ',' in a}
result=[]
for line in rows:
 i,uid,mem=[x.strip() for x in line.split(',')]
 if i!='7' and uid not in busy and float(mem)>130*1024:result.append(dict(gpu=int(i),uuid=uid,free_mib=float(mem)))
print(json.dumps(result))
'''
    rows=json.loads(remote(['python','-c',code]))
    if not rows:return None
    picked=rows[0]
    path=WORK/'released_resources.json'
    current=read(path) if path.exists() else {}
    current[str(picked['gpu'])]=dict(uuid=picked['uuid'],claimed_at=time.time(),
        no_compute_process_at_claim=True,previous_training_step=parent['step'],
        previous_training_complete=True,free_mib=picked['free_mib'])
    save(path,current)
    return picked['gpu']


def method_step(method,gpu,state):
    out=WORK/method;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return 'complete'
    if (out/'NOT_SCHEDULED.json').exists():return 'not_scheduled'
    if (out/'INCOMPLETE.json').exists():return 'incomplete'
    compile_loss=state.get('compiled_loss',False)
    model_freeze=read(out/'RUN_BUDGET.json')['model_freeze_unix'] if method=='rlhg' and (out/'RUN_BUDGET.json').exists() else FREEZE
    if not (out/'TRAIN_COMPLETE.json').exists():
        if time.time()>=model_freeze:
            save(out/'INCOMPLETE.json',dict(reason='Model freeze before complete official epoch budget',formal_result=False))
            return 'incomplete'
        if method=='rlhg':
            return 'train:'+ensure_job(method+'_train','vico_more_rlhg_budget.py',
                ['--compile-loss'] if compile_loss else [],gpu)
        args=[method,'train']+(['--compile-loss'] if compile_loss else [])
        return 'train:'+ensure_job(method+'_train','vico_more_train.py',args,gpu)
    stages=[('GENERATION_INPUT_AUDIT.json','generate','vico_more_generation_audit.py',[method],PY),
        ('RENDER_COMPLETE.json','render','vico_more_render.py',['render','--method',method],PY),
        ('EXTRACTION_COMPLETE.json','extract','vico_more_motion.py',[method,'extract'],PY),
        ('MOTION_RESULTS.json','motion_scores','vico_more_motion.py',[method,'score'],PY),
        ('IMAGE_COMPLETE.json','image_scores','vico_more_image_metrics.py',['--work',str(out)],MPY)]
    for marker,name,script,args,python in stages:
        if not (out/marker).exists():
            return name+':'+ensure_job(method+'_'+name,script,args,gpu,python)
    # Require all receipts, not only process exit status.
    parity_name='TRAINING_ADAPTER_PARITY_COMPILED.json' if compile_loss else 'TRAINING_ADAPTER_PARITY.json'
    files=[out/n for n in ['TRAIN_PROTOCOL.json','TRAIN_COMPLETE.json','GENERATION_COMPLETE.json',
        'GENERATION_INPUT_AUDIT.json','RENDER_COMPLETE.json','EXTRACTION_COMPLETE.json','MOTION_RESULTS.json','RESULTS.json','IMAGE_COMPLETE.json',
        'SCORER_REUSE_AUDIT.json','RENDER_PARITY.json','DECA_EXTRACTION_PARITY.json','BFM_EXTRACTION_PARITY.json',
        'LOADER_PARITY.json',parity_name,'DEVICE.json']]
    if method=='rlhg':files += [out/n for n in ['RUN_BUDGET.json','FULL_BUDGET_DECISION.json','EFFECTIVE_EXECUTION.json']]
    save(out/'COMPLETE.json',dict(success=True,method=method,pairs=43,formal_training_budget_complete=True,
        files={str(p):sha(p) for p in files},at=time.time()))
    return 'complete'


def finalize(stages,features):
    primary=stages['listenformer']=='complete' and read(WORK/'dualtalk/COMPLETE.json')['success']
    save(WORK/'FINISHED.json',dict(at=time.time(),methods=stages,
        primary_methods_complete=primary,
        all_planned_formal_methods_complete=all(stages[m]=='complete' for m in ['rlhg','listenformer']),
        pch=read(WORK/'PCH_FEASIBILITY.json')['status'],pch_feasibility_sha256=sha(WORK/'PCH_FEASIBILITY.json'),
        no_ovi_no_glare=True))
    save(WORK/'status.json',dict(at=time.time(),stage='complete_primary' if primary else 'finished_partial',
        features_complete=features,features_total=794,methods=stages,dualtalk_complete=True,
        renderer_verified=True,deadline_utc='2026-11-08T04:00:00Z'))
    if not primary:return
    page=ROOT/'data_previews/listen_benchmark_review/vico_more_20261107'
    assert read(WORK/'BROWSER_AUDIT.json')['data_sha256']==sha(page/'data.json')
    assert read(WORK/'LINK_AUDIT.json')['data_sha256']==sha(page/'data.json')
    assert read(WORK/'NUMERICAL_AUDIT.json')['comparison_sha256']==sha(WORK/'COMPARISON.json')
    assert read(WORK/'NATIVE_CURVES_AUDIT.json')['data_sha256']==sha(page/'data.json')
    assert read(WORK/'EXPORT_AUDIT.json')['comparison_sha256']==sha(WORK/'COMPARISON.json')
    for p in [ROOT/'README.md',ROOT.parent/'OmniResponse/evaluation/README.md']:
        text=p.read_text();lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('**ViCo新增基线执行中（2026-11-07'):
                relative='experiments/p2_vico_additional_baselines_20261107/README.md' if p==ROOT/'README.md' else '../../audio-driven-v5/experiments/p2_vico_additional_baselines_20261107/README.md'
                lines[i]='**ViCo优先新增基线已完成（2026-11-07）：** 按用户确认仅用node11 GPU7，DualTalk官方核心前向对照及43条动作迁移完成，ListenFormer完整200轮、本地ViCo43视频生成、统一画质与BFM／DECA评分及数值／浏览器验收通过。DualTalk几何对应未完全核实，单列代理；未完成方法及原因单独披露。现有31卡、原结果保持原样。[结果、协议与验收]('+relative+') · [检查入口](http://localhost:8876/listen_benchmark_review/vico_more_20261107/)。'
                break
        p.write_text('\n'.join(lines)+'\n')
    files=[WORK/n for n in ['COMPARISON.json','REVIEW.json','PAPER_TABLES.json','NUMERICAL_AUDIT.json',
        'LINK_AUDIT.json','BROWSER_AUDIT.json','NATIVE_CURVES_AUDIT.json','EXPORT_AUDIT.json','TRAINING_SELECTION.json',
        'SPLIT_ISOLATION_AUDIT.json','PCH_FEASIBILITY.json',
        'METHOD_COVERAGE.json','RESOURCE_SHARING_POLICY.json',
        'RESULTS.md','README.md','protocol.json','RESOURCE_PRIORITY.json',
        'DESIGN.md','FINISHED.json','status.json']]
    files += [page/'index.html',page/'data.json']
    files += [Path(p) for p in read(WORK/'REVIEW.json')['native_motion_files']]
    save(WORK/'COMPLETE.json',dict(success=True,scope='User priority DualTalk+ListenFormer, GPU7 only',
        primary_methods_complete=True,all_optional_methods_complete=stages['rlhg']=='complete',
        files={str(p):sha(p) for p in files},at=time.time()))


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'COMPLETE.json').exists():
        completed=read(WORK/'COMPLETE.json');assert completed['success']
        for p,h in completed['files'].items():assert sha(p)==h,p
        print('Already complete; sealed files verified; no job restarted',flush=True)
        return
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=WORK/'CONTROLLER_STATE.json'
        state=read(state_path) if state_path.exists() else dict(methods={},features_ready=False)
        save(WORK/'controller_runtime.json',dict(pid=os.getpid(),started=time.time(),deadline_utc='2026-11-08T04:00:00Z'))
        while True:
            try:
                stages={}
                if not state['features_ready']:
                    ready=all((WORK/'features'/f'BFM_COMPLETE_{i}.json').exists() for i in range(4))
                    if not ready:
                        for i in range(4):
                            if (WORK/'features'/f'BFM_COMPLETE_{i}.json').exists():continue
                            # Existing extraction jobs were launched before this controller.
                            stages['bfm_'+str(i)]=ensure_job('bfm_'+str(i),'vico_more_features.py',
                                ['bfm','--shard',str(i),'--shards','4'])
                    if ready and (WORK/'features/AUDIO_COMPLETE.json').exists():
                        if not (WORK/'features/COMPLETE.json').exists():execute_cpu('vico_more_features.py',['stats'])
                        if not (WORK/'features/TEST_INPUTS.json').exists():execute_cpu('vico_more_features.py',['test_inputs'])
                        state['features_ready']=True
                if state['features_ready']:
                    # User's latest explicit priority: GPU7 only, DualTalk + LF.
                    # Do not claim cards from the 31-GPU AV+Feature continuation.
                    if 'listenformer' not in state['methods']:
                        benchmark=read(WORK/'listenformer/BENCHMARK.json')
                        hours=benchmark['estimated_training_hours']*1.2
                        if time.time()+hours*3600>=FREEZE:
                            save(WORK/'listenformer/NOT_SCHEDULED.json',dict(reason='Remaining time cannot fit full200epoch budget',
                                estimated_hours_with_margin=hours,formal_result=False))
                            state['methods']['listenformer']=dict(gpu=7,compiled_loss=False)
                        else:state['methods']['listenformer']=dict(gpu=7,compiled_loss=False)
                    lf_done=(WORK/'listenformer/COMPLETE.json').exists()
                    lf_unavailable=any((WORK/'listenformer'/n).exists() for n in ['FAILED.json','NOT_SCHEDULED.json','INCOMPLETE.json'])
                    if 'rlhg' not in state['methods']:
                        if lf_done or lf_unavailable:
                            # Both candidates use all351 train sources and the unchanged
                            # full500epoch schedule. Reserve actual measured LF pipeline
                            # time x3, and never less than90min before user delivery.
                            status=ensure_job('rlhg_full_feature_benchmark','vico_more_train.py',['rlhg','benchmark'])
                            stages['rlhg_benchmark']=status
                            if status=='done':
                                compiled_status=ensure_job('rlhg_full_compiled_benchmark','vico_more_train.py',
                                    ['rlhg','benchmark','--compile-loss'])
                                stages['rlhg_compiled_benchmark']=compiled_status
                                if compiled_status=='done':
                                    benchmark=read(WORK/'rlhg/BENCHMARK.json')
                                    compiled=read(WORK/'rlhg/BENCHMARK_COMPILED.json')
                                    for item in [benchmark,compiled]:
                                        assert item['real_clips']==351 and item['full_steps']==3474000
                                        assert item['finite_gradients'] and item['future_target_perturbation_max_abs']==0
                                    if lf_done:
                                        started=read(WORK/'jobs/listenformer_generate/running.json')['started']
                                        measured=read(WORK/'listenformer/COMPLETE.json')['at']-started
                                    else:measured=4800  # fallback original4hour reserve
                                    reserve=max(5400,3*measured)
                                    cutoff=DELIVERY-reserve
                                    margin=1.15
                                    candidates=[(benchmark['estimated_training_hours']*margin,False),
                                        (compiled['estimated_training_hours']*margin,True)]
                                    hours,chosen=min(candidates)
                                    now=time.time();eligible=now+hours*3600<cutoff
                                    budget=dict(eligible=eligible,selected_compiled_loss=chosen,
                                        epochs=500,gpu=7,delivery_unix=DELIVERY,model_freeze_unix=cutoff,
                                        postprocessing_reserve_s=reserve,measured_lf_postprocessing_s=measured,
                                        reserve_multiplier=3,safety_margin=margin,
                                        estimated_training_hours_with_margin=hours,at=now,
                                        policy='Full500epochs only; reserve max90min or3x measured LF postprocessing; user04UTC deadline unchanged')
                                    save(WORK/'rlhg/RUN_BUDGET.json',budget)
                                    save(WORK/'rlhg/FULL_BUDGET_DECISION.json',dict(eager=benchmark,compiled=compiled,
                                        safety_margin=margin,compiled_chosen=chosen,eligible=eligible,at=now,
                                        model_freeze_unix=cutoff,delivery_unix=DELIVERY,
                                        postprocessing_reserve_s=reserve,run_budget_sha256=sha(WORK/'rlhg/RUN_BUDGET.json')))
                                    decision=read(WORK/'OPTIMIZATION_DECISION.json')
                                    decision['rlhg_full_feature_compiled_chosen']=chosen
                                    decision['rlhg_full_budget_decision_sha256']=sha(WORK/'rlhg/FULL_BUDGET_DECISION.json')
                                    decision['selected']='ListenFormer eager; RLHG '+('audited FP32 compiled loss' if chosen else 'eager')
                                    save(WORK/'OPTIMIZATION_DECISION.json',decision)
                                    if not eligible:
                                        save(WORK/'rlhg/NOT_SCHEDULED.json',dict(reason='Both measured full500epoch ETAs plus15% margin cross measured delivery cutoff',
                                            eager_hours_with_margin=candidates[0][0],compiled_hours_with_margin=candidates[1][0],
                                            model_freeze_unix=cutoff,formal_result=False))
                                    state['methods']['rlhg']=dict(gpu=7,compiled_loss=chosen)
                        else:stages['rlhg']='waiting until priority ListenFormer completes'
                    for method,spec in state['methods'].items():
                        if (WORK/method/'FAILED.json').exists():
                            stages[method]='failed';continue
                        try:stages[method]=method_step(method,spec['gpu'],spec)
                        except Exception as e:
                            if 'failed three times' not in str(e):raise
                            save(WORK/method/'FAILED.json',dict(reason=str(e),at=time.time(),formal_result=False))
                            stages[method]='failed'
                save(state_path,state)
                # Small explicit markers only; never scan full experiment trees.
                feature_count=sum(1 for t in read(WORK/'features/protocol.json')['tasks'] if t['split']!='test'
                    for role in ['speaker','listener'] if (WORK/'features'/t['split']/t['id']/(role+'.json')).exists())
                save(WORK/'status.json',dict(at=time.time(),stage='running',features_complete=feature_count,
                    features_total=794,methods=stages,deadline_utc='2026-11-08T04:00:00Z',
                    training_progress={m:read(WORK/m/'status.json') for m in ['rlhg','listenformer']
                        if (WORK/m/'status.json').exists()},
                    dualtalk_complete=(WORK/'dualtalk/COMPLETE.json').exists(),
                    renderer_verified=(WORK/'renderer/SMOKE.json').exists()))
                # Refresh the review/results only when method completion changes.
                signature=tuple((m,(WORK/m/'COMPLETE.json').exists()) for m in ['rlhg','listenformer'])
                if state.get('published_signature')!=str(signature):
                    execute_cpu('vico_more_report.py',[])
                    state['published_signature']=str(signature);save(state_path,state)
                terminal=['complete','not_scheduled','incomplete','failed']
                if all(stages.get(m) in terminal for m in ['rlhg','listenformer']):
                    execute_cpu('vico_more_report.py',[])
                    execute_cpu('vico_more_check.py',['numerical'])
                    execute_cpu('vico_more_check.py',['native_curves'])
                    execute_cpu('vico_more_check.py',['exports'])
                    execute_cpu('vico_more_check.py',['links'])
                    subprocess.run([str(ROOT/'compare/.envs/avatarforcing/bin/python'),
                        str(SCRIPTS/'vico_more_check.py'),'browser'],cwd=ROOT,check=True)
                    finalize(stages,feature_count)
                    return
                if time.time()>DELIVERY:
                    execute_cpu('vico_more_report.py',[])
                    save(WORK/'DEADLINE_STATUS.json',dict(at=time.time(),methods=stages,finished=False))
                    return
            except Exception as e:
                save(WORK/'controller_error.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc()))
                print(traceback.format_exc(),flush=True)
                # Do not claim success or restart an exhausted stage indefinitely.
                if 'failed three times' in str(e):
                    if str(e).startswith('rlhg_'):
                        save(WORK/'rlhg/FAILED.json',dict(reason=str(e),at=time.time(),formal_result=False))
                        state['methods']['rlhg']=dict(gpu=7,compiled_loss=False);save(state_path,state)
                    else:raise
            time.sleep(30)


if __name__=='__main__':run()
