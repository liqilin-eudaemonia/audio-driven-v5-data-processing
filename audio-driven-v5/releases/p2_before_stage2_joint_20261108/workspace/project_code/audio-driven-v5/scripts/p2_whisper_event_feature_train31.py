"""Explicit 8+8+8+7 training; immutable32-card tests stay unchanged."""
import argparse,copy,csv,fcntl,json,math,os,random,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
DEFAULT=ROOT/'experiments/p2_j_whisper_event_feature_train31_20261105'


def layout(cluster):
    if [n['name'] for n in cluster['nodes']]!=['node17','node12','node4','node8']:
        raise ValueError('Unexpected physical node order')
    result=[];offset=0
    for i,node in enumerate(cluster['nodes']):
        expected=list(range(7 if i==3 else 8))
        if node['gpus']!=expected:raise ValueError('Reserve node8 GPU7; use exactly8+8+8+7')
        result.append(dict(node=node['name'],gpus=expected,rank_offset=offset,ranks=list(range(offset,offset+len(expected)))))
        offset+=len(expected)
    if offset!=31:raise ValueError('Expected31 ranks')
    return result


def approved_config(c,parent):
    from p2_whisper_event_feature.config import validate
    validate(c);meta=c['world31_execution'];t=c['training'];old=parent['training']
    expected=copy.deepcopy(parent);expected['output']=c['output'];expected['world31_execution']=meta
    expected['training'].update(global_batch=31,steps=math.ceil(meta['samples']/31))
    expected['duration_sampling']=c['duration_sampling']
    spec=c['duration_sampling']
    actual_samples=len(read(Path(parent['j_cache'])/'manifest.json')['splits']['train'])
    if (c!=expected or meta['samples']!=actual_samples or meta['kind']!='fresh31-same-initialization-v1' or meta['reserved_node']!='node8'
            or meta['reserved_gpu']!=7 or meta['world']!=31 or old['global_batch']!=32
            or (spec['policy'],spec['window'],spec['start_step'])!=('duration',496,2)
            or spec['sample_list_sha256']!=parent['duration_sampling']['sample_list_sha256']
            or Path(c['output']).resolve()==Path(parent['output']).resolve()):
        raise ValueError('Unapproved31-card experiment change')
    return c


def install(c):
    """Versioned runtime adapter; never edit the existing checkpointed modules."""
    from p2_whisper_event_feature import config,training
    if getattr(config,'_world31_installed',False):return
    native_baseline=config.validate_baseline;native_source=config.source_identity
    def baseline(value,reference):
        meta=value['world31_execution'];parent=read(meta['parent_config'])
        if sha(meta['parent_config'])!=meta['parent_config_sha256']:raise ValueError('Frozen parent config changed')
        approved_config(value,parent);native_baseline(parent,reference)
        return value
    def source():
        value=native_source();value[str(SELF)]=sha(SELF);return value
    config.validate_baseline=training.validate_baseline=baseline
    config.source_identity=training.source_identity=source
    config._world31_installed=True


def prepare(out):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_event_feature.config import load,source_identity,validate_baseline
    from p2_whisper_event_feature.data import data_identity
    from p2_j_duration_execution import make_pairs,Schedule
    base=load(ROOT/'configs/p2_j_qwen_whisper_event_feature_20261105.json')
    validate_baseline(base,read(base['baseline_config']));old=Path(base['output']);oldrun=read(old/'run.json')
    oldpre=read(old/'preflight.json')
    if (oldrun['experiment']!=base or oldrun['source']!=source_identity()
            or oldpre['source']!=source_identity() or oldpre['data_identity']!=data_identity(base)):
        raise ValueError('32-card reference/source/full preflight changed')
    if out.exists():raise FileExistsError('Use an independent31-card output')
    out.mkdir(parents=True);atomic_json(out/'parent_config.json',base,immutable=True)
    path=Path(base['j_cache'])/'sample_list.csv'
    with path.open() as f:raw=list(csv.DictReader(f))
    rows=[dict(requested_duration_s=float(r['duration_s'])) for r in raw];size=len(rows);steps=math.ceil(size/31)
    pairs=make_pairs(rows,steps,31,base['training']['seed'],2,'duration',496)
    parent_pairs=Schedule(base).pairs
    real={o:i for i,o in pairs if o<size};old_real={o:i for i,o in parent_pairs if o<size}
    if len([o for _,o in pairs if o<size])!=size or real!=old_real or set(real)!=set(range(size)):
        raise ValueError('Real sample/noise ordinals lost or duplicated')
    plan=out/'duration.schedule.json'
    atomic_json(plan,dict(pairs=pairs,sample_list_sha256=sha(path)),immutable=True)
    c=copy.deepcopy(base);c['output']=str(out);c['training'].update(global_batch=31,steps=steps)
    c['duration_sampling'].update(window=496,start_step=2,manifest=str(plan),sha256=sha(plan))
    c['world31_execution']=dict(kind='fresh31-same-initialization-v1',world=31,samples=size,
        parent_config=str(out/'parent_config.json'),parent_config_sha256=sha(out/'parent_config.json'),
        reserved_node='node8',reserved_gpu=7)
    approved_config(c,base);Schedule(c)
    atomic_json(out/'config.json',c,immutable=True)
    install(c)
    from p2_whisper_event_feature.config import source_identity as current_source
    pre=copy.deepcopy(oldpre);pre['source']=current_source()
    pre['reused_full_preflight']=dict(path=str(old/'preflight.json'),sha256=sha(old/'preflight.json'),
        same_data_identity=True,reason='Only effective batch/rank layout/duration grouping changes; frozen media/H/text/labels/teacher/Whisper unchanged')
    atomic_json(out/'preflight.json',pre,immutable=True)
    cluster=read(ROOT/'configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json')
    cluster['nodes'][-1]['gpus']=list(range(7));cluster['master_port']=29877;cluster['training_script']=str(SELF)
    mapping=layout(cluster);atomic_json(out/'cluster.json',cluster,immutable=True)
    atomic_json(out/'PLAN_ACCEPTANCE.json',dict(success=True,samples=size,steps=steps,global_batch=31,
        exact_real_membership_and_noise_ordinals=True,padding=len(pairs)-size,initialization='same pretrained weights/seed; fresh start; prior18steps are tests only',
        parent_epoch_padding=len(parent_pairs)-size,sample_grouping_changed=True,trajectory_identical=False,
        equivalent_32gpu6000_step=math.ceil(6000*32/31),rank_layout=mapping),immutable=True)
    atomic_json(out/'RESERVED_GPU.json',dict(node='node8',host=cluster['nodes'][-1]['host'],gpu=7,training_visible_gpus=list(range(7))),immutable=True)
    return c


def commands(cluster,config,resume=None,attempt=None):
    mapping=layout(cluster);result=[]
    for i,node in enumerate(cluster['nodes']):
        command=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,node['gpus'])),'PYTHONUNBUFFERED=1',
            'TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cluster['network'].items()],
            cluster['python'],str(SELF),'node','--config',str(config),'--node-index',str(i)]
        if resume:command+=['--resume',str(resume)]
        if attempt:command+=['--attempt',str(attempt)]
        result.append(command)
    return result


def node(c,index,resume,attempt):
    cluster=read(Path(c['output'])/'cluster.json');mapping=layout(cluster);entry=mapping[index]
    mask=','.join(map(str,entry['gpus']))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=mask:raise ValueError('Physical GPU visibility differs, reservedGPU must stay hidden')
    out=Path(c['output']);logs=attempt or out/'queue/manual';logs.mkdir(parents=True,exist_ok=True)
    processes=[];streams=[]
    try:
        for local,rank in enumerate(entry['ranks']):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='31',LOCAL_WORLD_SIZE=str(len(entry['gpus'])),
                     MASTER_ADDR=cluster['nodes'][0]['host'],MASTER_PORT=str(cluster['master_port']))
            command=[cluster['python'],str(SELF),'train','--config',str(out/'config.json')]
            if resume:command+=['--resume',str(resume)]
            stream=(logs/f'rank{rank:02d}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT))
        atomic_json(logs/(entry['node']+'_workers.json'),dict(node=entry['node'],gpus=entry['gpus'],ranks=entry['ranks'],pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Local distributed rank failed')
            time.sleep(2)
        if any(p.returncode for p in processes):raise RuntimeError('Distributed training failed')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()


def verify_row(row):
    keys=['av','event','feature_video','feature_audio','weighted_total']
    if not all(math.isfinite(row[k]) for k in keys):raise ValueError('Nonfinite combined objective')
    expected=row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
    if abs(expected-row['weighted_total'])>1e-8:raise ValueError('Incorrect combined loss weights')
    for name in ['connector','events','features','text_lora','whisper']:
        if not math.isfinite(row['grad_norm'][name]) or row['grad_norm'][name]<=0:raise ValueError('Missing/nonfinite gradient: '+name)


def selected_resources(cluster):
    from p2_whisper_event_feature_test32 import resources
    value=resources(cluster);nodes={n['node']:n for n in value['nodes']}
    for node in cluster['nodes']:
        report=nodes[node['name']];selected=[g for g in report.get('gpus',[]) if g['gpu'] in node['gpus']]
        report['selected_idle']=len(selected)==len(node['gpus']) and all(g['used_mib']<2048 and g['util']<10 for g in selected)
    value['selected_all_idle']=all(n.get('selected_idle') for n in value['nodes'])
    return value


def run(out):
    from listen_cluster import launch
    from p2_speaker_whisper_after_annotation import process_record
    c=read(out/'config.json');install(c);cluster=read(out/'cluster.json');q=out/'queue';q.mkdir(exist_ok=True)
    with (q/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(q/'controller.json',process_record())
        def state(stage,**kw):
            value=dict(stage=stage,at=time.time(),**kw);atomic_json(q/'state.json',value);print(json.dumps(value),flush=True)
        for number in range(1,6):
            while True:
                if (q/'STOP').exists() or (out/'STOP').exists():state('stopped_before_launch');return
                snapshot=selected_resources(cluster);atomic_json(q/'resources.json',snapshot)
                if snapshot['selected_all_idle']:break
                state('waiting_selected31_gpus');time.sleep(15)
            from p2_whisper_event_feature.config import source_identity
            if read(out/'preflight.json')['source']!=source_identity():raise ValueError('Source changed after acceptance')
            markers=sorted(out.glob('step_*.complete.json'));resume=None
            if markers:
                marker=markers[-1];resume=marker.with_suffix('').with_suffix('.pt')
                if sha(resume)!=read(marker)['sha256']:raise ValueError('Latest complete checkpoint corrupted')
            logs=q/('attempt_'+str(number)+'_'+str(time.time_ns()))
            cmds=commands(cluster,out/'config.json',resume,logs/'ranks')
            atomic_json(q/'commands.json',dict(world=31,commands=cmds,resume=str(resume) if resume else None))
            state('training31',attempt=number,resume=str(resume) if resume else None,budget=c['training']['steps'],logs=str(logs))
            done=threading.Event();cancel=threading.Event();started=time.time()
            def monitor():
                while not done.wait(20):
                    try:
                        if (q/'STOP').exists():(out/'STOP').touch()
                        snapshot=selected_resources(cluster);atomic_json(q/'resources.json',snapshot)
                        with (q/'resources.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                        p=out/'metrics.jsonl';stamp=p.stat().st_mtime if p.exists() else started
                        if time.time()-stamp>(900 if p.exists() else 1800):raise TimeoutError('Training has stalled')
                        if p.exists():
                            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-20000));lines=f.read().splitlines()
                            try:row=json.loads(lines[-1])
                            except (ValueError,IndexError):continue
                            verify_row(row)
                            if row['step']>=5 and not (out/'STARTUP_ACCEPTANCE.json').exists():
                                atomic_json(out/'STARTUP_ACCEPTANCE.json',dict(success=True,world=31,step=row['step'],five_groups_receive_gradients=True,
                                    objective_correct=True,reserved_node='node8',reserved_gpu=7,sample=row,at=time.time()))
                    except Exception as exc:
                        atomic_json(q/'WATCH_ERROR.json',dict(error=repr(exc),at=time.time()));cancel.set();return
            watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
            try:launch(cluster,cmds,logs,cancel_event=cancel)
            except Exception as exc:
                state('retry_pending',attempt=number,error=repr(exc))
                if number==5:raise
                time.sleep(15);continue
            finally:done.set();watcher.join(timeout=45)
            final=read(out/'status.json')
            if final['complete'] and final['step']==c['training']['steps']:state('complete',**final);return
            if (out/'STOP').exists():state('stopped',**final);return
            raise RuntimeError('Workers ended before the epoch finished')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','node','train'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--config',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--node-index',type=int);p.add_argument('--attempt',type=Path);a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.output.resolve()),indent=2))
    elif a.command=='run':run(a.output.resolve())
    elif a.command=='node':node(read(a.config),a.node_index,a.resume,a.attempt)
    else:
        from p2_whisper_av.bootstrap import activate_ltx
        activate_ltx()
        from p2_whisper_event_feature.config import load
        from p2_whisper_event_feature.training import train
        c=load(a.config);install(c);print(json.dumps(train(c,a.resume)),flush=True)
