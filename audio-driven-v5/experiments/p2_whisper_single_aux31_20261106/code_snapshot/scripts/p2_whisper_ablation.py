"""Prepare or explicitly launch the two single-auxiliary 31-GPU Whisper arms."""
import argparse,copy,csv,json,os,sys
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_whisper_ablation.config import REFERENCE,load,validate_baseline,Schedule
from p2_whisper_ablation import VERSION


def config_path(arm):return ROOT/f'configs/p2_whisper_{arm}_20261106.json'


def emit():
    reference=read(REFERENCE);result=[]
    for arm in ['event','feature']:
        c=copy.deepcopy(reference)
        for key in ['whisper_aux_experiment','world31_execution']:c.pop(key,None)
        out=ROOT/f'experiments/p2_whisper_{arm}_20261106'
        c.update(id=f'guided_av_{arm}_s1',loss=f'av_{arm}',output=str(out),
            teacher_cache=reference['teacher_cache'] if arm=='feature' else None,
            event_head_training=arm=='event',baseline_config=str(REFERENCE),
            whisper_ablation_experiment=VERSION,
            ablation_reference=dict(config=str(REFERENCE),sha256=sha(REFERENCE),sampling='exact frozen full-plan prefix'))
        c['training']['steps']=8000;validate_baseline(c,reference)
        out.mkdir(exist_ok=True);atomic_json(config_path(arm),c,immutable=True)
        atomic_json(out/'config.json',c,immutable=True)
        cluster=read(REFERENCE.parent/'cluster.json');cluster['master_port']=29891 if arm=='event' else 29892
        cluster['training_script']=str(SELF);atomic_json(out/'cluster.json',cluster,immutable=True)
        result.append(dict(arm=arm,config=str(config_path(arm)),output=str(out),training_started=False))
    return result


def verify_cache(c):
    import torch
    from p2_whisper_ablation.data import Dataset,data_identity
    torch.set_num_threads(2);dataset=Dataset(c);plan=Schedule(c)
    with (Path(c['j_cache'])/'sample_list.csv').open() as f:rows=list(csv.DictReader(f))
    indices=list(dict.fromkeys([min(range(len(rows)),key=lambda i:float(rows[i]['duration_s'])),
        max(range(len(rows)),key=lambda i:float(rows[i]['duration_s'])),len(rows)//2]))
    samples=[]
    for index in indices:
        s=dataset.sample(index,'cpu');s['condition'].validate();s['whisper'].validate()
        known=sum(int(v['known'].sum()) for v in s['events'].values())
        if c['loss']=='av_event' and s['teacher'] is not None:raise ValueError('Event arm loaded teacher')
        if c['loss']=='av_feature' and (s['teacher'] is None or known):raise ValueError('Feature branch/label mismatch')
        samples.append(dict(sample_id=s['sample_id'],duration_s=float(rows[index]['duration_s']),
            teacher_loaded=s['teacher'] is not None,known_event_classes=known,
            whisper_shape=list(s['whisper'].hidden.shape),whisper_end_s=float(s['whisper'].bounds[0,0,-1,1])))
    report=dict(success=True,samples=samples,data_identity=data_identity(c),
        sample_noise_pairs=len(plan.pairs),same_reference_schedule_prefix=True,
        cuda_initialized=torch.cuda.is_initialized(),training_started=False)
    atomic_json(Path(c['output'])/'CPU_CACHE_CHECK.json',report);return report


def commands(c,resume=None,attempt=None,config=None):
    import p2_whisper_event_feature_train31 as launcher
    with patch.object(launcher,'SELF',SELF):
        return launcher.commands(read(Path(c['output'])/'cluster.json'),config or config_path(c['loss'][3:]),resume,attempt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['emit','inspect','preflight','verify-cache','commands','launch','node','train'])
    p.add_argument('--arm',choices=['event','feature']);p.add_argument('--config',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--logs',type=Path)
    p.add_argument('--node-index',type=int);p.add_argument('--attempt',type=Path)
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if a.command=='emit':print(json.dumps(emit(),indent=2));return
    path=a.config or (config_path(a.arm) if a.arm else None)
    if path is None:p.error('--arm or --config is required')
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx();c=load(path);validate_baseline(c,read(c['baseline_config']))
    if a.command=='inspect':
        from p2_whisper_ablation.execution import settings
        result=dict(id=c['id'],loss=c['loss'],active_coefficients=dict(av=1.,**({'event':.05} if c['loss']=='av_event' else {'video_feature':.025,'audio_feature':.025})),
            steps=8000,batch=31,save_every=500,execution=settings(c),teacher_cache=c['teacher_cache'],
            reserved_gpu='node8:7',requires_new_cache=False,training_started=False)
    elif a.command=='commands':result=commands(c,a.resume,a.attempt,path)
    elif a.command=='verify-cache':result=verify_cache(c)
    elif a.command=='preflight':
        from p2_whisper_ablation.training import preflight
        result=preflight(c)
    elif a.command=='node':
        import p2_whisper_event_feature_train31 as launcher
        with patch.object(launcher,'SELF',SELF):result=launcher.node(c,a.node_index,a.resume,a.attempt)
    elif a.command=='train':
        from p2_whisper_ablation.training import train
        result=train(c,a.resume,a.stop_after)
    else:
        if a.logs is None:p.error('launch requires --logs')
        import fcntl
        gate=ROOT/'experiments/p2_whisper_single_aux31_20261106';gate.mkdir(exist_ok=True)
        launch_lock=(gate/'launch.lock').open('a');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (Path(c['output'])/'STOP').exists():raise ValueError('User STOP exists')
        if (Path(c['output'])/'run.json').exists() and a.resume is None:raise ValueError('Explicit strict resume required')
        atomic_json(Path(c['output'])/'config.json',c,immutable=True)
        from p2_seamless200_whisper_eval import idle_snapshot
        cluster=read(Path(c['output'])/'cluster.json')
        snapshot=idle_snapshot(cluster);atomic_json(Path(c['output'])/'launch_resources.json',snapshot)
        if not snapshot['all_idle']:raise ValueError('Selected31 GPUs are occupied or unavailable; no eviction')
        from p2_whisper_ablation.training import preflight
        preflight(c)
        from listen_cluster import launch
        launch(cluster,commands(c,a.resume,a.logs/'ranks',path),a.logs)
        result=read(Path(c['output'])/'status.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
