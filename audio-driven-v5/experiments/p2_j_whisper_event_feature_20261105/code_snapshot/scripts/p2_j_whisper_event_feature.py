"""Independent AV+Event+Feature+speaker-Whisper experiment; no implicit training."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_whisper_event_feature.config import DEFAULT,load,validate_baseline
from p2_listen.contracts import read,atomic_json,sha
CLUSTER=ROOT/'configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json'


def commands(cluster,config,resume=None,stop_after=None):
    from p2_j_s1_queue import commands as native
    result=native(cluster,config,resume=resume,stop_after=stop_after)
    for command in result:command[command.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(Path(__file__).resolve())
    return result


def verify_cache(c,limit=3):
    import csv,torch
    from p2_whisper_event_feature.data import Dataset,data_identity
    from p2_whisper_event_feature.config import source_identity
    from p2_j_duration_execution import Schedule
    if type(limit) is not int or limit<1:raise ValueError('Expected a positive sample limit')
    torch.set_num_threads(2);dataset=Dataset(c);Schedule(c)
    with (Path(c['j_cache'])/'sample_list.csv').open() as f:rows=list(csv.DictReader(f))
    order=list(dict.fromkeys([0,min(range(len(rows)),key=lambda i:float(rows[i]['duration_s'])),
                             max(range(len(rows)),key=lambda i:float(rows[i]['duration_s'])),len(rows)//2]))
    order+= [i for i in range(len(rows)) if i not in order]
    result=[]
    for i in order[:limit]:
        value=dataset.sample(i,'cpu');speaker=value['whisper'].validate()
        value['condition'].validate()
        if value['teacher'] is None or value['condition'] is None or len(value['events'])!=len(dataset.names):
            raise ValueError('Missing combined cache component')
        frozen=not any(v.requires_grad for v in [value['condition'].hidden,speaker.hidden,
                      value['teacher']['video'],value['teacher']['audio']])
        if not frozen:raise ValueError('Frozen H/Whisper/teacher unexpectedly carries gradients')
        result.append(dict(sample_id=dataset.rows[i]['sample_id'],duration_s=float(rows[i]['duration_s']),
          teacher_shapes={m:list(value['teacher'][m].shape) for m in ['video','audio']},
          whisper_shape=list(speaker.hidden.shape),whisper_first_s=float(speaker.bounds[0,0,0,0]),
          whisper_last_s=float(speaker.bounds[0,0,-1,1]),event_classes=len(value['events']),
          teacher_and_hidden_frozen=frozen))
    report=dict(success=True,samples=result,total_training_samples=len(dataset.rows),data_identity=data_identity(c),
        source=source_identity(),duration_plan_verified=True,cuda_initialized=torch.cuda.is_initialized(),training_started=False)
    atomic_json(Path(c['output'])/'CPU_CACHE_CHECK.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inspect','verify-cache','preflight','commands','train','launch'])
    p.add_argument('--config',type=Path,default=DEFAULT);p.add_argument('--cluster',type=Path,default=CLUSTER)
    p.add_argument('--processes',type=int,default=8);p.add_argument('--limit',type=int,default=3)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--logs',type=Path)
    a=p.parse_args();c=load(a.config);validate_baseline(c,read(c['baseline_config']))
    if a.command=='inspect':
        from p2_whisper_av.execution import settings
        result=dict(experiment=c['id'],objective=c['loss_weights'],execution=settings(c),
          teacher_cache=c['teacher_cache'],speaker_whisper_cache=c['whisper_cache'],
          base_h_cache=c['j_cache'],steps=c['training']['steps'],global_batch=c['training']['global_batch'],
          save_every=c['training']['save_every'],training_started=False)
    elif a.command=='commands':result=commands(read(a.cluster),a.config,a.resume,a.stop_after)
    elif a.command=='verify-cache':result=verify_cache(c,a.limit)
    elif a.command=='preflight':
        from p2_whisper_event_feature.training import preflight
        result=preflight(c,a.processes)
    elif a.command=='train':
        from p2_whisper_event_feature.training import train
        result=train(c,a.resume,a.stop_after)
    else:
        from listen_cluster import launch
        from p2_whisper_event_feature.training import preflight
        if a.logs is None:p.error('launch requires a separate --logs directory')
        if (Path(c['output'])/'STOP').exists():raise ValueError('Explicit output STOP exists')
        if (Path(c['output'])/'run.json').exists() and not a.resume:raise ValueError('Explicit resume required')
        preflight(c,a.processes)
        launch(read(a.cluster),commands(read(a.cluster),a.config,a.resume,a.stop_after),a.logs)
        result=read(Path(c['output'])/'status.json')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
