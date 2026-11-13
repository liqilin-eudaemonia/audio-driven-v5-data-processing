"""Prepare/verify current500h leave-one-out pipelines; preparation never launches GPUs."""
from pathlib import Path
import argparse,copy,fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,atomic_json
from p2_current_av_event_only import VERSION,ARMS
from p2_current_av_event_only.config import REFERENCE,FULL,load,validate
DEFAULT=ROOT/'experiments/p2_current_component_ablations31_20261108'

def layout(cluster):
 if [n['name']for n in cluster['nodes']]!=['node12','node8','node4','node11']:raise ValueError('Expected explicit four-node31 layout')
 result=[];offset=0
 for n in cluster['nodes']:
  g=list(range(7 if n['name']=='node11'else 8))
  if n['gpus']!=g:raise ValueError('Reserve node11 GPU7')
  result.append(dict(node=n['name'],gpus=g,rank_offset=offset,ranks=list(range(offset,offset+len(g)))));offset+=len(g)
 return result

def launcher():
 import p2_whisper_event_feature_train31 as native
 native.layout=layout;native.SELF=SELF
 return native

def commands(cluster,config,resume=None,stop_after=None):
 c=load(config);layout(cluster)
 if resume is not None and Path(resume).resolve().parent!=Path(c['output']).resolve():raise ValueError('Resume only this exact arm')
 if stop_after is not None and(type(stop_after)is not int or stop_after<1):raise ValueError('Invalid stop-after')
 result=launcher().commands(cluster,config,resume,Path(c['output'])/'rank_logs')
 for command in result:command.insert(1,'P2_COMPONENT_STOP_AFTER='+(''if stop_after is None else str(stop_after)))
 return result

def prepare(out,arm):
 out=Path(out).resolve()
 if(out/'config.json').exists():raise FileExistsError('Use immutable existing config or a new output')
 f=ARMS[arm];c=copy.deepcopy(read(REFERENCE/'config.json'));full=read(FULL);index=read(c['current_cache_index'])
 for k in ('current_av_whisper_experiment','whisper_reference','whisper_reference_sha256'):c.pop(k,None)
 loss='av_event_feature'if f['feature']and f['event']else('av_event'if f['event']else'av_feature')
 c.update(current_component_ablation=VERSION,component_arm=arm,component_reference=str(REFERENCE/'config.json'),component_reference_sha256=sha(REFERENCE/'config.json'),full_reference=str(FULL),full_reference_sha256=sha(FULL),output=str(out),id='guided_'+loss+'_s1',loss=loss,event_head_training=f['event'],teacher_cache=index['caches']['listener_teacher']['work']if f['feature']else None,loss_weights=dict(av=1.,event=.05 if f['event']else 0.,video_feature=.025 if f['feature']else 0.,audio_feature=.025 if f['feature']else 0.),hidden_event_supervision=dict(readout='last_block_before_output',clip_weight=1. if f['event']else 0.,temporal_weight=0.))
 if not f['whisper']:
  for k in ('whisper','whisper_cache'):c.pop(k)
 for k in ('clip_labels','clip_numeric_labels','current_source'):c[k]=full[k]if f['event']else None
 validate(c)
 from p2_current_av_event_only.schedule import Schedule
 schedule=Schedule(c);assert schedule.size==284218 and len(schedule.pairs)==9169*31
 cluster=copy.deepcopy(read(REFERENCE/'cluster.json'));cluster.update(training_script=str(SELF),training_config=str(out/'config.json'),master_port={'av_event':29864}[arm])
 out.mkdir(parents=True,exist_ok=True);atomic_json(out/'config.json',c,immutable=True);atomic_json(out/'cluster.json',cluster,immutable=True)
 atomic_json(out/'COMMANDS.json',commands(cluster,out/'config.json'),immutable=True)
 value=dict(prepared=True,training_started=False,at=time.time(),arm=arm,components=f,samples=index['samples'],hours=index['hours'],source_split_counts=index['source_split_counts'],world=31,steps=9169,save_every=500,initialization='pretrained_only',schedule_reused_exactly=True,cache_increment_required=False,reserved_gpu='node11:7')
 atomic_json(out/'PREPARED.json',value,immutable=True);return value

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','preflight','commands','train','launch','node'])
 p.add_argument('--arm',choices=ARMS,default='av_event');p.add_argument('--output',type=Path);p.add_argument('--config',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--logs',type=Path);p.add_argument('--node-index',type=int);p.add_argument('--attempt',type=Path)
 a=p.parse_args();out=(a.output or DEFAULT/a.arm).resolve()
 if a.command=='prepare':result=prepare(out,a.arm)
 else:
  config=(a.config or out/'config.json').resolve();c=load(config);out=Path(c['output'])
  if a.command=='node':return launcher().node(c,a.node_index,a.resume,a.attempt)
  if a.command=='commands':result=commands(read(out/'cluster.json'),config,a.resume,a.stop_after)
  elif a.command=='preflight':
   import torch
   torch.set_num_threads(2)
   from p2_current_av_event_only.training import preflight
   v=preflight(c);result=dict(success=True,arm=c['component_arm'],samples=v['samples'],canary_reads=len(v['canary_reads']),training_started=False)
  elif a.command=='train':
   from p2_current_av_event_only.training import train
   stop_after=a.stop_after
   if stop_after is None and os.environ.get('P2_COMPONENT_STOP_AFTER'):stop_after=int(os.environ['P2_COMPONENT_STOP_AFTER'])
   result=train(c,a.resume,stop_after)
  else:
   from listen_cluster import launch
   if not a.logs:p.error('Explicit launch requires --logs')
   if(out/'STOP').exists():raise ValueError('Preserve explicit STOP')
   if(out/'run.json').exists()and not a.resume:raise ValueError('Existing run requires strict resume')
   cluster=read(out/'cluster.json')
   with(DEFAULT/'launch.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources=launcher().selected_resources(cluster);atomic_json(out/'RESOURCE_PRECHECK.json',resources)
    if not resources['selected_all_idle']:raise RuntimeError('Selected31 GPUs are busy or observation failed; existing tasks are preserved')
    launch(cluster,commands(cluster,config,a.resume,a.stop_after),a.logs)
   result=read(out/'status.json')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
