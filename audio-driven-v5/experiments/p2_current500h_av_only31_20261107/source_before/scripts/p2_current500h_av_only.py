"""Prepare the original Qwen-only AV baseline on final500h; no implicit GPU launch."""
from pathlib import Path
import argparse,copy,csv,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,atomic_json
from p2_current_av_only import VERSION
from p2_current_av_only.config import BASE,REFERENCE,load,validate
from p2_current_av_only.schedule import Schedule
DEFAULT=ROOT/'experiments/p2_current500h_av_only_20261107'


def commands(cluster,config,resume=None,stop_after=None):
 c=load(config);nodes=cluster['nodes'];world=c['training']['global_batch']
 expected=['node12','node8','node4']+(['node11']if world==32 else[])
 if [n['name']for n in nodes]!=expected or any(n['gpus']!=list(range(8))for n in nodes):raise ValueError('Cluster differs from explicit AV-only24/32 profile')
 if resume is not None and Path(resume).resolve().parent!=Path(c['output']).resolve():raise ValueError('Resume only this AV-only experiment')
 result=[]
 for i,node in enumerate(nodes):
  command=['env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','PYTHONUNBUFFERED=1','TOKENIZERS_PARALLELISM=false',
   *[k+'='+str(v)for k,v in cluster['network'].items()],cluster['python'],'-m','torch.distributed.run',
   '--nnodes='+str(len(nodes)),'--nproc_per_node=8','--node_rank='+str(i),
   '--master_addr='+nodes[0]['host'],'--master_port='+str(cluster['master_port']),str(SELF),'train','--config',str(config)]
  if resume is not None:command+=['--resume',str(resume)]
  if stop_after is not None:command+=['--stop-after',str(stop_after)]
  result.append(command)
 return result


def prepare(out,world=32):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'config.json').exists():raise FileExistsError('Use the existing immutable config or a new output')
 c=copy.deepcopy(read(BASE));reference=read(REFERENCE/'config.json');index=read(reference['current_cache_index'])
 c.update(output=str(out),teacher_cache=None,clip_labels=None,current_av_only_experiment=VERSION,
  current_cache_index=reference['current_cache_index'],sample_list=str(Path(reference['current_source'])/'sample_list.csv'),
  j_cache=index['caches']['qwen_hidden']['work'],plan_cache=index['caches']['qwen_hidden']['work'],
  baseline_config=str(BASE),baseline_config_sha256=sha(BASE),qwen_hidden_source='base_pre_sft',initialization='pretrained_only',
  loss_weights=dict(av=1.,event=0.,video_feature=0.,audio_feature=0.),
  execution=dict(prefetch_batches=2,gradient_reduction='streaming_fp64',gradient_bucket_mb=512,
                 optimizer_state_cpu_offload=False,longest_sample_check=True))
 c['training'].update(global_batch=world,steps=math.ceil(index['samples']/world),save_every=500,gradient_checkpointing='all')
 if world==32:c['duration_sampling']=copy.deepcopy(reference['duration_sampling'])
 else:
  from p2_j_duration_execution import make_pairs
  with Path(c['sample_list']).open()as f:rows=list(csv.DictReader(f))
  plan=out/'duration.schedule.json';list_sha=sha(c['sample_list'])
  pairs=make_pairs([dict(requested_duration_s=float(r['duration_s']))for r in rows],c['training']['steps'],world,c['training']['seed'],0,'duration',world*16)
  atomic_json(plan,dict(pairs=pairs,sample_list_sha256=list_sha),immutable=True)
  c['duration_sampling']=dict(policy='duration',window=world*16,start_step=0,manifest=str(plan),sha256=sha(plan),sample_list_sha256=list_sha)
 validate(c);schedule=Schedule(c)
 real=[(i,o)for i,o in schedule.pairs if o<index['samples']]
 if len(real)!=index['samples']or len({i for i,o in real})!=index['samples']or {o for i,o in real}!=set(range(index['samples'])):
  raise ValueError('Lost/duplicated real sample or noise ordinal')
 cluster=copy.deepcopy(read(REFERENCE/'cluster.json'));cluster['nodes']=cluster['nodes'][:world//8];cluster['master_port']=29848 if world==32 else 29849
 for n in cluster['nodes']:n['gpus']=list(range(8))
 cluster.update(training_script=str(SELF),training_config=str(out/'config.json'))
 atomic_json(out/'config.json',c,immutable=True);atomic_json(out/'cluster.json',cluster,immutable=True)
 cmds=commands(cluster,out/'config.json');atomic_json(out/'COMMANDS.json',cmds,immutable=True)
 value=dict(prepared=True,training_started=False,at=time.time(),samples=index['samples'],hours=index['hours'],
  source_split_counts=index['source_split_counts'],world=world,global_batch=world,steps=c['training']['steps'],
  padding=len(schedule.pairs)-index['samples'],real_samples_and_noise_ordinals_exact=True,
  exact_initial_full32_schedule=world==32,whisper=False,event_loss=False,feature_loss=False,
  initialization='original pretrained weights and baseline seed; no trained checkpoint inherited')
 atomic_json(out/'PREPARED.json',value,immutable=True);return value


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','preflight','commands','train','launch'])
 p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--world',type=int,choices=[24,32],default=32)
 p.add_argument('--config',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--logs',type=Path)
 a=p.parse_args()
 if a.command=='prepare':result=prepare(a.output,a.world)
 else:
  config=(a.config or a.output/'config.json').resolve();c=load(config);out=Path(c['output'])
  if a.command=='commands':result=commands(read(out/'cluster.json'),config,a.resume,a.stop_after)
  elif a.command=='preflight':
   import torch
   torch.set_num_threads(2)
   from p2_current_av_only.training import preflight
   v=preflight(c);result=dict(success=True,samples=v['samples'],canary_reads=len(v['canary_reads']),training_started=False)
  elif a.command=='train':
   from p2_current_av_only.training import train
   result=train(c,a.resume,a.stop_after)
  else:
   from listen_cluster import launch
   if (out/'STOP').exists():raise ValueError('Preserve explicit STOP')
   if not a.logs:p.error('Explicit launch requires --logs')
   if (out/'run.json').exists()and not a.resume:raise ValueError('Existing run requires strict resume')
   # Preparation does not authorize auto-launch: this branch is only reached by
   # a separate explicit CLI launch invocation after machine allocation.
   launch(read(out/'cluster.json'),commands(read(out/'cluster.json'),config,a.resume,a.stop_after),a.logs)
   result=read(out/'status.json')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
