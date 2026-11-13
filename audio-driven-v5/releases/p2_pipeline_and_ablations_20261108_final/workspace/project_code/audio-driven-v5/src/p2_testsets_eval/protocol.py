import hashlib,json,math,re,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
OMNI=ROOT.parent/'OmniResponse'
DEFAULT=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
METRIC_PYTHON=Path('/private/software/conda/envs/AnyTalker/bin/python')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def geometry(row,dataset):
 g=row['media']['listener']['video'] if dataset=='responsenet' else row
 n=g['frames'];fps=float(g['fps']);duration=float(row['duration_s'])
 assert n>0 and duration>0 and fps>0
 native=8*math.ceil((n-1)/8)+1
 return dict(fps=fps,width=512,height=512,source_width=g['width'],source_height=g['height'],frames=n,native_frames=native,
             duration_s=duration,native_duration_s=native/fps,gt_frame_grid_offset_s=n/fps-duration)
def partition(items,n):
 # Long clips go first; deterministic least-loaded assignment bounds stragglers.
 result=[[] for _ in range(n)];loads=[0.]*n
 for r in sorted(items,key=lambda x:(-x['geometry']['native_frames']*x['geometry']['width']*x['geometry']['height'],x['key'])):
  k=min(range(n),key=lambda i:loads[i]);result[k].append(r)
  loads[k]+=r['geometry']['native_frames']*r['geometry']['width']*r['geometry']['height']
 # Every GPU smoke covers both supported dataset adapters.
 for shard in result:
  first=[]
  for dataset in ('responsenet','seamless'):
   match=next((x for x in shard if x.get('dataset')==dataset),None)
   if match is not None:first.append(match)
  shard[:]=first+[x for x in shard if x not in first]
 return result

def prepare(work):
 work=Path(work);work.mkdir(exist_ok=False)
 datasets={};items=[]
 for name,folder,expected in [('responsenet','responsenet_pairs',586)]:
  path=OMNI/'evaluation'/folder/'prepared_20261102'
  refs=rows(path/'references.jsonl');assert len(refs)==expected and len({r['id'] for r in refs})==expected
  assert sha(path/'references.jsonl')==read(path/'seal.json')['references_sha256']
  datasets[name]=dict(prepared=str(path),references_sha256=sha(path/'references.jsonl'),count=len(refs),
                      face_padding=64 if name=='responsenet' else 0)
  for r in refs:
   assert re.fullmatch(r'[A-Za-z0-9_.-]+',r['id'])
   items.append(dict(key=name+'/'+r['id'],id=r['id'],dataset=name,reference=r,geometry=geometry(r,name)))
 arms=[]
 for kind,folder in [('av_event','p2_j_guided_av_event_s1_32gpu_alternating_20261101'),('av_only','p2_j_guided_av_only_s1_32gpu_20261101')]:
  for step in [6000,8927]:
   cp=ROOT/'experiments'/folder/f'step_{step:06d}.pt'
   arms.append(dict(name=f'{kind}_{step}',kind=kind,step=step,checkpoint=str(cp)))
 p=dict(schema='j-s1-testsets-autonomous-v1',created_at=time.time(),datasets=datasets,arms=arms,
        seed=137,planner='unmodified_Qwen2.5-Omni-3B',planner_adapter=None,
        planner_max_new_tokens=512,planner_retry_tokens=1024,planner_batch=16,text_batch=16,
        generation=dict(steps=30,video_cfg=3,audio_cfg=7),workers=32,total_samples=len(items),
        total_outputs=len(items)*len(arms),conditioning='speaker_audio_and_listener_first_frame_only',
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'),
        metrics_workers_per_gpu=4,metric_batch_size=64,execution_order='checkpoint_outer_then_responsenet',
        metrics=['psnr','ssim','fid','fvd','lpips','emo_acc','emo_div','delta_resp','utmos'],
        metric_assets=str(OMNI/'evaluation/video/paired_metrics/assets.local.json'))
 c=read(ROOT/'experiments/p2_j_av_event_then_av_20261101/cluster.json');c['nodes']=[n for n in c['nodes'] if n['name'] in ['node17','node12','node4','node8']]
 assert len(c['nodes'])==4
 save(work/'cluster.json',c)
 m=dict(c);m['known_hosts']=str(DEFAULT/'known_hosts.metrics');m['nodes']=[dict(name='evaluation',host='172.29.189.119',ssh_user='root',ssh_port=22,ssh_key='/private/ssh_keys/root_dedicated-developjob-zz-dev-machine-1-l1o08_ed25519',gpus=list(range(8)))]
 save(work/'metrics_cluster.json',m)
 save(work/'protocol.json',p)
 for i,shard in enumerate(partition(items,p['workers'])):save(work/'shards'/f'{i:02d}.json',shard)
 save(work/'status.json',dict(stage='prepared_waiting_training',total_outputs=p['total_outputs']))
 return p
