"""Real native audio VAE + complete training read + repair checks on eight source clips, CPU only."""
import json,sys,time,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.config import load,activate
from p2_listen.cache import load_part
from p2_listen.training_data import load_training_sample
from listen_audio_postprocess.encoder import NativeAudioEncoder
from listen_audio_postprocess.overlay import sample_contract
from listen_audio_full.worker import cache_one,audit_task,task_rows
from listen_audio_full.preparation import prepare_archive
from listen_audio_full.indexing import connect
OUT=Path(__file__).resolve().parent;RUN=OUT/'validation100'
torch.set_num_threads(8);c=load(RUN/'view/config.json');activate(c);encoder=None
baseline=read(ROOT/'experiments/p2_train50h_audio_postprocess_20261026/gpu100_acceptance.json')
unchanged=[r['sample_id'] for r in baseline['records'] if not r['updated_audio']]
db=connect(RUN);longest=[r[0] for r in db.execute('SELECT sid FROM samples ORDER BY json_extract(obs,"$.requested_duration_s") DESC LIMIT 8')];db.close()
selected=set((unchanged+longest)[:8]);results=[];started=time.time()
def get_encoder():
 global encoder
 if encoder is None:encoder=NativeAudioEncoder(c,'cpu')
 return encoder
for path in sorted((RUN/'ready').glob('*.json')):
 task=read(path);jobs,_=task_rows(task)
 if not all(j['observation']['sample_id'] in selected for j in jobs):continue
 for job in jobs:
  o,t=job['observation'],job['target'];sid=o['sample_id'];result=cache_one(RUN,c,job,get_encoder)
  row=load_training_sample(c,o,t,'cpu');_,_,_,_,_,identity=sample_contract(c,o,t)
  patch=load_part(RUN/'view/cache/audio_postprocess'/sid,identity)
  assert row[2] is None
  if not patch['reuse_base_audio']:assert torch.equal(row[3]['av']['audio_latents']['latents'],patch['audio_latents']['latents'])
  result.update(training_read_passed=True);results.append(result)
 audit_task(RUN,c,task,final=True)
assert len(results)==8
# A real missing payload must be rebuilt rather than accepted through a finished marker.
sid=next(r['sample_id'] for r in results if r.get('audio_encoded'))
task=next(read(p) for p in (RUN/'ready').glob('*.json') if any(j['observation']['sample_id']==sid for j in task_rows(read(p))[0]))
job=next(j for j in task_rows(task)[0] if j['observation']['sample_id']==sid)
folder=RUN/'view/cache/audio_postprocess'/sid;before=torch.load(folder/'payload.pt',weights_only=True)['audio_latents']['latents'].clone()
(folder/'payload.pt').unlink();cache_one(RUN,c,job,get_encoder);after=torch.load(folder/'payload.pt',weights_only=True)['audio_latents']['latents']
assert torch.equal(before,after);audit_task(RUN,c,task,final=True)
# Rebuild a missing recipe + ready + open-integrated file from the actual original source.
db=connect(RUN);key=db.execute('SELECT key FROM archives WHERE key LIKE ?',(task['task_id'].split('-')[0]+'%',)).fetchone()[0];db.close()
(RUN/'view/audio_postprocess/recipes'/f'{sid}.json').unlink();(RUN/'ready'/f'{task["task_id"]}.json').unlink();Path(task['integrated']).unlink()
prepare_archive(RUN,key);assert read(RUN/'ready'/f'{task["task_id"]}.json')==task
audit_task(RUN,c,task,final=True)
report=dict(status='passed',samples=8,native_audio_vae_fp32=True,device='cpu',gpu_used=False,training_reads=8,
 missing_payload_reencoded_bit_exact=True,missing_recipe_ready_integrated_rebuilt=True,seconds=time.time()-started,records=results)
atomic_json(OUT/'native_cpu_validation.json',report);print(json.dumps(report,indent=2))
