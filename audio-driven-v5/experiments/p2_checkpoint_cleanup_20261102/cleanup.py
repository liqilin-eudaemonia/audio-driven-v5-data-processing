"""Delete only explicitly inventoried checkpoints from the two authorized S1 runs."""
import argparse,hashlib,json,re,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_checkpoint_cleanup_20261102'
NAMES=[
 'p2_j_guided_av_event_s1_32gpu_20261101',
 'p2_j_guided_av_event_s1_32gpu_duration512_20261101',
 'p2_j_guided_av_event_s1_32gpu_resume200_20261101',
 'p2_j_guided_av_event_s1_32gpu_alternating_20261101',
 'p2_j_guided_av_only_s1_32gpu_20261101']
DIRS=[ROOT/'experiments'/n for n in NAMES]

def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def inventory():
 result=[]
 for d in DIRS:
  for p in sorted(d.glob('step_*.pt')):
   assert re.fullmatch(r'step_\d{6}\.pt',p.name)
   marker=p.with_suffix('.complete.json')
   if not marker.exists():continue
   assert not p.is_symlink() and not marker.is_symlink()
   m=read(marker);step=int(p.stem.split('_')[1]);assert m['step']==step
   st=p.stat()
   result.append(dict(path=str(p),marker=str(marker),step=step,bytes=st.st_size,mtime_ns=st.st_mtime_ns,
                      inode=st.st_ino,sha256=m['sha256']))
 return result

def plan():
 assert not (WORK/'plan.json').exists()
 records=inventory();latest=sorted(r['step'] for r in records if Path(r['path']).parent==DIRS[-1])[-2:]
 keep=[];delete=[]
 for r in records:
  reasons=[]
  if r['step'] in {0,2000,4000,6000,8927}:reasons.append('representative_or_final_checkpoint')
  if r['step']==4201 and Path(r['path']).parent in DIRS[2:4]:reasons.append('queue_migration_anchor_or_source')
  if Path(r['path']).parent==DIRS[-1] and r['step'] in latest:reasons.append('active_training_latest_two')
  r['reasons']=reasons
  (keep if reasons else delete).append(r)
 job=read(ROOT/'experiments/p2_j_av_event_then_av_20261101/job.json')
 assert job['av_event_checkpoint'] in {r['path'] for r in keep}
 data=dict(created_at=time.time(),scope=NAMES,active_latest_two=latest,keep=keep,delete=delete,
           delete_checkpoint_count=len(delete),delete_checkpoint_bytes=sum(r['bytes'] for r in delete))
 save(WORK/'plan.json',data)
 print(json.dumps(dict(delete_count=len(delete),delete_gib=data['delete_checkpoint_bytes']/2**30,
                      keep=[(Path(r['path']).parent.name,r['step']) for r in keep])),flush=True)

def apply():
 p=read(WORK/'plan.json');assert not (WORK/'complete.json').exists()
 # Verify retained recovery anchors before permanently removing old alternatives.
 for r in p['keep']:
  h=hashlib.sha256()
  with open(r['path'],'rb') as f:
   for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
  assert h.hexdigest()==r['sha256'],r['path']
 print('All retained checkpoint hashes verified',flush=True)
 current=inventory();active=sorted((r for r in current if Path(r['path']).parent==DIRS[-1]),key=lambda r:r['step'])[-2:]
 protected={r['path'] for r in p['keep']}|{r['path'] for r in active}
 deleted=[];skipped=[]
 with (WORK/'deletions.jsonl').open('a',buffering=1) as log:
  for r in p['delete']:
   path=Path(r['path']);marker=Path(r['marker'])
   assert path.parent in DIRS and path.name==f"step_{r['step']:06d}.pt"
   if str(path) in protected:skipped.append(str(path));continue
   st=path.stat()
   assert (st.st_size,st.st_mtime_ns,st.st_ino)==(r['bytes'],r['mtime_ns'],r['inode'])
   assert read(marker)==dict(step=r['step'],sha256=r['sha256'])
   # Remove completion marker first so readers never treat a missing tensor as complete.
   marker.unlink();path.unlink()
   log.write(json.dumps(dict(**r,deleted_at=time.time()))+'\n');deleted.append(r)
 for path in protected:assert Path(path).is_file() and Path(path).with_suffix('.complete.json').is_file()
 assert all(not Path(r['path']).exists() and not Path(r['marker']).exists() for r in deleted)
 result=dict(status='complete',deleted_checkpoints=len(deleted),deleted_markers=len(deleted),
             reclaimed_checkpoint_bytes=sum(r['bytes'] for r in deleted),skipped=skipped,
             retained=[dict(path=r['path'],step=r['step']) for r in inventory()],
             retained_hashes_verified=True,training_status=read(DIRS[-1]/'status.json'),completed_at=time.time())
 save(WORK/'complete.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','apply']);a=p.parse_args()
 {'plan':plan,'apply':apply}[a.action]()
