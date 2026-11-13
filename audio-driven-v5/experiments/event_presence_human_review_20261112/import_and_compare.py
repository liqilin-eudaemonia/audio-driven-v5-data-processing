"""Transcribe the user's two submitted tables, preserving labels and rater01."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'experiments/p2_event_presence_pilot_20261108/run'
NEW=ROOT/'experiments/seamless_upright_eval_fix_20261112/event_presence'
IDS=['0846ca313d1440c0a70d388c84bf36b0b244ae119bf4021ab4318e650abfa849','0a3ac306ee9f8b9848183465174c6fa906cefa1f63e06de45a16a12ee287b24a','39cd92f6befae26619cc3f3fc6508ac902f57b439eeee20364adab3faf763329','54215a56ea359066eb69972a5bf582e196e53991fe89ca3be21d3c976a9814eb','6cb32307bb678b8429a4cef602b59580e66d4e7fee50c2cfe20810314059f379','a1fcfc0a634481a8382609dfa854d469c00435c1fa51eb42a44af1d04c72394c','a6ef434d9bd1d204e5ac24110b4d2e6acf004d792f0d086b51127435131bfcdb','c805d700499778275dcb225aebace7e18e95397d8379e946450f69198aa329fa']
CLASSES={'audio':['audible_laughter','brief_verbal_response','nonlexical_response'],'visual':['nodding','head_shaking','smiling','frowning']}
NAMES=dict(zip(CLASSES['audio']+CLASSES['visual'],['可听笑声','简短词语回应','非词汇回应','点头','摇头','微笑／笑态','皱眉']))
# Exact positive entries from the user; all other submitted entries are absent.
POS={'audio':{0:{'brief_verbal_response'}},'visual':{0:{'nodding','smiling','frowning'},2:{'nodding','smiling'},3:{'nodding'},5:{'nodding','smiling'},6:{'nodding','smiling'},7:{'frowning'}}}
FIELDS=['media_id','modality','class','state','evidence','rater_id']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,values,fields=FIELDS):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
def media(work):return {m['media_id']:m for r in rows(work/'prepared_pairs.jsonl') for m in r['prepared_media'].values()}
def compare(annotations,work,name):
 diffs=[];totals={};perclass={}
 for row in annotations:
  if not row['state']:continue
  mod,c=row['modality'],row['class'];path=work/'media'/row['media_id']/'detection.json'
  det=read(path)['labels'][mod][c] if path.exists() else {'state':'pending','evidence':'detection not finished'}
  h,a=row['state'],det['state'];t=totals.setdefault(mod,dict(human_units=0,determinate_units=0,agree=0,disagree=0,unknown=0,pending=0,error=0));t['human_units']+=1
  k=perclass.setdefault(mod+'/'+c,dict(tp=0,tn=0,fp=0,fn=0,human_positives=0,unknown=0,pending=0,error=0));k['human_positives']+=h=='present'
  if a in ('present','absent'):
   t['determinate_units']+=1;t['agree' if h==a else 'disagree']+=1;k['tp' if a==h=='present' else 'tn' if a==h=='absent' else 'fp' if a=='present' else 'fn']+=1
  else:t[a]+=1;k[a]+=1
  if h!=a:diffs.append(dict(**row,automatic_state=a,automatic_evidence=det['evidence']))
 for t in totals.values():t['determinate_agreement']=t['agree']/t['determinate_units'] if t['determinate_units'] else None
 for k in perclass.values():
  k['precision']=k['tp']/(k['tp']+k['fp']) if k['tp']+k['fp'] else None
  k['recall']=k['tp']/(k['tp']+k['fn']) if k['tp']+k['fn'] else None
  k['f1']=2*k['tp']/(2*k['tp']+k['fp']+k['fn']) if k['human_positives'] and k['tp']+k['fp']+k['fn'] else None
  k['scope']='determinate detection subset; unknown/pending/error separate'
 result=dict(modalities=totals,classes=perclass,differences=diffs,rater='01',one_rater_small_pilot=True,thresholds_adjusted=False)
 save(OUT/(name+'.json'),result);write_csv(OUT/(name+'_differences.csv'),diffs,FIELDS+['automatic_state','automatic_evidence']);return result

if __name__=='__main__':
 original=[]
 for mod,cs in CLASSES.items():
  part=[]
  for i,mid in enumerate(IDS):
   for c in cs:
    state='present' if c in POS[mod].get(i,set()) else 'absent';name=NAMES[c]
    ev=f'人工核验：本段观察到{name}。' if state=='present' else f'人工核验：完整观看／收听后，未观察到{name}。'
    part.append(dict(media_id=mid,modality=mod,**{'class':c},state=state,evidence=ev,rater_id='01'))
  write_csv(OUT/f'submitted_{mod}_rater01.csv',part);original+=part
 write_csv(OUT/'submitted_combined_rater01.csv',original);assert len(original)==56
 oldmedia,newmedia=media(OLD),media(NEW);assert set(IDS)==set(oldmedia)
 # Match sides of the same source sample, not unrelated identical silent waves.
 newpairs={r['id']:r for r in rows(NEW/'prepared_pairs.jsonl')};mapping={}
 for r in rows(OLD/'prepared_pairs.jsonl'):
  if r['id'] not in newpairs or r.get('self_check_only'):continue
  for side,m in r['prepared_media'].items():mapping[m['media_id']]=newpairs[r['id']]['prepared_media'][side]['media_id']
 carried=[];audit=[]
 for row in original:
  mid=row['media_id'];newmid=mapping[mid];mod=row['modality'];field='rgb_sha256' if mod=='visual' else 'audio_sha256';a,b=oldmedia[mid]['identity'],newmedia[newmid]['identity']
  same=a['duration_s']==b['duration_s'] and a[field] is not None and a[field]==b[field] and not a['errors'].get(mod) and not b['errors'].get(mod)
  if same:carried.append(dict(row,media_id=newmid))
  audit.append(dict(old_media_id=mid,new_media_id=newmid,modality=mod,**{'class':row['class']},carried=same,proof_field=field,old_hash=a[field],new_hash=b[field]))
 write_csv(OUT/'upright_carried_rater01.csv',carried)
 complete={(r['media_id'],r['modality'],r['class']):r for r in carried};pending=[];full=[]
 for mid in sorted(newmedia):
  for mod,cs in CLASSES.items():
   for c in cs:
    key=(mid,mod,c);r=complete.get(key,dict(media_id=mid,modality=mod,**{'class':c},state='',evidence='',rater_id='01'));full.append(r)
    if not r['state']:pending.append(r)
 write_csv(OUT/'upright_review_rater01.csv',full);write_csv(OUT/'upright_pending_rater01.csv',pending)
 save(OUT/'TRANSFER_AUDIT.json',dict(submitted=56,carried=len(carried),pending=len(pending),matches=audit))
 print('carried',len(carried),'pending',len(pending))
 for name,work,data in [('old_detector_vs_human',OLD,original),('upright_detector_vs_human',NEW,carried)]:
  r=compare(data,work,name);print(name,r['modalities'])
