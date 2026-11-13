"""Compare every changed initial interval and source-aligned word coverage."""
import json,sys,math
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import listen_local_recovery as base

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sig(r):return (r['speaker_index'],r['start'],r['end'])
def roots(result):
 ids=set(result['initial_ids'])
 return sorted([r for r in result['clips']+result['superseded_attempts'] if r['sample_id'] in ids],key=lambda r:r['start'])
def covered(w,rows,role=False):
 return any(r['start']<=w['start']+1e-6 and r['end']>=w['end']-1e-6 and (not role or r['speaker_index']==w['person']) for r in rows)
def words(d):
 return [dict(start=w['start'],end=w['end'],text=w['text'],person=i) for i,p in enumerate(d['people']) for w in p['words'] if all(math.isfinite(w[k]) for k in ('start','end')) and 0<w['end']-w['start']<=1.5]
def changes(old,new):
 a={sig(r):r for r in old};b={sig(r):r for r in new}
 rows=sorted([(r['start'],r['end'],'before',r) for k,r in a.items() if k not in b]+[(r['start'],r['end'],'after',r) for k,r in b.items() if k not in a],key=lambda x:x[:2])
 groups=[]
 for start,end,side,row in rows:
  if not groups or start>groups[-1]['end']+.001:groups.append(dict(start=start,end=end,before=[],after=[]))
  g=groups[-1];g['end']=max(end,g['end']);g[side].append(row)
 return groups

def evaluate(d,old,new):
 ws=words(d);a={sig(r) for r in old};b={sig(r) for r in new}
 added=[w for w in ws if covered(w,new,True) and not covered(w,old,True)]
 lost=[w for w in ws if covered(w,old,True) and not covered(w,new)]
 changed_role=[w for w in ws if covered(w,old,True) and covered(w,new) and not covered(w,new,True)]
 cuts=[dict(time=edge,word=w,sample_id=r['sample_id']) for r in new for edge in (r['start'],r['end']) for w in ws if w['start']+1e-6<edge<w['end']-1e-6]
 return dict(old_count=len(a),new_count=len(b),unchanged=len(a&b),changed_groups=len(changes(old,new)),added_speaker_words=added,previous_speaker_words_lost=lost,previous_speaker_words_now_on_other_role=changed_role,new_word_crossing_edges=cuts,duration_violations=[r['sample_id'] for r in new if not 2-1e-6<=r.get('duration_s',r['end']-r['start'])<=11.95+1e-6],overlaps=[(u['sample_id'],v['sample_id']) for u,v in zip(new,new[1:]) if u['end']>v['start']+1e-6])

if __name__=='__main__':
 results=[]
 for item in read(EXP/'selection.json')['items']:
  pair=item['pair'];d=read(EXP/(pair+'.json'))
  old=read(EXP/'gap5'/pair/'result.json');new=read(EXP/'localbc'/pair/'result.json')
  result=dict(pair=pair,seconds=d['limit'],initial=evaluate(d,roots(old),roots(new)),terminal=evaluate(d,old['clips'],new['clips']),kept=evaluate(d,[r for r in old['clips'] if r['audio_final']['pool']=='keep_candidate'],[r for r in new['clips'] if r['audio_final']['pool']=='keep_candidate']),old_kept_seconds=old['kept_seconds'],new_kept_seconds=new['kept_seconds'],local_feedback_audit=new['local_feedback_audit'])
  results.append(result)
 write(EXP/'comparison.json',dict(pairs=results,scope='three preselected independent source conversations',production_modified=False))
 print(json.dumps([dict(pair=r['pair'],**{k:v for k,v in r['initial'].items() if not isinstance(v,list)}) for r in results],indent=2))
