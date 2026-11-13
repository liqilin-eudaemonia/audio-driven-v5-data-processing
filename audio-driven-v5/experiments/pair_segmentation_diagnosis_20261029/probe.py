"""Read-only segmentation ablations on the seven already cached sources; no QC rerun."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1].parent;EXP=Path(__file__).resolve().parent;SEVEN=ROOT/'experiments/pair_head_seven_20261029'
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import listen_local_recovery as v1
import listen_local_recovery_v2 as f
import listen_local_recovery_gap5 as g
import listen_local_recovery_localbc as l
import listen_local_recovery_head_attach as h
read=lambda p:json.loads(p.read_text())
runner.EXP=SEVEN
simple=lambda r:{k:r.get(k) for k in ['start','end','speech_start','speech_end','text','semantic_start','semantic_end','recovery_complete_sentence','boundary_repair','speaker_index']}
report=dict(matrix=[],recovery_traces=[])
PAIRS=read(SEVEN/'results/contract.json')['pairs']
for pair in PAIRS:
 entry=dict(pair=pair,versions={})
 for name,module in [('V1',v1),('F',f),('G',g),('L',l),('H',h)]:
  runner.repair=module;ctx=runner.context(pair);rows,diag=ctx.propose([])
  entry['versions'][name]=dict(rows=[simple(r) for r in rows],count=len(rows))
  if name=='H':
   actual=read(SEVEN/'results'/pair/'initial.json')['clips']
   sig=lambda rs:[(r['speaker_index'],r['start'],r['end'],r['text']) for r in rs]
   assert sig(rows)==sig(actual)
 report['matrix'].append(entry);print('MATRIX',pair,[(k,v['count']) for k,v in entry['versions'].items()],flush=True)
# Instrument in this diagnostic process only; production policy files untouched.
original=l.recover_region
current={}
def trace(people,rms,si,a,b,limit):
 greedy,greedy_fail=l._greedy_recover_region(people,rms,si,a,b,limit)
 final,final_fail=original(people,rms,si,a,b,limit)
 if [simple(r) for r in greedy]!=[simple(r) for r in final]:
  report['recovery_traces'].append(dict(**current,speaker_index=si,safe_start=a,safe_end=b,greedy=[simple(r) for r in greedy],after_tail_repair=[simple(r) for r in final],greedy_failures=greedy_fail,final_failures=final_fail))
 return final,final_fail
l.recover_region=trace
for pair in PAIRS:
 runner.repair=h;ctx=runner.context(pair);current=dict(pair=pair,stage='initial',parent=None);ctx.propose([])
 result=read(SEVEN/'results'/pair/'result.json')
 for row in result['superseded_attempts']:
  if row['audio_final']['pool']=='keep_candidate':continue
  if row.get('head_attachment_original'):continue
  current=dict(pair=pair,stage='quality_recovery',parent=simple(row),pool=row['audio_final']['pool'],reasons=row['audio_final']['reasons'])
  ctx.propose([row],domain=row)
l.recover_region=original
(EXP/'probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for e in report['matrix']:
 if e['pair']==PAIRS[0]:
  for name,v in e['versions'].items():print(name,[(r['start'],r['end'],r['text']) for r in v['rows'] if 20<r['start']<44])
for t in report['recovery_traces']:
 if t['pair']==PAIRS[0] and t['stage']=='quality_recovery' and 140<t['safe_start']<155:print('TRACE',json.dumps(t,ensure_ascii=False))
print('DONE',len(report['recovery_traces']),'changed tail-repair calls',flush=True)
