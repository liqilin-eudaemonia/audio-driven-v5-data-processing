import inspect,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/pair_boundary_real_audit_20261029'
sys.path[:0]=[str(ROOT/'scripts'),str(EXP)]
import rerun_fixed_pairs as run
import listen_local_recovery as base
import listen_local_recovery_v2 as fixed
import listen_policy_v8 as policy

def clone(fn,namespace,old,new):
    source=inspect.getsource(fn)
    assert source.count(old)==1,(fn.__name__,old)
    exec(compile(source.replace(old,new),'<gap5-trial>','exec'),namespace)
    return namespace[fn.__name__]

unit_ns=dict(vars(policy));units=clone(policy.sentence_units,unit_ns,'elif gap>3:', 'elif gap>5:')
base_ns=dict(vars(base));base_ns['sentence_units']=units
role=clone(base.role_regions,base_ns,"atom['end']-atom['start'] <= 3:","atom['end']-atom['start'] <= 5:")
recover=clone(base.recover_region,base_ns,"words[j]['start']-words[j-1]['end'] > 3)","words[j]['start']-words[j-1]['end'] > 5)")
adapter=types.SimpleNamespace(**vars(base));adapter.role_regions=role;adapter.recover_region=recover;adapter.sentence_units=units
ns=dict(vars(fixed));ns['base']=adapter;ns['REVISION']='gap5-trial-only'
exec(inspect.getsource(fixed.role_regions),ns)
clone(fixed.recover_region,ns,"v['start']-u['end'] > 3 for", "v['start']-u['end'] > 5 for")
exec(inspect.getsource(fixed.SourceContext),ns)
results=[]
for pair in run.PAIRS:
    ctx=run.context(pair);before,bad_before=ctx.propose([])
    ctx.__class__=ns['SourceContext'];ctx.regions,ctx.source_gaps=ns['role_regions'](ctx.people,ctx.limit,[])
    after,bad=ctx.propose([])
    def covers(w,rs,si):return any(r['speaker_index']==si and r['start']<=w['start']+1e-6 and r['end']>=w['end']-1e-6 for r in rs)
    lost=[w for si,p in enumerate(ctx.people) for w in p['words'] if covers(w,before,si) and not covers(w,after,si)]
    added=[w for si,p in enumerate(ctx.people) for w in p['words'] if not covers(w,before,si) and covers(w,after,si)]
    previous={(r['speaker_index'],r['start'],r['end']) for r in before}
    changed=[r for r in after if (r['speaker_index'],r['start'],r['end']) not in previous]
    d=dict(pair=pair,before_count=len(before),after_count=len(after),lost_words=lost,added_words=added,changed_clips=changed,
           clips=after,diagnostic=bad,regions=ctx.regions)
    results.append(d)
    print(json.dumps(dict(pair=pair,before=len(before),after=len(after),lost=[w['text'] for w in lost],added=[w['text'] for w in added],
        changed=[dict(start=r['start'],end=r['end'],text=r['text'],flags=r['boundary_review_flags']) for r in changed]),ensure_ascii=False),flush=True)
run.write(EXP/'gap5_trial/segmentation.json',dict(pause_threshold_seconds=5,scope=run.PAIRS,quality_recomputed=False,production_modified=False,pairs=results))
