"""Re-cut ONLY the two reviewed conversations, then apply unchanged QC rules.

Initial segmentation deliberately precedes automatic audio exclusions so each
withheld full pair is inspectable. The unchanged localized recovery then runs
on fresh failures; parents and children are kept separate in the result.
"""
import collections
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=EXP/'fixed_pairs'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import listen_local_recovery_v2 as repair
import run_listen_v8_audio_full as screening
from p3_listen.common import sha

PAIRS=['naturalistic_V02_S3579_I00000175','naturalistic_V03_S1909_I00000205']
PROD=ROOT/'data_corpora/seamless_listen_local_full'
PREV=ROOT/'data_corpora/seamless_listen_v8_audio_final_full'
EXTRA_CACHE_DIRS=[]
def read(p):return json.loads(p.read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n');tmp.replace(p)

def context(pair):
    d=read(EXP/(pair+'.json'))
    ctx=repair.SourceContext.__new__(repair.SourceContext)
    ctx.item=d['item'];ctx.people=repair.prepare_people(d['people']) if hasattr(repair,'prepare_people') else d['people'];ctx.rms=[np.array(x) for x in d['rms']]
    ctx.limit=d['limit'];ctx.sources=d['sources'];ctx.records=[dict(id=s['id']) for s in d['sources']]
    # Geometry does not participate in this audio/segmentation audit. Full
    # uncropped participant pictures are used in the HTML.
    ctx.geometry=[dict(s,rotation_degrees=0,arrays={}) for s in d['sources']]
    ctx.shots=[];ctx.shots_by_track=[[],[]]
    ctx.regions,ctx.source_gaps=repair.role_regions(ctx.people,ctx.limit,[])
    ctx.reviews=read(ROOT/'data_reviews/listener_noise_rejections.json')['rows']
    return ctx

def main():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    screening.FEEDBACK=read(PROD/'feedback.json')['rows']
    screening.OLD_FEEDBACK=read(PROD/'qc4_review_feedback.json')['rows']
    feedback=dict(rejects=read(PROD/'human_rejections.json')['rows'],positives=read(PROD/'human_retained.json')['rows'])
    frozen=read(PROD/'contract.json')['code_sha256']
    assert all(sha(ROOT/p)==digest for p,digest in frozen.items())
    write(OUT/'contract.json',dict(pairs=PAIRS,revision=repair.REVISION,
        repair_module=str(Path(repair.__file__).relative_to(ROOT)),repair_sha256=sha(Path(repair.__file__)),
        pause_threshold_seconds=getattr(repair,'MAX_WORD_GAP',3.),
        screening_revision=screening.policy.REVISION,screening_rules_changed=False,
        initial_segmentation='Before automatic quality blocks; human holds are enforced by unchanged routing',
        frozen_code_sha256=frozen,production_results={p:sha(PROD/'results'/(p+'.json')) for p in PAIRS},
        device='cpu',production_modified=False))
    models=[]
    for pair in PAIRS:
        began=time.monotonic();ctx=context(pair)
        initial,diagnostic=ctx.propose([])
        write(OUT/pair/'initial.json',dict(pair=pair,clips=initial,diagnostic=diagnostic,regions=ctx.regions,
            local_feedback_audit=[p.get('local_feedback_audit',[]) for p in ctx.people]))
        print(pair,'INITIAL',len(initial),flush=True)
        norm=read(PROD/'normalized_pairs'/(pair+'.json'))
        assert all(sha(t['path'])==t['sha256'] for t in norm['tracks'])
        cache={}
        for folder in [PREV,PROD]:
            path=folder/'results'/(pair+'.json');d=read(path)
            for row in d['clips']+d.get('superseded_attempts',[]):
                cache[screening.interval_key(row)]=(row,str(path),sha(path))
        for directory in [*EXTRA_CACHE_DIRS,OUT]:
            for path in (directory/pair/'qc').glob('*.json'):
                row=read(path);cache[screening.interval_key(row)]=(row,str(path),sha(path))
        pending=[(r,0,None) for r in initial];seen=set();leaves=[];parents=[];counts=collections.Counter()
        while pending:
            plan,depth,parent=pending.pop(0);key=screening.interval_key(plan)
            if key in seen:continue
            seen.add(key);plan.update(recovery_depth=depth,recovery_parent_id=parent)
            old=cache.get(key)
            if old and screening.cache_matches(plan,old[0],norm):
                row=copy.deepcopy(old[0]);provenance=dict(reused=True,file=old[1],sha256=old[2],exact_interval=True)
                counts['reused']+=1
            else:
                if not models:
                    print('LOADING CPU AST / PANNs',flush=True)
                    models=[screening.full.legacy.EventModel(screening.full.legacy.MODEL,device='cpu',batch_size=4),
                            screening.full.legacy.PannsModel(device='cpu',batch_size=4)]
                print('FRESH',pair,plan['start'],plan['end'],flush=True)
                row=screening.RAW_SCREEN(plan,norm,*models,feedback['rejects'],feedback['positives'])
                row=screening.original.assess(row,screening.OLD_FEEDBACK)
                provenance=dict(reused=False,fresh_qc=True,device='cpu',algorithm='unchanged QC3 + QC4')
                counts['fresh']+=1
            row=screening.finish_row(plan,row,feedback);row['quality_provenance']=provenance
            write(OUT/pair/'qc'/(row['sample_id']+'.json'),row)
            children=[]
            fallback=repair.quality_fallback(plan,row) if hasattr(repair,'quality_fallback') else None
            if fallback:
                children=fallback
                for child in children:
                    assert plan['start']<=child['start'] and child['end']<=plan['end']
                    assert child['sample_id'] not in seen
                    pending.append((child,depth+1,row['sample_id']))
                diagnostic.setdefault('head_quality_fallback',[]).append(dict(
                    attempted_id=plan['sample_id'],fallback_ids=[c['sample_id'] for c in children],
                    reason='optional_head_extension_failed_quality'))
            elif row['audio_final']['pool']!='keep_candidate':
                _,kind=repair.base.quality_blocks(row)
                if kind=='localized_quality' and depth<6:
                    children,detail=ctx.propose([row],domain=row)
                    diagnostic['blocked'].extend(detail['blocked']);diagnostic['unresolved'].extend(detail['unresolved'])
                    children=[c for c in children if screening.interval_key(c) not in seen]
                    for child in children:
                        assert child['duration_s']<row['duration_s']-1e-5
                        pending.append((child,depth+1,row['sample_id']))
                elif kind=='localized_quality':
                    diagnostic['unresolved'].append(dict(start=row['start'],end=row['end'],reason='recovery_depth_limit'))
            (parents if children else leaves).append(row)
            print('DONE',pair,row['start'],row['end'],row['audio_final']['pool'],'remaining',len(pending),flush=True)
        leaves.sort(key=lambda r:(r['start'],r['end']))
        assert all(a['end']<=b['start']+1e-6 for a,b in zip(leaves,leaves[1:]))
        kept=[(r['start'],r['end']) for r in leaves if r['audio_final']['pool']=='keep_candidate']
        result=dict(pair=pair,revision=repair.REVISION,source_seconds=ctx.limit,clips=leaves,
            superseded_attempts=parents,initial_ids=[r['sample_id'] for r in initial],
            diagnostic=diagnostic,regions=ctx.regions,source_gaps=ctx.source_gaps,
            local_feedback_audit=[p.get('local_feedback_audit',[]) for p in ctx.people],
            counts=dict(counts),pool_counts=dict(collections.Counter(r['audio_final']['pool'] for r in leaves)),
            kept_seconds=repair.base.seconds(kept),elapsed=time.monotonic()-began,complete=True,training_admitted=False)
        write(OUT/pair/'result.json',result)
        print('COMPLETE',pair,result['pool_counts'],result['counts'],flush=True)
    assert all(sha(ROOT/p)==digest for p,digest in frozen.items())

if __name__=='__main__':main()
