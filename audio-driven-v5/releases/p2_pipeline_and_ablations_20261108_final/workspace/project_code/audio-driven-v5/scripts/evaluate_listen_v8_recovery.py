"""Bounded recovery experiment alongside immutable QC4 production artifacts."""
import argparse,collections,concurrent.futures,json,os,random,shutil,sys,time,traceback
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_qc4_full as baseline
import listen_v8_recovery as recovery
from p3_listen.common import read,write,sha
ROOT=baseline.ROOT
BASE=baseline.OUT
OUT=ROOT/'experiments/p3_listen_v8_recovery_20261023'
SCRIPT=Path(__file__).resolve()
CATALOG=None

def setup():
    baseline.install();baseline.full.verify()
    for d in ('audit','normalized_pairs','results','errors','logs','progress'):(OUT/d).mkdir(parents=True,exist_ok=True)
    baseline.full.OUT=OUT;baseline.full.configure_normalization()
    recovery.install_planner_patch()

def key(row):return (row['speaker_id'],row['listener_id'],round(row['start'],6),round(row['end'],6))

def prepare_one(item):
    marker=OUT/'normalized_pairs'/(item['pair']+'.json');parent=BASE/'normalized_pairs'/marker.name
    if parent.exists() and not marker.exists():shutil.copyfile(parent,marker)
    audit,norm,seconds=baseline.full.prepare(item)
    old=read(BASE/'results'/marker.name)
    oldkeys={key(r) for r in old['clips']}
    novel=[r for r in audit['clips'] if key(r) not in oldkeys]
    write(OUT/'prepared' / marker.name,dict(pair=item['pair'],split=item['split'],group_id=item['group_id'],
        novel_clips=novel,normalized=norm,old_clips=len(old['clips']),new_clips=len(audit['clips']),
        old_result_sha256=sha(BASE/'results'/marker.name),preparation_seconds=seconds))
    return dict(pair=item['pair'],old=len(old['clips']),new=len(audit['clips']),novel=len(novel))

def prepare_init():
    global CATALOG
    setup();CATALOG={r['pair']:r for r in read(BASE/'catalog.json')['rows']}

def prepare_attempt(pair):
    try:return prepare_one(CATALOG[pair]),None
    except Exception as e:
        value=dict(pair=pair,error=repr(e),traceback=traceback.format_exc());write(OUT/'errors'/f'prepare_{pair}.json',value);return None,value

def prepare(workers):
    setup()
    assert 1<=workers<=8
    pairs=read(ROOT/'experiments/p3_listen_v8_yield_audit_20261023/sample_pairs.json')
    write(OUT/'selection.json',dict(pairs=pairs,seed=2026102306,scope='Same 2048-pair audit sample; not a new favorable sample'))
    write(OUT/'contract.json',dict(revision=recovery.REVISION,baseline_contract_sha256=sha(BASE/'contract.json'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (SCRIPT,ROOT/'scripts/listen_v8_recovery.py')},
        selection_sha256=sha(OUT/'selection.json'),training_admitted=False))
    counts=collections.Counter();errors=[];remaining=[]
    for pair in pairs:
        path=OUT/'prepared'/(pair+'.json')
        if path.exists():
            r=read(path);counts.update(dict(pairs=1,old=r['old_clips'],new=r['new_clips'],novel=len(r['novel_clips'])))
        else:remaining.append(pair)
    with concurrent.futures.ProcessPoolExecutor(workers,initializer=prepare_init) as pool:
        for value,error in pool.map(prepare_attempt,remaining):
            if error:errors.append(error)
            else:counts.update({k:value[k] for k in ('old','new','novel')})
            counts['pairs']+=1
            if counts['pairs']%64==0:
                write(OUT/'prepare_progress.json',dict(utc=baseline.full.now(),counts=dict(counts),errors=len(errors)))
                print(json.dumps(dict(counts)),flush=True)
    write(OUT/'prepare_summary.json',dict(utc=baseline.full.now(),counts=dict(counts),errors=errors))

def screen(rank,world):
    setup();import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    c=read(OUT/'contract.json')
    assert c['baseline_contract_sha256']==sha(BASE/'contract.json')
    for f,digest in c['code_sha256'].items():assert sha(ROOT/f)==digest
    ast=baseline.full.legacy.EventModel(baseline.full.legacy.MODEL,batch_size=16)
    panns=baseline.full.legacy.PannsModel(batch_size=16)
    rejects=read(BASE/'human_rejections.json')['rows'];positives=read(BASE/'human_retained.json')['rows']
    feedback=read(BASE/'qc4_review_feedback.json')['rows'];counts=collections.Counter()
    for path in sorted((OUT/'prepared').glob('*.json'))[rank::world]:
        item=read(path);rows=[]
        assert sha(BASE/'results'/path.name)==item['old_result_sha256']
        try:
            for row in item['novel_clips']:
                # BASE_SCREEN calls the frozen audio checks, then the QC4 wrapper
                # uses the revised compatibility evidence for this new interval.
                r=baseline.assess(baseline.BASE_SCREEN(row,item['normalized'],ast,panns,rejects,positives),feedback)
                r['recovery']=recovery.analysis(r);rows.append(r)
            write(OUT/'results'/path.name,dict(pair=item['pair'],split=item['split'],clips=rows,complete=True))
            counts['pairs']+=1;counts['clips']+=len(rows)
        except Exception as e:
            counts['errors']+=1;write(OUT/'errors'/f'screen_{path.name}',dict(error=repr(e),traceback=traceback.format_exc()))
        write(OUT/'progress'/f'rank_{rank}.json',dict(utc=baseline.full.now(),counts=dict(counts),state='running'))
    write(OUT/'progress'/f'rank_{rank}.json',dict(utc=baseline.full.now(),counts=dict(counts),state='finished'))

def inventory():
    baseline.install();baseline.full.verify();OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted((BASE/'results').glob('*.json'));hours=collections.Counter();counts=collections.Counter();examples={};seen=collections.Counter();rng=random.Random(2026102307)
    def offer(group,row):
        seen[group]+=1;bucket=examples.setdefault(group,[])
        if len(bucket)<100:bucket.append(row)
        else:
            i=rng.randrange(seen[group])
            if i<100:bucket[i]=row
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for result in pool.map(read,paths):
            counts['pairs']+=1
            for r in result['clips']:
                a=recovery.analysis(r);h=r['duration_s']/3600;prefix=r['split']+'/'+r['admission_pool']+'/'
                hours[prefix+'baseline_'+r['release']['pool']]+=h
                if a['boundary_recovery_candidate']:
                    hours[prefix+'boundary_recovery_candidate']+=h;counts[prefix+'boundary_recovery_candidate']+=1
                    if r['split']=='train' and r['admission_pool']=='primary':offer('boundary_soft',dict(r,recovery=a))
                if a['identity_review_stratum']:
                    group=a['identity_review_stratum'];hours[prefix+group]+=h;counts[prefix+group]+=1
                    if r['split']=='train' and r['admission_pool']=='primary':offer(group,dict(r,recovery=a))
            if counts['pairs']%2000==0:write(OUT/'inventory_progress.json',dict(utc=baseline.full.now(),pairs=counts['pairs'],total=len(paths)))
    total=len(read(BASE/'catalog.json')['rows']);errors={p.stem for p in (BASE/'errors').glob('*.json')}-{p.stem for p in paths}
    write(OUT/'inventory_summary.json',dict(utc=baseline.full.now(),counts=dict(counts),hours=dict(hours),snapshot_pairs=len(paths),expected_pairs=total,unresolved_error_pairs=len(errors),pending_pairs=total-len(paths)-len(errors),
        note='Exact hours over the snapshotted completed pairs. Identity potential is not admitted; boundary soft candidates require review.'))
    write(OUT/'review_reservoirs.json',examples)
    print(json.dumps(dict(pairs=len(paths),hours=dict(hours))),flush=True)

def summarize():
    selection=read(OUT/'selection.json')['pairs'];hours=collections.Counter();counts=collections.Counter();review=[]
    for pair in selection:
        old=read(BASE/'results'/(pair+'.json'));path=OUT/'results'/(pair+'.json')
        if not path.exists():counts['missing_results']+=1;continue
        new=read(path)['clips'];counts['pairs']+=1
        base=[r for r in old['clips'] if r['release']['pool']=='candidate']
        for r in base:hours[r['split']+'/'+r['admission_pool']+'/baseline_candidate']+=r['duration_s']/3600
        used=[(r['start'],r['end']) for r in base]
        groups=[('boundary_recovery',[r for r in old['clips'] if recovery.analysis(r)['boundary_recovery_candidate']]),
            ('asr_recovery',[r for r in new if r['release']['pool']=='candidate']),
            ('asr_and_boundary_recovery',[r for r in new if recovery.analysis(r)['boundary_recovery_candidate']])]
        for group,rows in groups:
            for r in rows:
                added=(r['start'],r['end']);increment=recovery.incremental_seconds(used,[added]);prefix=r['split']+'/'+r['admission_pool']+'/'
                # Do not truncate whole sentences or double count replacement
                # intervals. A replacement is reported separately, not appended.
                if abs(increment-r['duration_s'])>1e-5:
                    hours[prefix+group+'_replacement_only_union_gain']+=increment/3600;continue
                used.append(added);hours[prefix+group]+=r['duration_s']/3600;counts[prefix+group]+=1
                if group.startswith('asr'):review.append(dict(r,recovery=recovery.analysis(r),recovery_group=group))
    write(OUT/'sample_gain_summary.json',dict(utc=baseline.full.now(),counts=dict(counts),hours=dict(hours),
        note='Increment after keeping existing candidates and then boundary-only candidates; overlapping replacements are not appended. No training admission.'))
    write(OUT/'asr_review_candidates.json',review)
    print(json.dumps(dict(counts=dict(counts),hours=dict(hours)),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','screen','inventory','summarize']);p.add_argument('--workers',type=int,default=8);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    elif a.stage=='screen':screen(a.rank,a.world)
    elif a.stage=='inventory':inventory()
    else:summarize()
