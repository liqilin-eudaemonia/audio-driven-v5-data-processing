"""Record review feedback and re-route unseen, already-scored source pairs."""
import collections,copy,json,random
from pathlib import Path
import evaluate_listen_v8_recovery as e
import listen_v8_audio_final as policy
from p3_listen.common import read,write,sha

ROOT=e.ROOT;OUT=ROOT/'experiments/p3_listen_v8_audio_final_20261023'
PREVIEW=ROOT/'data_previews/seamless_listen_v8_audio_final'
FEEDBACK=ROOT/'data_reviews/listener_v8_recovery_feedback_20261023.json'
KINDS={'keep_control':'保留候选 · 常规对照','keep_relaxed_noise':'保留候选 · 轻微底噪/单个弱提示',
    'keep_relaxed_match':'保留候选 · 短暂较弱的匹配','reject_crosstalk':'筛掉对照 · 重复或较强串音',
    'reject_noise':'筛掉对照 · 明显杂音','audio_review':'暂不放行 · 证据仍不明确'}

def record_feedback():
    manifest=ROOT/'data_previews/seamless_listen_v8_recovery/samples.json';data=read(manifest)
    expected={28:'naturalistic_V03_S0314_I00000055_v8_s0_004',33:'improvised_V03_S0371_I00000582_v8_s0_002',34:'naturalistic_V03_S0497_I00000050_v8_s0_004',
        35:'naturalistic_V03_S0200_I00000125_v8_s0_000',36:'naturalistic_V03_S1788_I00000202_v8_s1_006'}
    for n,sid in expected.items():assert data['samples'][n-1]['sample_id']==sid
    rows=[]
    for n,r in enumerate(data['samples'],1):
        f={k:r[k] for k in ('sample_id','pair','speaker_id','listener_id','start','end')}
        f.update(page_number=n,disposition='repair_reserve' if n in (28,33,34) else 'accepted_exact',
            note='User: speaker bleed later in clip; poor audio' if n==28 else
                'User: audio too poor; exclude' if n in (33,34) else
                'User: may retain; boundary less important than audio quality' if n in (35,36) else
                'User: may retain for now' if n in (31,32) else 'User: all other samples may be retained')
        rows.append(f)
    value=dict(review='v8-recovery-comparison',sample_manifest=str(manifest),sample_manifest_sha256=sha(manifest),
        selection_sha256=data['selection_sha256'],rows=rows,priority='audio_quality_first',
        next_step='New small batch with retained and excluded samples before focusing on crop screening',
        full_rollout_of_new_policy_authorized=False,training_authorized=False)
    if FEEDBACK.exists():assert read(FEEDBACK)==value
    else:write(FEEDBACK,value)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
    feedback=record_feedback();old_feedback=read(e.BASE/'qc4_review_feedback.json')['rows'];latest=read(ROOT/'data_reviews/listener_v8_audio_first_feedback_20261023.json')['rows'];all_feedback=old_feedback+feedback+latest
    e.baseline.install();e.baseline.full.verify()
    # All historical displayed examples are excluded; classification uses cached,
    # unchanged waveform/model evidence. This is a routing change, not a new model.
    seen_pairs=set();seen_groups=set();seen_files=[]
    def remember(value):
        if isinstance(value,dict):
            if 'pair' in value:seen_pairs.add(value['pair'])
            if 'group_id' in value:seen_groups.add(value['group_id'])
            for child in value.values():remember(child)
        elif isinstance(value,list):
            for child in value:remember(child)
    for path in sorted((ROOT/'data_previews').glob('*/samples.json')):
        if path.parent==PREVIEW:continue
        value=read(path);rows=value.get('samples',value.get('cases',[])) if isinstance(value,dict) else value
        remember(rows)
        seen_files.append(dict(path=str(path),sha256=sha(path)))
    population=[p for p in read(e.OUT/'selection.json')['pairs'] if p not in seen_pairs]
    random.Random(2026102315).shuffle(population);pairs=population[:1024]
    write(OUT/'selection.json',dict(pairs=pairs,seed=2026102315,previously_displayed_manifests=seen_files,
        scope='1024 not-previously-displayed pairs from the fixed random recovery audit; cached QC3/QC4 evidence with repaired planner intervals'))
    counts=collections.Counter();hours=collections.Counter();eligible=[];provenance=[]
    for pair in pairs:
        oldpath=e.BASE/'results'/(pair+'.json');newpath=e.OUT/'results'/(pair+'.json');auditpath=e.OUT/'audit'/(pair+'.json')
        old=read(oldpath);new=read(newpath);audit=read(auditpath)
        scored={e.key(r):r for r in old['clips']+new['clips']};oldkeys={e.key(r) for r in old['clips']}
        provenance.append(dict(pair=pair,old_sha256=sha(oldpath),recovery_sha256=sha(newpath),audit_sha256=sha(auditpath)))
        routed=[]
        for plan in audit['clips']:
            key=e.key(plan);assert key in scored,(pair,key)
            r=copy.deepcopy(scored[key])
            for field in ('listener_compatibility','boundary_review_flags','asr_gap_tail_exemptions'):
                if field in plan:r[field]=plan[field]
            r['audio_first']=policy.route(r,all_feedback);r['recovery']=e.recovery.analysis(r)
            r['new_asr_interval']=key not in oldkeys;r['training_admitted']=False
            routed.append(r);prefix=r['split']+'/'+r['admission_pool']+'/'+r['audio_first']['pool']
            counts[prefix]+=1;hours[prefix]+=r['duration_s']/3600
            if r['split']!='train' or r['admission_pool']!='primary' or r['group_id'] in seen_groups:continue
            q=r['audio_first']
            if q['pool']=='keep_candidate':
                soft=set(q['quality']['soft_reasons'])
                kind='keep_relaxed_match' if soft & {'brief_weak_reference_match','brief_weak_local_match'} else (
                    'keep_relaxed_noise' if soft & {'mild_stationary_background','isolated_speaker_music_prediction'} else 'keep_control')
            elif q['pool']=='audio_reject':
                kind='reject_crosstalk' if any('match' in x or 'leakage' in x for x in q['reasons']) else 'reject_noise'
            elif q['pool']=='audio_review':kind='audio_review'
            else:continue
            eligible.append(dict(r,review_kind=kind,review_label=KINDS[kind]))
        write(OUT/'results'/(pair+'.json'),dict(pair=pair,clips=routed,complete=True))
    rng=random.Random(2026102316);rng.shuffle(eligible);selected=[]
    # Independent strata; the displayed keep/reject ratio is deliberately balanced,
    # not an estimate of acceptance frequency.
    choices={}
    for kind in KINDS:
        rows=[r for r in eligible if r['review_kind']==kind]
        choices[kind]={}
        for r in rows:
            choices[kind].setdefault(r['group_id'],r)
    # Joint matching avoids exhausting a rare stratum's identity components
    # while selecting an earlier, abundant stratum.
    assigned={}
    def match(slot,visited):
        for group in choices[slot[0]]:
            if group in visited:continue
            visited.add(group)
            if group not in assigned or match(assigned[group],visited):
                assigned[group]=slot;return True
        return False
    slots=[(kind,i) for kind in KINDS for i in range(4)]
    for slot in slots:assert match(slot,set()),(slot,{k:len(v) for k,v in choices.items()})
    by_slot={slot:group for group,slot in assigned.items()}
    for slot in slots:
        r=choices[slot[0]][by_slot[slot]];r['preview']=f"media/{r['sample_id']}/";selected.append(r)
    write(OUT/'review_selection.json',dict(samples=selected,training_admitted=False,seed=2026102316,
        note='24 distinct identity components, excluded from prior displayed components. Four samples per stratum, not prevalence.'))
    regression=[]
    for r,f in zip(read(ROOT/'data_previews/seamless_listen_v8_recovery/samples.json')['samples'],feedback):
        automatic=policy.route(r,[]);final=policy.route(r,all_feedback)
        expected='audio_reject' if f['disposition']=='repair_reserve' else 'keep_candidate'
        assert final['pool']==expected,(f['page_number'],final)
        regression.append(dict(page_number=f['page_number'],sample_id=r['sample_id'],expected=expected,
            automatic=automatic,with_human_feedback=final))
    current_rows=read(ROOT/'data_previews/seamless_listen_v8_audio_first/samples.json')['samples']
    for f in latest:
        r=next(r for r in current_rows if r['sample_id']==f['sample_id'])
        automatic=policy.route(r,[]);final=policy.route(r,all_feedback)
        expected='audio_reject' if f['disposition']=='repair_reserve' else 'keep_candidate'
        assert final['pool']==expected,(f['page_number'],final)
        regression.append(dict(source='audio_first',page_number=f['page_number'],sample_id=r['sample_id'],expected=expected,
            automatic=automatic,with_human_feedback=final))
    write(OUT/'reviewed_regression.json' ,dict(rows=regression,note='Automatic results and exact human overrides reported separately; this is not independent validation.'))
    write(OUT/'summary.json',dict(pairs=len(pairs),counts=dict(counts),hours=dict(hours),preview_samples=len(selected),
        preview_groups=collections.Counter(r['review_kind'] for r in selected),cached_model_evidence=True,
        framing_unchanged=True,training_admitted=False))
    sources=[Path(__file__),ROOT/'scripts/listen_v8_audio_first.py',ROOT/'scripts/listen_v8_audio_final.py']
    write(OUT/'contract.json',dict(revision=policy.REVISION,policy=policy.POLICY,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        feedback_sha256=sha(FEEDBACK),latest_feedback_sha256=sha(ROOT/'data_reviews/listener_v8_audio_first_feedback_20261023.json'),baseline_contract_sha256=sha(e.BASE/'contract.json'),selection_sha256=sha(OUT/'selection.json'),
        review_selection_sha256=sha(OUT/'review_selection.json'),evidence=provenance,training_admitted=False))
    print(json.dumps(read(OUT/'summary.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
