"""Prepare a small composition sample and still-excluded controls; no production changes."""
import collections,json,random
from pathlib import Path
import evaluate_listen_v8_recovery as e
import listen_v8_downward_crop as downward
import plan_listen_v8 as planner
from p3_listen.common import read,write

OUT=e.ROOT/'experiments/p3_listen_v8_downward_crop_preview_20261023'
PILOT=e.ROOT/'experiments/p3_listen_v8_head_shoulders_20261023'
# Explicit visual corrections for this preview, not a general head detector.
PREVIEW_Y_OFFSETS={
    'naturalistic_V02_S3018_I00000462_v8_s0_011':48,
    'naturalistic_V03_S1009_I00000147_v8_s1_003':-64,
    'naturalistic_V03_S0119_I00000129_v8_s1_007':-32,
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog={r['pair']:r for r in read(e.BASE/'catalog.json')['rows']}
    rows=[]
    for p in sorted((PILOT/'results').glob('*.json')):
        rows.extend(r for r in read(p)['clips'] if r['split']=='train' and r['admission_pool']=='primary'
            and r['listener_crop'].get('crop_recovery') and (r['release']['pool']=='candidate'
                or e.recovery.analysis(r)['boundary_recovery_candidate']))
    random.Random(2026102311).shuffle(rows)
    chosen=[];groups=set();failures=collections.Counter()
    for r in rows:
        if r['group_id'] in groups:continue
        item=catalog[r['pair']]
        record=item['a' if r['listener_index']==0 else 'b']
        _,g,_=planner.load_person(record)
        try:c=downward.crop(g,r['start'],r['end'],r['listener_crop']['old_tight_crop'])
        except ValueError as ex:failures[str(ex)]+=1;continue
        # Prefer reasonably sized faces for this first qualitative composition check.
        if c['output_face_height_median_px'] < 180:failures['preview_face_height_below_180']+=1;continue
        if r['sample_id'] in PREVIEW_Y_OFFSETS:
            dy=PREVIEW_Y_OFFSETS[r['sample_id']]
            assert 0<=c['y']+dy<=g['height']-c['size']
            c['automatic_proposal_y']=c['y'];c['y']+=dy
            c['preview_visual_adjustment']=dict(y_offset_px=dy,reason='Visually adjust bald-head whitespace or preserve hat/hair; preview only')
            c['estimated_top_space_fraction']-=dy/c['size']
        chosen.append(dict(r,listener_crop=c,recovery=e.recovery.analysis(r),training_admitted=False))
        groups.add(r['group_id'])
        if len(chosen)==6:break
    assert len(chosen)==6, (len(chosen),failures)
    write(OUT/'review_candidates.json',chosen)
    # Draw from the fixed audit sample, not from hand-picked familiar failures.
    rejected={k:[] for k in ('rejected_content','rejected_audio','rejected_boundary')}
    pairs=list(read(PILOT/'selection.json')['pairs']);random.Random(2026102312).shuffle(pairs)
    for pair in pairs:
        for r in read(e.BASE/'results'/(pair+'.json'))['clips']:
            if r['split']!='train' or r['admission_pool']!='primary' or r['group_id'] in groups:continue
            a=e.recovery.analysis(r);pool=r['release']['pool']
            kind=('rejected_content' if pool=='content_review' else
                'rejected_audio' if pool=='repair_reserve' and not a['identity_only_review_potential'] else
                'rejected_boundary' if pool=='boundary_review' and not a['boundary_recovery_candidate'] else None)
            if kind is None or len(rejected[kind])>=2:continue
            rejected[kind].append(dict(r,recovery=a,review_kind=kind,training_admitted=False))
            groups.add(r['group_id'])
        if all(len(v)==2 for v in rejected.values()):break
    assert all(len(v)==2 for v in rejected.values())
    write(OUT/'rejected_controls.json',[r for v in rejected.values() for r in v])
    write(OUT/'summary.json',dict(crop_samples=len(chosen),rejected_counts={k:len(v) for k,v in rejected.items()},
        preview_selection_skips=dict(failures),training_admitted=False,
        note='Qualitative composition examples and stratified exclusions; not a prevalence or capacity estimate.'))
    print(json.dumps(read(OUT/'summary.json')),flush=True)

if __name__=='__main__':main()
