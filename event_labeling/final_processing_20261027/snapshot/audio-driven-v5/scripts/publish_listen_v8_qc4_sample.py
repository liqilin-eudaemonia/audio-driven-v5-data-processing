"""Publish a bounded, reproducible QC4 review; never authorize full processing."""
import concurrent.futures
from collections import Counter
import json, os, random, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_listen.common import read,write,sha
import publish_listen_v7_sample as exporter
ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'data_corpora/seamless_listen_v8_qc4_sample'
OLD=ROOT/'data_previews/seamless_listen_v8_repair'
OUT=ROOT/'data_previews/seamless_listen_v8_qc4'
EXP=ROOT/'experiments/p3_listen_v8_qc4_sample'
SEED=2026102303

def main():
    runtime=read(CORPUS/'runtime.json')
    assert runtime['state']=='awaiting_user_review' and runtime['workers']==0
    rows=[r for f in sorted((CORPUS/'results').glob('*.json')) for r in read(f)['clips']]
    rng=random.Random(SEED);rng.shuffle(rows)
    selected=[];used=set();groups=set()
    def add(r,kind):
        selected.append(dict(r,review_kind=kind,preview=f"media/{r['sample_id']}/"))
        used.add(r['sample_id']);groups.add(r['group_id'])
    def pick(predicate,n,kind):
        eligible=[r for r in rows if predicate(r) and r['sample_id'] not in used]
        # Prefer explicit transcribed reactions within the human-evidence stratum.
        eligible.sort(key=lambda r:not bool(r['listener_events']))
        picked=0
        for r in eligible:
            if r['group_id'] in groups:continue
            add(r,kind);picked+=1
            if picked==n:break
        assert picked==n,(kind,picked,n)
    pick(lambda r:r['qc4']['status']=='candidate' and r['listener_human_evidence'],7,'candidate_voice')
    pick(lambda r:r['qc4']['status']=='candidate' and not r['listener_human_evidence'],5,'candidate_quiet')
    pick(lambda r:r['qc4']['status']=='review' and r['qc4']['task_reasons'],2,'contrast')
    pick(lambda r:r['qc4']['status']=='review' and r['qc4']['reference']['reasons'],2,'contrast')
    pick(lambda r:r['qc4']['status']=='review' and r['qc4']['speech_identity_windows'],1,'contrast')
    pick(lambda r:r['qc4']['status']=='exclude',1,'contrast')
    feedback=read(ROOT/'data_reviews/listener_v8_repair_feedback.json')
    assert sha(OLD/'samples.json')==feedback['sample_manifest_sha256']
    oldrows={r['sample_id']:r for r in read(OLD/'samples.json')['samples']}
    regression={r['sample_id']:r['qc4'] for r in read(EXP/'development_regression.json')}
    for f in feedback['rows']:
        r=dict(oldrows[f['sample_id']],qc4=regression[f['sample_id']],prior_feedback=f)
        assert r['qc4']['automatic_status']=='review'
        add(r,'known')
    assert len(selected)==24 and len(used)==24
    summary=dict(runtime['summary'],review_counts=dict(Counter(r['review_kind'] for r in selected)),
        fresh_review_samples=18,known_problem_samples=6,workers_used=8,prep_slots=8,
        candidate_human_evidence_counts=dict(Counter(str(bool(r['listener_human_evidence'])) for r in rows if r['qc4']['status']=='candidate')))
    manifest=dict(seed=SEED,summary=summary,samples=selected,
        source_results_sha256={r['pair']:sha(CORPUS/'results'/(r['pair']+'.json')) for r in selected if r['review_kind']!='known'},
        source_selection_sha256=sha(CORPUS/'selection.json'),feedback_sha256=sha(ROOT/'data_reviews/listener_v8_repair_feedback.json'),
        note='18 fresh clips from 18 distinct groups plus 6 known development problems. Stratified review, not prevalence or accuracy estimation.')
    write(OUT/'selection.json',manifest)
    exporter.OUT=OUT
    def export(r):
        if r['review_kind']=='known':
            folder=OUT/r['preview'];folder.mkdir(parents=True,exist_ok=True)
            for src in (OLD/r['preview']).iterdir():
                if src.name=='export.json':continue
                dest=folder/src.name
                if not dest.exists():os.link(src,dest)
            for mode,check in r['media_checks'].items():assert sha(folder/(mode+'.mp4'))==check['sha256']
            result=dict(r,media_reused_without_changes=True)
            write(folder/'export.json',result);return result
        marker=OUT/r['preview']/'export.json'
        if marker.exists():
            old=read(marker)
            if old.get('media_verified') and all(old[k]==r[k] for k in ('start','end','sources','listener_crop','normalized_audio')):
                if all(sha(OUT/r['preview']/(mode+'.mp4'))==check['sha256'] for mode,check in old['media_checks'].items()):return dict(old,**r)
        return exporter.export(r)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:samples=list(pool.map(export,selected))
    write(OUT/'samples.json',dict(summary=summary,samples=samples,selection_sha256=sha(OUT/'selection.json'),training_admitted=False,full_run_authorized=False))
    shutil.copyfile(ROOT/'scripts/listen_v8_qc4_review.html',OUT/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    content=(OUT/'index.html').read_text()+(OUT/'samples.json').read_text()
    source=os.environ.get('LISTEN_REVIEW_FONT','/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=TTFont(source,fontNumber=2);sub=subset.Subsetter();sub.populate(text=content);sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    assert not ({ord(c) for c in content if '\u4e00'<=c<='\u9fff'}-set(font.getBestCmap()))
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_v8_qc4'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    assert link.resolve()==OUT.resolve()
    print(json.dumps(dict(url='http://127.0.0.1:8875/listen_v8_qc4/',summary=summary),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
