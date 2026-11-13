"""Export only a fixed small-batch review selection; never launches screening."""
import concurrent.futures
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import publish_listen_v7_sample as exporter
from p3_listen.common import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'data_corpora/seamless_listen_v8_full'
EXP=ROOT/'experiments/p3_listen_v8_load_tuning_20261023'
REPAIR=ROOT/'experiments/p3_listen_v8_error_repair_20261023'
OUT=ROOT/'data_previews/seamless_listen_v8_repair'
SEED=20261023


def select():
    queue=read(EXP/'small_batch_queue.json')['pairs']
    assert len(queue)==128
    rows=[];failures=[];results=[]
    for pair in queue:
        path=CORPUS/'results'/(pair+'.json')
        if path.exists():
            result=read(path);assert result['complete'];results.append(result);rows.extend(result['clips'])
        else:
            failures.append(read(CORPUS/'errors'/(pair+'.json')))
    selected=[];used=set();used_pairs=set();rng=random.Random(SEED)
    omitted=[]
    def add(row,kind):
        selected.append(dict(row,review_kind=kind,preview=f"media/{row['sample_id']}/"))
        used.add(row['sample_id']);used_pairs.add(row['pair'])
    for repaired in read(REPAIR/'failed_pair_validation.json'):
        if repaired['status']!='planned':continue
        result=read(CORPUS/'results'/(repaired['pair']+'.json'))
        conflict=repaired['overlap_units'][0];omitted.append(dict(pair=repaired['pair'],**conflict))
        near=sorted(result['clips'],key=lambda c:min(abs(c['start']-conflict['start']),abs(c['end']-conflict['end'])))
        for row in near[:2]:add(row,'repaired')
    def pick(predicate,n,kind):
        pool=[r for r in rows if predicate(r) and r['sample_id'] not in used]
        rng.shuffle(pool);pool.sort(key=lambda r:r['pair'] in used_pairs)
        assert len(pool)>=n,(kind,len(pool),n)
        for row in pool[:n]:add(row,kind)
    pick(lambda r:r['v8_combined_status']=='exclude',1,'exclude')
    pick(lambda r:r['duration_s']>10,3,'long')
    pick(lambda r:r['duration_s']<3,2,'short')
    pick(lambda r:r['v8_combined_status']=='candidate',6,'candidate')
    pick(lambda r:r['audio_qc_status']=='keep' and r['v8_combined_status']=='review',3,'boundary_review')
    pick(lambda r:r['audio_qc_status']=='review',3,'audio_review')
    assert len(selected)==24 and len(used)==24
    start=read(EXP/'after_small_batch_contract.json')
    summary=dict(source_pairs=128,completed_pairs=len(results),error_pairs=len(failures),
        candidate_clips=len(rows),combined_counts=dict(Counter(r['v8_combined_status'] for r in rows)),
        audio_counts=dict(Counter(r['audio_qc_status'] for r in rows)),review_samples=24,
        new_batch_samples=18,repaired_source_samples=6,workers=8,prep_slots=8,
        processing_contract_revision=start['revision'],full_run_waiting_for_user=True,
        failed_pairs=[dict(pair=f['pair'],error=f['error']) for f in failures])
    write(OUT/'selection.json',dict(seed=SEED,summary=summary,samples=selected,omitted_overlap_units=omitted,
        note='18 stratified small-batch clips plus 6 recovered clips from the 3 repaired pairs; not a prevalence estimate.',
        source_results_sha256={r['pair']:sha(CORPUS/'results'/(r['pair']+'.json')) for r in selected}))
    return selected,summary,omitted


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected,summary,omitted=select()
    exporter.OUT=OUT
    def export(row):
        marker=OUT/row['preview']/'export.json'
        if marker.exists():
            old=read(marker)
            if old.get('media_verified') and all(old[k]==row[k] for k in ('start','end','sources','listener_crop','normalized_audio')):
                if all(sha(OUT/row['preview']/(mode+'.mp4'))==v['sha256'] for mode,v in old['media_checks'].items()):
                    return dict(old,review_kind=row['review_kind'])
        return exporter.export(row)
    # Four exports, each FFmpeg with at most two codec threads; no GPU models.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        samples=list(pool.map(export,selected))
    write(OUT/'samples.json',dict(summary=summary,samples=samples,omitted_overlap_units=omitted,
        selection_sha256=sha(OUT/'selection.json'),training_admitted=False))
    shutil.copyfile(ROOT/'scripts/listen_v8_repair_review.html',OUT/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    text=(OUT/'index.html').read_text()+(OUT/'samples.json').read_text()
    source=Path(os.environ.get('LISTEN_REVIEW_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    if source.exists():
        font=TTFont(source,fontNumber=2)
        sub=subset.Subsetter();sub.populate(text=text);sub.subset(font)
        font.flavor='woff';font.save(OUT/'review.woff')
    font=TTFont(OUT/'review.woff')
    missing={ord(c) for c in text if '\u4e00'<=c<='\u9fff'}-set(font.getBestCmap())
    if missing:raise RuntimeError('Review font lacks Chinese glyphs; set LISTEN_REVIEW_FONT to NotoSansCJK-Regular.ttc')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_v8_repair'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    assert link.resolve()==OUT.resolve()
    print(json.dumps(dict(url='http://127.0.0.1:8875/listen_v8_repair/',**summary)),flush=True)


if __name__=='__main__':main()
