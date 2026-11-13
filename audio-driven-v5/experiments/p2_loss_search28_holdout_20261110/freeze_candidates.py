"""Select final-confirmation candidates without inspecting model quality scores.

Exclude training/tuning source components and any locally found historical
generated/latent clip. This audit describes its exact filesystem scope; it does
not claim knowledge of outputs stored outside the workspace or deleted history.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import random
import sqlite3
import sys
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
EVAL=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
POOL=ROOT/'experiments/seamless_quick200_latest_20261105/candidates.latest.jsonl'
SOURCE=ROOT/'experiments/p2_ltx_event_current_sources_v2_20261106'
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from p2_listen.contracts import read,sha,atomic_json
from evaluation.seamless.bind import load_references


def main():
    accepted=read(EVAL/'VALIDATION_MEMBERSHIP_ACCEPTED.json')
    if sha(POOL)!=accepted['sources'][str(POOL)]:raise ValueError('Candidate pool changed')
    trainfile=SOURCE/'sample_list.csv'
    if sha(trainfile)!=accepted['sources'][str(trainfile)]:raise ValueError('Training list changed')
    with trainfile.open()as f:training={r['sample_id']for r in csv.DictReader(f)}
    with sqlite3.connect((SOURCE/'sources.sqlite').resolve().as_uri()+'?mode=ro',uri=True)as conn:
        train_groups={r[0]for r in conn.execute("SELECT DISTINCT json_extract(value,'$.source.group_id') FROM samples")}
    tuning=read(EVAL/'tuning200.json');tune_ids={r['id']for r in tuning};tune_groups={r['group_id']for r in tuning}
    all_rows=[json.loads(s)for s in POOL.read_text().splitlines()if s.strip()]
    candidates=[r for r in all_rows if r.get('split')=='test'and r.get('group_id')and
        r['group_id']not in train_groups|tune_groups and r['sample_id']not in training|tune_ids]
    ids={r['sample_id']for r in candidates}
    # Bounded audit of known experiment layouts, not a recursive workspace scan.
    roots=sorted(p for p in (ROOT/'experiments').glob('p2_*')if p.is_dir())
    extra=[]
    for root in roots:
        extra.extend(p for pattern in ('evaluation/step_*','evaluations/*')for p in root.glob(pattern)if p.is_dir())
    roots=sorted(set(roots+extra));exposure={};scanned=[]
    for root in roots:
        for category in ('generated','latents'):
            base=root/category
            if not base.is_dir():continue
            folders=set(base.glob('*/seamless'))
            if(base/'seamless').is_dir():folders.add(base/'seamless')
            # Some trial runners have a flat per-ID directory.
            folders.add(base)
            for folder in sorted(folders):
                scanned.append(str(folder))
                for item in folder.iterdir():
                    if item.name in ids and item.is_dir() and any((item/f).exists()for f in ('metadata.json','complete.json','generated.mp4','payload.pt')):
                        exposure.setdefault(item.name,[]).append(str(item))
    available=[r for r in candidates if r['sample_id']not in exposure]
    parent=Path(read(ROOT/'experiments/seamless_quick200_latest_20261105/prepared_200/seal.json')['parent_prepared'])
    refs={r['id']:r for r in load_references(parent)}
    unavailable=[r['sample_id']for r in available if r['sample_id']not in refs]
    available=[r for r in available if r['sample_id']in refs]
    buckets=collections.defaultdict(list)
    for row in sorted(available,key=lambda r:r['sample_id']):
        duration='short'if row['duration_s']<4 else'mid'if row['duration_s']<8 else'long'
        buckets[row['group_id'],bool(row['listener_audio_is_silent']),duration].append(row)
    rng=random.Random(137)
    for values in buckets.values():rng.shuffle(values)
    chosen=[];used_pairs=set()
    while any(buckets.values())and len(chosen)<128:
        for key in sorted(buckets):
            values=buckets[key]
            while values and values[-1]['pair']in used_pairs:values.pop()
            if values and len(chosen)<128:
                row=values.pop();chosen.append(row);used_pairs.add(row['pair'])
    if not chosen:raise ValueError('No unexposed holdout candidates in audited scope')
    assert len({r['sample_id']for r in chosen})==len(chosen)==len(used_pairs)
    assert not({r['group_id']for r in chosen}&(train_groups|tune_groups))
    atomic_json(WORK/'candidates.json',chosen,immutable=True)
    atomic_json(WORK/'historical_exposure.json',exposure,immutable=True)
    atomic_json(WORK/'parent_references.json',[refs[r['sample_id']]for r in chosen],immutable=True)
    by_id={r['sample_id']:r for r in candidates}
    historical_pairs={by_id[s]['pair']for s in exposure}
    historical_groups={by_id[s]['group_id']for s in exposure}
    report=dict(success=True,at=time.time(),training_samples=len(training),tuning_samples=len(tuning),
        disjoint_component_pool=len(candidates),locally_exposed_clips=len(exposure),
        absent_from_prepared_parent=unavailable,selected_clips=len(chosen),selected_pairs=len(used_pairs),
        selected_components=len({r['group_id']for r in chosen}),
        components=dict(collections.Counter(r['group_id']for r in chosen)),
        silent_clips=sum(r['listener_audio_is_silent']for r in chosen),
        scope='Known local p2 experiment generated/latent layouts, including evaluation/step_* and evaluations/*; no assertion about deleted or externally stored history',
        scanned_directories=scanned,source_roots=len(roots),selection_seed=137,
        independent_of_current_tuning=True,model_quality_scores_used=False,
        selected_with_historical_pair_exposure=sum(r['pair']in historical_pairs for r in chosen),
        selected_with_historical_component_exposure=sum(r['group_id']in historical_groups for r in chosen),
        globally_unexposed_final_test_accepted=False,
        parent_prepared=str(parent),parent_seal_sha256=sha(parent/'seal.json'),
        selected_sha256=sha(WORK/'candidates.json'),references_sha256=sha(WORK/'parent_references.json'),
        final_test_ready=False,pending=['latest-media/crop/audio compatibility','source media canary reads','conditioning preparation','strict historical source independence not established'],
        limitation='Clips are not independent observations when they share a source component; final uncertainty must respect component grouping')
    atomic_json(WORK/'CANDIDATE_AUDIT.json',report)
    print(json.dumps({k:v for k,v in report.items()if k!='scanned_directories'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
