"""CPU replay of the exact64/28-shard binding using historical completed media.

No new checkpoint inference and no candidate score. Historical media identities
and steps stay unchanged; only the fixture directory and selected population vary.
"""
from pathlib import Path
import copy
import os
import shutil
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
EVAL=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
OLD=ROOT/'experiments/p2_ltx_all_lora7500_seamless200_sft2752_gpu7_20261109'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse'),str(EVAL)]
from p2_listen.contracts import read,sha,atomic_json
from p2_testsets_eval.protocol import partition
from p2_testsets_eval.runtime import ready,complete
from binder import ShardBinder


def main():
    ids={r['id']for r in read(EVAL/'tuning64.json')}
    p=copy.deepcopy(read(OLD/'protocol.json'));arm=p['arms'][0]['name']
    items=[r for r in read(OLD/'shared_items.json')if r['dataset']=='seamless'and r['id']in ids]
    assert len(items)==64
    p.update(workers=28,total_samples=64,total_outputs=64,
        datasets=dict(seamless=dict(p['datasets']['seamless'],prepared=str(EVAL/'prepared64'),
            references_sha256=sha(EVAL/'prepared64/references.jsonl'),count=64)),
        audit_only=True,not_a_search_candidate_result=True)
    atomic_json(WORK/'protocol.json',p,immutable=True)
    atomic_json(WORK/'shared_items.json',items,immutable=True)
    for i,values in enumerate(partition(items,28)):atomic_json(WORK/'shards'/f'{i:02d}.json',values,immutable=True)
    for item in items:
        source=OLD/'generated'/arm/item['key'];target=WORK/'generated'/arm/item['key']
        assert ready(source)
        target.mkdir(parents=True,exist_ok=True)
        for name in ('generated.mp4','listener.wav'):
            if not(target/name).exists():os.link(source/name,target/name)
            assert(target/name).stat().st_ino==(source/name).stat().st_ino
        if not(target/'metadata.json').exists():shutil.copy2(source/'metadata.json',target/'metadata.json')
        assert sha(target/'metadata.json')==sha(source/'metadata.json')
        if not ready(target):complete(target,[target/name for name in ('generated.mp4','listener.wav','metadata.json')])
    binder=ShardBinder(WORK);binder.refresh();binder.publish_ready()
    assert len(binder.committed)==64 and len(binder.published)==28
    binder.canonical_audit()
    atomic_json(WORK/'ACCEPTED.json',dict(success=True,at=time.time(),historical_fixture=str(OLD),
        samples=64,shards=28,original_model_step=7500,no_new_generation=True,no_candidate_scores=True,
        generated_and_gt_media_unchanged=True,own_generated_audio_bound=True,
        progressive_matches_native_full_binding=True,
        source={str(f):sha(f)for f in [Path(__file__),EVAL/'binder.py',EVAL/'prepared64/references.jsonl']}))
    print('64 historical media pairs /28 shards match native full binding; no search generation claimed',flush=True)


if __name__=='__main__':main()
