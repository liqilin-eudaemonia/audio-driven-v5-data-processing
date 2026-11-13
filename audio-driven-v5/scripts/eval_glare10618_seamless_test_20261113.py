"""GLARE10618, current full Seamless6624, node11 physical GPU4 only.

Reuse verified native resident inference, complete audio/prosody caches and
paired scorer; independently generate every prediction for this checkpoint.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT.parent/'OmniResponse')]
import eval_glare_seamless_test_20261112 as native
from p2_testsets_eval.protocol import read, rows, sha

WORK = ROOT/'experiments/glare10618_seamless_test6624_20261113'
PARENT = ROOT/'experiments/glare19050_seamless_test6624_20261112'
CURRENT_INPUTS = ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112'
SELF = Path(__file__).resolve()
CHECKPOINT = ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111/before_extension20epochs.pt'
SOURCE_VERSION = 'seamless-upright-evaluation-20261112'
native.WORK = WORK
native.SELF = SELF
core = native.core
core.WORK = WORK
core.SELF = SELF
core.CHECKPOINT = CHECKPOINT
core.MODEL = 'GLARE step10618'
core.ARM = 'glare_step10618'
core.claim = native.claim
save = core.save


def annotated_save(path, value):
    if Path(path).name.endswith('_RESULTS.json') or Path(path) == WORK/'COMPLETE.json':
        value = dict(value, source_version=SOURCE_VERSION,
            historical_glare_comparison='33 source identities updated relative to historical19050/32940 full-test runs; see protocol')
    save(path, value)


native.save = annotated_save


def prepare():
    import torch
    import numpy as np
    from PIL import Image
    WORK.mkdir(exist_ok=True)
    if (WORK/'PREPARED.json').exists():
        p = read(WORK/'protocol.json')
        assert p['checkpoint_sha256'] == sha(CHECKPOINT) and p['source_version'] == SOURCE_VERSION
        return
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['step'] == 10618
    del checkpoint
    assert read(PARENT/'COMPLETE.json')['passed']
    parent = read(PARENT/'protocol.json')
    current = read(ROOT/'evaluation/SEAMLESS_CURRENT.json')
    assert current['version'] == SOURCE_VERSION
    prepared = Path(current['prepared_full6624']['prepared'])
    references = rows(prepared/'references.jsonl')
    assert len(references) == len({x['id'] for x in references}) == 6624
    assert sha(prepared/'references.jsonl') == read(prepared/'seal.json')['references_sha256']
    refs = {x['id']: x for x in references}
    assert set(refs) == {x['id'] for x in parent['samples']}
    changed = set(current['prepared_full6624']['replaced_ids'])
    assert len(changed) == 33
    items = {x['id']: x for x in read(CURRENT_INPUTS/'shared_items.json')}
    assert set(items) == set(refs)
    assert read(CURRENT_INPUTS/'INPUT_COMPLETE.json')
    source = PARENT/'prosody'
    target = WORK/'prosody'
    if not target.exists():target.symlink_to(source, target_is_directory=True)
    _, key = core.prosody_id()

    def one(source):
        old = read(source['ready'])['sample']
        uid = old['id']; reference = refs[uid]
        assert old['frames'] == reference['frames'] and old['duration'] == reference['duration_s']
        oldrow = old['item']['reference']['gt_reference']['row_sha256']
        newrow = reference['gt_reference']['row_sha256']
        assert (oldrow != newrow) == (uid in changed)
        folder = CURRENT_INPUTS/'conditions'/items[uid]['key']
        image = folder/'first_frame.png'
        audit = read(folder/'input_audit.json')
        assert audit['source_only'] and not audit['target_description_in_model_input']
        assert not audit['listener_future_media_in_model_input']
        audio = Path(old['audio'])
        assert sha(audio) == old['audio_sha256']
        image_sha = sha(image)
        if uid in changed:
            before = read(old['source_input_audit'])
            assert all(audit.get(k) == before.get(k) for k in
                       ['speaker_audio_sha256','speaker_first_sample','speaker_last_sample'])
        elif image_sha != old['reference_sha256']:
            with Image.open(image) as a, Image.open(old['reference']) as b:
                assert np.array_equal(np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))), uid
        s = dict(id=uid, key='seamless/'+uid, dataset='seamless', reference=str(image),
            reference_sha256=image_sha, audio=str(audio), audio_sha256=old['audio_sha256'],
            ready=str(WORK/'inputs'/uid/'READY.json'), duration=old['duration'], frames=old['frames'],
            fps=25, size=512, quick200=old['quick200'], item=items[uid], evaluation_reference=reference,
            source_input_audit=str(folder/'input_audit.json'), source_version=SOURCE_VERSION,
            input_audio_reused_from=old['ready'], source_updated=uid in changed)
        assert core.cache_path(s,key).exists(), uid
        return s

    samples = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample in pool.map(one, parent['samples']):
            samples.append(sample)
            if len(samples) % 128 == 0:
                save(WORK/'PREPARE_STATUS.json',dict(at=time.time(),verified=len(samples),total=6624))
    p = {k: copy.deepcopy(v) for k,v in parent.items() if k not in
         ['samples','code','reused_completed200','expand_generation_after','expanded_generation_workers']}
    p.update(schema='GLARE10618-current-Seamless-full6624-node11-GPU4-v1',created=time.time(),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),step=10618,model=core.MODEL,
        physical_gpu=4,physical_gpus=[4],gpu_uuid=core.UUID,source_version=SOURCE_VERSION,
        samples=samples,references=str(prepared/'references.jsonl'),references_sha256=sha(prepared/'references.jsonl'),
        seal_sha256=sha(prepared/'seal.json'),source_updated_ids=sorted(changed),
        total=6624,generation_workers=4,decode_batch=4,generation_min_free_mib=24*1024,
        score_workers_during_generation=2,score_workers_after_generation=8,
        reused_generated=0,reused_all_inputs_and_prosody=str(PARENT),
        input_reuse='All6624 speaker WAVs and official intensity caches reused; all listener frame0 and GT use current canonical sources',
        historical_comparability='33 source identities differ from historical GLARE19050/32940 full-test evaluations',
        acceleration='GPU4 only:4 resident native generators,22GiB each,decode4;2 scorers during generation,8 afterward;batch64/GPU faces/emotion/pixel4;GT feature cache reused',
        code={str(f):sha(f) for f in [SELF,ROOT/'scripts/eval_glare_seamless_test_20261112.py',
            ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py']})
    save(WORK/'protocol.json',p);protocol_sha=sha(WORK/'protocol.json')
    for s in samples:
        save(Path(s['ready']),dict(sample=s,reused_audio=True,reused_current_reference=True,source_only=True,protocol_sha256=protocol_sha))
    blocks = [samples[i:i+25] for i in range(0,200,25)] + [samples[i:i+32] for i in range(200,6624,32)]
    assert len(blocks) == p['metric_shards'] == 209 and sum(map(len,blocks)) == 6624
    for i,b in enumerate(blocks):save(WORK/'blocks'/f'{i:03d}.json',b)
    with (WORK/'ready_queue.jsonl').open('w') as f:
        for s in samples:f.write(json.dumps(dict(id=s['id'],ready=s['ready']))+'\n')
    save(WORK/'INPUT_COMPLETE.json',dict(clips=6624,audio_reused=6624,current_reference_reused=6624,source_updated=33))
    save(WORK/'PROSODY_COMPLETE.json',dict(clips=6624,reused=True))
    save(WORK/'PREPARED.json',dict(passed=True,step=10618,total=6624,source_version=SOURCE_VERSION,
        checkpoint_sha256=p['checkpoint_sha256'],reused_audio_and_prosody=6624,reused_generated=0,
        physical_gpus=[4],gpu7_reserved=True,protocol_sha256=protocol_sha))


def generate(slot):
    import torch
    g=core.gpu()
    torch.cuda.set_per_process_memory_fraction(22*1024/g['total_mib'])
    core.generate(slot,4)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','generate','score','merge'])
    parser.add_argument('--slot',type=int,default=0)
    args=parser.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='run':
            assert read(WORK/'PREPARED.json')['passed']
            native.run()
        elif args.action=='generate':generate(args.slot)
        elif args.action=='score':native.score(args.slot)
        else:native.merge()
    except BaseException:
        save(WORK/(f'FAILURE_{args.action}_{args.slot}.json'),dict(at=time.time(),traceback=traceback.format_exc()))
        raise
