"""Read-only comparison of the two frozen 500h datasets; no media/model jobs."""
import sqlite3
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')
OLD = ROOT / 'experiments/p2_j_train500h_cache_20261101'
NEW = ROOT / 'experiments/p2_ltx_event_current_sources_v2_20261106'
OUT = Path(__file__).resolve().parent
OLD_IDS = None


def init_worker():
    global OLD_IDS
    with (OLD / 'sample_list.csv').open() as f:
        OLD_IDS = {r['sample_id'] for r in csv.DictReader(f)}


def compare_chunk(bounds):
    counts = Counter()
    old_origins = Counter()
    examples = {}
    hours = Counter()
    db = sqlite3.connect(f'file:{NEW / "sources.sqlite"}?mode=ro', uri=True)
    for sid, raw in db.execute('SELECT sid,value FROM samples WHERE ordinal>=? AND ordinal<? ORDER BY ordinal', bounds):
        new = json.loads(raw)
        seconds = new['source']['duration_s']
        counts['current_rows'] += 1
        hours['current'] += seconds / 3600
        if sid not in OLD_IDS:
            counts['new_only'] += 1
            counts['new_only_' + new['split']] += 1
            hours['new_only'] += seconds / 3600
            continue
        old = json.loads((OLD / 'samples' / (sid + '.json')).read_bytes())
        if old['sample_id'] != sid:
            raise ValueError('Old frozen sample mismatch: ' + sid)
        counts['common'] += 1
        hours['common'] += seconds / 3600
        text_equal = old['caption'] == new['caption']
        counts['caption_text_equal' if text_equal else 'caption_text_changed'] += 1
        counts['caption_full_json_sha_equal' if old['caption_sha256'] == new['caption_sha256'] else 'caption_full_json_sha_changed'] += 1
        if not text_equal and ' '.join(old['caption'].split()) == ' '.join(new['caption'].split()):
            counts['caption_whitespace_only_change'] += 1
        for key in ['source_start_s', 'source_end_s', 'duration_s', 'listener_id', 'speaker_id', 'source', 'video_transform', 'normalized_audio']:
            counts['source_' + key + ('_equal' if old['source'].get(key) == new['source'].get(key) else '_changed')] += 1
        for key in ['audio_recipe', 'listener_audio_identity']:
            counts[key + ('_equal' if old.get(key) == new.get(key) else '_changed')] += 1
        same_video = all(old['source'].get(k) == new['source'].get(k) for k in ['source','source_start_s','source_end_s','duration_s','listener_id','video_transform'])
        counts['video_input_equal' if same_video else 'video_input_changed'] += 1
        restored = 'orientation_restoration' in new
        counts['orientation_restored' if restored else 'not_orientation_restored'] += 1
        if restored:
            original = new['historical_source']
            counts['restored_original_video_identity_equal' if all(old['source'].get(k) == original.get(k) for k in ['source','source_start_s','source_end_s','duration_s','listener_id','video_transform']) else 'restored_original_video_identity_changed'] += 1
        origin = str(old['statistics'].get('caption_file', old.get('caption_origin','unknown')))
        relative = origin.split('/experiments/')[-1]
        old_origins[relative.split('/')[0]] += 1
        category = 'text_equal' if text_equal else 'text_changed'
        if restored:
            category = 'rotation_restored'
        if category not in examples:
            examples[category] = dict(sample_id=sid,old_caption=old['caption'],new_caption=new['caption'],old_caption_file=old['statistics'].get('caption_file'),new_caption_file=new['statistics']['caption_file'],same_video_input=same_video,orientation_restored=restored)
    db.close()
    return dict(counts=counts,hours=hours,old_caption_origins=old_origins,examples=examples)


def main():
    started = time.time()
    old_rows = list(csv.DictReader((OLD / 'sample_list.csv').open()))
    new_rows = list(csv.DictReader((NEW / 'sample_list.csv').open()))
    a = {r['sample_id']:r for r in old_rows}
    b = {r['sample_id']:r for r in new_rows}
    if len(a) != len(old_rows) or len(b) != len(new_rows):
        raise ValueError('Duplicate sample IDs')
    counts, hours, origins, examples = Counter(), Counter(), Counter(), {}
    chunks = [(i,min(i+2048,len(b))) for i in range(0,len(b),2048)]
    with ProcessPoolExecutor(max_workers=12,initializer=init_worker) as pool:
        for completed, result in enumerate(as_completed([pool.submit(compare_chunk,x) for x in chunks]),1):
            result = result.result()
            counts.update(result['counts']);hours.update(result['hours']);origins.update(result['old_caption_origins'])
            for category,example in result['examples'].items():
                examples.setdefault(category,example)
            if completed % 10 == 0 or completed == len(chunks):
                print(json.dumps(dict(completed_chunks=completed,total_chunks=len(chunks),rows=counts['current_rows'],elapsed_s=round(time.time()-started,1))),flush=True)
    if counts['common'] != len(a.keys() & b.keys()) or counts['new_only'] != len(b.keys()-a.keys()) or counts['current_rows'] != len(b):
        raise ValueError('Comparison coverage mismatch')
    hours['old_only'] = sum(float(a[s]['duration_s'])/3600 for s in a.keys()-b.keys())
    hours['old'] = sum(float(x['duration_s'])/3600 for x in old_rows)
    sources = {}
    for label,path in [('old_sample_list',OLD/'sample_list.csv'),('old_manifest',OLD/'manifest.json'),('current_source_manifest',NEW/'manifest.json')]:
        sources[label] = dict(path=str(path),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
    result=dict(complete=True,read_only=True,baseline_note='Old actual cached 500h is train-only. Current 500h is train+dev; compare both without silently relabeling the old split.',old_count=len(a),current_count=len(b),old_only=len(a.keys()-b.keys()),counts=dict(counts),hours=dict(hours),common_fraction_of_current=counts['common']/len(b),common_fraction_of_old=counts['common']/len(a),old_split_counts=dict(Counter(x['split'] for x in old_rows)),current_split_counts=dict(Counter(x['split'] for x in new_rows)),old_caption_origins=dict(origins),examples=examples,sources=sources,elapsed_s=time.time()-started)
    tmp=OUT/'COMPARISON.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,OUT/'COMPARISON.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','sources')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
