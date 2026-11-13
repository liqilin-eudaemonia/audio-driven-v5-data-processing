"""Freeze the CURRENT train+dev selection for labels, without recaching tensors."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import csv,hashlib,json,multiprocessing,sqlite3

from p2_listen.contracts import read,sha,digest,atomic_json,ROOT
from p2_clip_events.labels import derive
from p2_clip_events import VERSION as CLIP_VERSION
from . import VERSION


def checked_index(path):
    index=read(path)
    if index['status']!='complete' or not index['source_split_counts'].get('dev'):
        raise ValueError('Require the complete current train+dev cache index')
    for name,cache in index['caches'].items():
        if (sha(cache['complete'])!=cache['complete_sha256'] or
                sha(Path(cache['work'])/'protocol.json')!=cache['protocol_sha256'] or cache['samples']!=index['samples']):
            raise ValueError('Changed current cache completion/protocol: '+name)
    if sha(index['joint_readback'])!=index['joint_readback_sha256'] or not read(index['joint_readback'])['passed']:
        raise ValueError('Current cache joint readback changed/failed')
    return index


def _freeze_row(raw):
    row=json.loads(raw);sid=row['sample_id']
    if row['split'] not in ('train','dev') or row['source_split']!=row['split']:
        raise ValueError('Current training view must preserve source train/dev and exclude test')
    cap_raw=Path(row['caption_file']).read_bytes()
    if hashlib.sha256(cap_raw).hexdigest()!=row['caption_file_sha256']:raise ValueError('Current caption changed: '+sid)
    cap=json.loads(cap_raw)
    for key in ('sample_id','caption','source','audio_recipe','crop_provenance','finish_reason','review_flags'):
        if cap[key]!=row[key]:raise ValueError('Current caption/selection mismatch: '+sid+'/'+key)
    with Path(row['statistics_file']).open('rb') as stream:
        stream.seek(row['statistics_offset']);stat_raw=stream.readline();stat=json.loads(stat_raw)
    if (stat['sample_id']!=sid or stat['caption']!=row['caption'] or stat['source']!=row['source']
            or stat['caption_file_sha256']!=row['caption_file_sha256'] or stat['counted_types']!=row['counted_types']):
        raise ValueError('Current statistics/selected caption or media mismatch: '+sid)
    item=dict(schema='ltx-current-label-source-v1',sample_id=sid,split=row['split'],source=row['source'],
        caption=row['caption'],caption_sha256=row['caption_file_sha256'],audio_recipe=row['audio_recipe'],
        listener_audio_identity=cap['input_evidence']['audio_identity'],selected_row_sha256=digest(row),
        statistics=dict(caption_file=row['caption_file'],caption_file_sha256=row['caption_file_sha256'],
            file=row['statistics_file'],offset=row['statistics_offset'],line_sha256=hashlib.sha256(stat_raw).hexdigest()),
        selection_group=row['selection_group'],human_ground_truth=False)
    if 'orientation_restoration' in row:
        o=row['orientation_restoration'];raw_prov=Path(o['provenance_file']).read_bytes()
        if hashlib.sha256(raw_prov).hexdigest()!=o['provenance_sha256']:
            raise ValueError('Changed upright timeline provenance')
        prov=json.loads(raw_prov);original=cap['original_effective_source'];current=row['source']
        if (prov['original_source']!=original or original!=o['original_effective_source']
                or not prov['timestamps_preserved'] or not prov['external_audio_recipe_unchanged']
                or prov['spatial_resize'] or prov['rotation_ccw_degrees'] not in (-90,90)
                or prov['rotation_ccw_degrees']!=o['rotation_ccw_degrees']
                or prov['derived_archive_sha256']!=o['derived_archive_sha256']
                or o['derived_archive_sha256']!=current['orientation_restoration']['archive_sha256']):
            raise ValueError('Unproven original -> upright media/time lineage')
        for key in ('sample_id','pair','split','listener_id','speaker_id','source_start_s','source_end_s','duration_s','normalized_audio'):
            if original[key]!=current[key]:raise ValueError('Upright restoration changed '+key)
        item['historical_source']=original
        item['orientation_restoration']=row['orientation_restoration']
    clip=derive(dict(item,statistics=stat));clip['source_digest']=digest(item)
    return item,clip


def freeze_current(cache_index,output,processes=16):
    cache_index=Path(cache_index).resolve();index=checked_index(cache_index)
    q=read(Path(index['caches']['qwen_hidden']['work'])/'protocol.json');selection=Path(q['selection'])
    if sha(selection)!=index['selection_sha256'] or q['sample_ids']!=read(Path(index['caches']['av_vae']['work'])/'protocol.json')['sample_ids']:
        raise ValueError('Current selection/H/AV membership mismatch')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);clips=output/'clip_labels';clips.mkdir()
    dbpath=output/'sources.sqlite';clip_db=clips/'labels.sqlite'
    if dbpath.exists() or clip_db.exists():raise FileExistsError('Use a new current label-source output')
    ids=[];entries={};counts=Counter();pos=Counter();weak_pos=Counter();weak_unknown=Counter();weak_rows=0;conflicts=0;hours=0.
    with sqlite3.connect(dbpath) as db,sqlite3.connect(clip_db) as cd,selection.open('rb') as stream:
        db.execute('CREATE TABLE samples(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE NOT NULL,value TEXT NOT NULL)')
        cd.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE NOT NULL,value TEXT NOT NULL)')
        with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork')) as pool:
            while batch:=list(islice(stream,2048)):
                for item,clip in pool.map(_freeze_row,batch,chunksize=8):
                    sid=item['sample_id'];ordinal=len(ids);ids.append(sid);entries[sid]=digest(item)
                    db.execute('INSERT INTO samples VALUES (?,?,?)',(ordinal,sid,json.dumps(item,separators=(',',':'))))
                    cd.execute('INSERT INTO labels VALUES (?,?,?)',(ordinal,sid,json.dumps(clip,separators=(',',':'))))
                    counts[item['split']]+=1;hours+=item['source']['duration_s']/3600
                    pos.update(clip['positive']);conflicts+=len(clip['conflicts'])
                    if clip['allow_weak']:weak_rows+=1;weak_pos.update(clip['positive']);weak_unknown.update(clip['unknown'])
                db.commit();cd.commit()
                print(json.dumps(dict(stage='freeze_current_label_sources',samples=len(ids),total=index['samples'])),flush=True)
    if ids!=q['sample_ids'] or dict(counts)!=index['source_split_counts'] or len(ids)!=len(set(ids)):
        raise ValueError('Current source coverage/order/split mismatch')
    with (output/'sample_list.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','split','duration_s','selection_group']);writer.writeheader()
        db=sqlite3.connect(dbpath)
        for (value,) in db.execute('SELECT value FROM samples ORDER BY ordinal'):
            item=json.loads(value);writer.writerow(dict(sample_id=item['sample_id'],split=item['split'],
                duration_s=item['source']['duration_s'],selection_group=item['selection_group']))
        db.close()
    manifest=dict(schema=VERSION,kind='current-label-sources',training_ids=ids,source_split_counts=counts,
        samples=len(ids),hours=hours,entries=entries,database_sha256=sha(dbpath),current_cache_index=str(cache_index),
        current_cache_index_sha256=sha(cache_index),selection_sha256=index['selection_sha256'],
        source_split_preserved=True,test_excluded=True,code_sha256=sha(__file__))
    atomic_json(output/'manifest.json',manifest,immutable=True)
    stats={}
    for name in sorted(pos):
        neg=weak_rows-weak_pos[name]-weak_unknown[name];enabled=pos[name]>0 and neg>0
        stats[name]=dict(positive=pos[name],weak_negative=neg,known_weight=pos[name]+.1*neg,enabled=enabled,
            pos_weight=max(1.,min(10.,.1*neg/pos[name])),disabled_reason=None if enabled else 'no_positive_negative_contrast')
    atomic_json(clips/'manifest.json',dict(schema=CLIP_VERSION,cache_work=str(output),cache_manifest_sha256=sha(output/'manifest.json'),
        database_sha256=sha(clip_db),samples=len(ids),sample_ids_sha256=digest(ids),full_train=True,
        classes=dict(video=[k for k in sorted(pos) if k.startswith('visual:')],audio=[k for k in sorted(pos) if k.startswith('audio:')]),
        statistics=stats,absence_weight=.1,label_policy='counted_positive; uncertain_mask; omitted_weak_negative',
        temporal_targets=False,class_mapping='existing_open_names_without_new_merges',conflicts_masked=conflicts,
        human_ground_truth=False,source_split_counts=counts,source_split_preserved=True),immutable=True)
    return dict(samples=len(ids),hours=hours,source_split_counts=counts,source=str(output),clip_labels=str(clips),test_excluded=True)
