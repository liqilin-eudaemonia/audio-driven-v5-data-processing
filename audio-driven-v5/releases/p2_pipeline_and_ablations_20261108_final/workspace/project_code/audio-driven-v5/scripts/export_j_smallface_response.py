"""Export retained response-text data: small-face complement or the full quality-passed pool."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_response_sft.data import SCHEMA, require
from listen_visual_quality.io import canonical_hash
from listen_benchmark.j_omni_crop import REVISION


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection-db',type=Path,default=ROOT/'experiments/listen_j_final_quality_selection_20261101/selection.sqlite')
    ap.add_argument('--supplement',type=Path,default=ROOT/'experiments/listen_j_final_quality_selection_20261101/train500h_audio_supplement_feasibility/supplement_trial.jsonl')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scope',choices=['smallface','full'],default='smallface')
    args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/'COMPLETE.json').exists(),'Output already complete; choose another output')
    cfg=json.loads((ROOT/'experiments/listen_j_omni_20261030/config.json').read_text())
    dump(out/'crop_config.json',cfg);cfg_sha=canonical_hash(cfg)
    supplement=[json.loads(x) for x in args.supplement.open()]
    require(len(supplement)==len({r['sid'] for r in supplement})==21584,'Unexpected supplement snapshot')
    db=sqlite3.connect(f'file:{args.selection_db}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TEMP TABLE excluded(sid TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO excluded VALUES(?)',[(r['sid'],) for r in supplement])
    require(db.execute("SELECT COUNT(*) FROM samples s JOIN excluded e ON s.sid=e.sid WHERE s.split='train' AND s.face_p10<224").fetchone()[0]==21584,'Supplement not in source pool')
    if args.scope=='full':
        db.execute('CREATE TEMP TABLE chosen AS SELECT * FROM samples')
    else:
        db.execute("CREATE TEMP TABLE chosen AS SELECT s.* FROM samples s LEFT JOIN excluded e ON s.sid=e.sid WHERE s.face_p10<224 AND e.sid IS NULL")
    selected_500h={r['sid'] for r in supplement}
    selected_500h.update(r['sid'] for r in db.execute("SELECT sid FROM samples WHERE split='train' AND face_p10>=224"))
    require(len(selected_500h)==285638,'500h membership changed')
    expected={r['split']:dict(clips=r['n'],hours=r['h']) for r in db.execute('SELECT split,COUNT(*) n,SUM(duration)/3600 h FROM chosen GROUP BY split')}
    expected_counts=dict(train=1086217,dev=8822,test=12284) if args.scope=='full' else dict(train=800579,dev=5806,test=7341)
    require({k:v['clips'] for k,v in expected.items()}==expected_counts,'Unexpected candidate population')
    require(db.execute('SELECT COUNT(*) FROM (SELECT group_id FROM chosen GROUP BY group_id HAVING COUNT(DISTINCT split)>1)').fetchone()[0]==0,'Cross-split connected group')
    index=sqlite3.connect(out/'index.sqlite');index.execute('DROP TABLE IF EXISTS samples')
    index.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
    files={s:(out/f'{s}.jsonl.tmp').open('wb') for s in expected}
    digests={s:hashlib.sha256() for s in expected};counts=collections.Counter();hours=collections.Counter();flags=collections.Counter()
    sizes=collections.defaultdict(lambda:collections.Counter());source_handle=None;prev=None;pending=[];t=time.time()
    for i,row in enumerate(db.execute('SELECT * FROM chosen ORDER BY file,offset'),1):
        if row['file']!=prev:
            if source_handle:source_handle.close()
            prev=row['file'];source_handle=open(prev,'rb')
        source_handle.seek(row['offset']);raw=source_handle.readline();r=json.loads(raw);src=r['source'];q=r['quality']
        sid=row['sid'];split=row['split'];duration=row['duration']
        require(r['sample_id']==sid==src['sample_id'] and r['split']==split==src['split'],'Source identity mismatch')
        require(abs(r['duration_s']-duration)<1e-8 and abs(src['source_end_s']-src['source_start_s']-duration)<=1/48000+1e-9,f'Source duration mismatch: {sid}')
        require(q['decision']=='retain' and q['sample_id']==sid and q['contract_sha256']==cfg['visual_quality_gate']['contract_sha256'],'Wrong quality decision')
        crop_prov=r.get('crop_provenance')
        if crop_prov:
            require(crop_prov['revision']==REVISION and crop_prov['original_source_sha256']==q['source_sha256'] and crop_prov['effective_source_sha256']==canonical_hash(src),'Corrected quality/source mismatch')
            require(crop_prov['measurement_sha256']==q['measurement_sha256'] and crop_prov['measurement_file']==q['measurement_file'],'Corrected measurement mismatch')
        else:
            require(q['source_sha256']==canonical_hash(src),'Legacy quality/source mismatch')
        require(src['speaker_id']!=src['listener_id'] and bool(r['caption'].strip()),'Missing target or wrong participants')
        require(r['face_short_side_p10']==row['face_p10'],'Face measurement mismatch')
        require(bool(r.get('caption_file_sha256')),'Missing caption digest')
        item=dict(schema=SCHEMA,sample_id=sid,split=split,pair=row['pair'],group_id=row['group_id'],duration_s=duration,
            face=dict(short_side_p10_px=row['face_p10'],short_side_p50_px=row['face_p50']),
            selection_reason=('all_final_quality_retained' if args.scope=='full' else 'below_224px_primary_threshold_and_outside_500h_trial'),
            inputs=dict(speaker_audio=dict(participant_id=src['speaker_id'],source_result_file=src['source_result'],
                source_result_sha256=src['source_result_sha256'],clip_index=src['source_clip_index'],start_s=src['source_start_s'],end_s=src['source_end_s'],role='speaker',processing='paired_normalized_track_no_listener_gate'),
                listener_first_frame=dict(participant_id=src['listener_id'],source=src['source'],source_time_s=src['source_start_s'],output_size=[512,512],crop_revision=REVISION,
                    measurement_file=q['measurement_file'],measurement_sha256=q['measurement_sha256'],frame_policy='nearest_source_pts_to_pair_start')),
            target=dict(text=r['caption'],source_file=r['caption_file'],source_sha256=r['caption_file_sha256'],format='original_caption_verbatim'),
            provenance=dict(statistics_file=row['file'],statistics_byte_offset=row['offset'],statistics_line_sha256=hashlib.sha256(raw).hexdigest(),
                crop_config_sha256=cfg_sha,original_training_admitted=bool(r.get('training_admitted',False)),human_ground_truth=False,
                processed_listener_waveform_nonzero=not r['audio']['waveform_full_silence'],admission_overridden=False))
        if args.scope=='full':item['provenance']['in_current_500h_trial']=sid in selected_500h
        blob=(json.dumps(item,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
        f=files[split];offset=f.tell();f.write(blob);digests[split].update(blob)
        pending.append((sid,split,row['pair'],src['listener_id'],row['group_id'],src['source_start_s'],src['source_end_s'],duration,row['face_p10'],f'{split}.jsonl',offset,len(blob)))
        counts[split]+=1;hours[split]+=duration/3600
        flags[split+':listener_nonzero']+=not r['audio']['waveform_full_silence']
        lo=next((x for x in [512,384,320,256,224,216,208,200,192,184,176,160,144,128,112,96,64,0] if row['face_p10']>=x),0)
        sizes[split][str(lo)]+=duration/3600
        if len(pending)>=2000:index.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);index.commit();pending=[]
        if i%50000==0:print(json.dumps(dict(done=i,total=sum(x['clips'] for x in expected.values()),elapsed_s=round(time.time()-t,1))),flush=True)
    if source_handle:source_handle.close()
    if pending:index.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',pending);index.commit()
    for s,f in files.items():
        f.close();require(counts[s]==expected[s]['clips'] and abs(hours[s]-expected[s]['hours'])<1e-6,'Export count/time mismatch')
        (out/f'{s}.jsonl.tmp').replace(out/f'{s}.jsonl')
    index.execute('CREATE INDEX split_index ON samples(split,sid)');index.commit()
    overlaps=index.execute('''SELECT COUNT(*) FROM (SELECT start,MAX(end) OVER (PARTITION BY pair,listener ORDER BY start,end ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM samples) WHERE start<previous_end-0.000001''').fetchone()[0]
    require(overlaps==0,'Overlapping source intervals')
    summary=dict(schema=SCHEMA,status='reference_export_complete_media_resolved_on_demand',splits=expected,total_clips=sum(counts.values()),total_hours=sum(hours.values()),
        scope=args.scope,source_selection_db=str(args.selection_db.resolve()),reference_500h_supplement_sha256=hashlib.sha256(args.supplement.read_bytes()).hexdigest(),
        selected_500h_train_overlap=(285638 if args.scope=='full' else 0),source_time_overlaps=overlaps,group_split_overlap=0,
        face_upper_bound_exclusive_px=(None if args.scope=='full' else 224),minimum_face_threshold_added=False,
        face_bin_lower_bound_hours={k:dict(v) for k,v in sizes.items()},audio_metadata=dict(flags),
        manifests={s:dict(file=f'{s}.jsonl',sha256=h.hexdigest(),bytes=(out/f'{s}.jsonl').stat().st_size) for s,h in digests.items()},
        crop_config_sha256=cfg_sha,physical_wav_exports=0,physical_image_exports=0,
        input_contract=['paired_speaker_audio_segment','single_final_crop_listener_first_frame'],target='existing_listener_caption_verbatim',
        waveform_or_frames_checked_for_every_row=False,prior_training_admission_preserved=True,training_started=False,elapsed_s=time.time()-t)
    dump(out/'summary.json',summary);dump(out/'COMPLETE.json',dict(status='complete',scope='reference_manifests_not_all_media_decode',summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()))
    index.close();db.close();print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
