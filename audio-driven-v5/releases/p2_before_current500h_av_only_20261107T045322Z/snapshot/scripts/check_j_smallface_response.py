"""Bounded real-media audit of response-text reference exports, plus tamper checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_response_sft.data import load_example, resolve_example, require


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args();out=args.data
    summary=json.loads((out/'summary.json').read_text());config=json.loads((out/'crop_config.json').read_text())
    require((out/'COMPLETE.json').exists(),'Export incomplete')
    idx=sqlite3.connect(f'file:{out}/index.sqlite?mode=ro',uri=True);idx.row_factory=sqlite3.Row
    checks=[];rows=[];changed=set()
    for split,n in [('train',8),('dev',4),('test',4)]:
        count=idx.execute('SELECT COUNT(*) FROM samples WHERE split=?',(split,)).fetchone()[0]
        require(count==summary['splits'][split]['clips'],'Index count mismatch')
        choices=random.Random(f'20261101:smallface:{split}').sample(range(count),n)
        for offset in choices:
            r=idx.execute('SELECT * FROM samples WHERE split=? ORDER BY sid LIMIT 1 OFFSET ?',(split,offset)).fetchone()
            with (out/r['file']).open('rb') as f:f.seek(r['offset']);blob=f.read(r['bytes'])
            row=json.loads(blob);require(row['sample_id']==r['sid'],'Index offset mismatch');rows.append(row)
            x=load_example(row,config);inputs=x['inputs'];wave=inputs['speaker_audio'];image=inputs['listener_first_frame']
            require(set(inputs)=={'speaker_audio','speaker_sample_rate','listener_first_frame'},'Unexpected/leaking model input')
            require(image.size==(512,512) and image.mode=='RGB','First frame geometry mismatch')
            require(x['target_text']==row['target']['text'],'Target was rewritten')
            changed.add(x['audit']['annotation_crop_differs'])
            checks.append(dict(sample_id=row['sample_id'],split=split,duration_s=row['duration_s'],face_p10=row['face']['short_side_p10_px'],
                speaker_samples=len(wave),speaker_rate=inputs['speaker_sample_rate'],speaker_pcm_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                first_frame_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),audit=x['audit']))
            print('media_ok',len(checks),split,row['sample_id'],flush=True)
    first=rows[0];negative=[]
    for kind in ['target','speaker_identity','audio_time','frame_time','crop_measurement']:
        bad=copy.deepcopy(first)
        if kind=='target':bad['target']['text']+=' changed'
        elif kind=='speaker_identity':bad['inputs']['speaker_audio']['participant_id']=bad['inputs']['listener_first_frame']['participant_id']
        elif kind=='audio_time':bad['inputs']['speaker_audio']['start_s']+=1
        elif kind=='frame_time':bad['inputs']['listener_first_frame']['source_time_s']+=1
        elif kind=='crop_measurement':bad['inputs']['listener_first_frame']['measurement_sha256']='0'*64
        try:resolve_example(bad,config)
        except ValueError as e:negative.append(dict(case=kind,rejected=True,error=str(e)))
        else:raise AssertionError('Tampered example was accepted: '+kind)
    manifests={}
    full=summary.get('scope')=='full';seen=set();flagged=0;expected_500h=set()
    if full:
        expected_500h={r['sid'] for r in idx.execute("SELECT sid FROM samples WHERE split='train' AND face_p10>=224")}
        supplement=ROOT/'experiments/listen_j_final_quality_selection_20261101/train500h_audio_supplement_feasibility/supplement_trial.jsonl'
        expected_500h.update(json.loads(x)['sid'] for x in supplement.open())
        require(len(expected_500h)==285638,'Unexpected 500h reference membership')
    for split,info in summary['manifests'].items():
        h=hashlib.sha256();lines=0
        with (out/info['file']).open('rb') as f:
            for line in f:
                h.update(line);lines+=1
                if full:
                    obj=json.loads(line);sid=obj['sample_id']
                    require(obj['split']==split and sid not in seen,'Duplicate ID or wrong manifest split')
                    seen.add(sid)
                    flag=obj['provenance']['in_current_500h_trial']
                    require(flag==(sid in expected_500h),'Wrong 500h membership flag')
                    flagged+=int(flag)
        require(h.hexdigest()==info['sha256'] and lines==summary['splits'][split]['clips'],'Manifest hash/count mismatch')
        manifests[split]=dict(lines=lines,sha256=h.hexdigest())
    if full:
        require(flagged==285638 and len(seen)==1107323,'Full manifest population mismatch')
        require(all(r['sid'] in seen for r in idx.execute('SELECT sid FROM samples')),'Index and manifest IDs differ')
    report=dict(passed=True,full_manifest_unique_ids=(len(seen) if full else None),verified_500h_membership_flags=(flagged if full else None),all_manifest_hashes_and_counts=manifests,real_media_checked=len(checks),media_checks=checks,tamper_checks=negative,
        final_crop_differs_from_annotation_seen=sorted(changed),full_dataset_media_decode_verified=False,training_started=False)
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,real_media_checked=len(checks),tamper_checks=len(negative)),ensure_ascii=False))


if __name__=='__main__':main()
