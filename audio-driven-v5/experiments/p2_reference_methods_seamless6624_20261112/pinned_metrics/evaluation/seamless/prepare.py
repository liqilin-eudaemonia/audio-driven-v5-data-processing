"""Freeze the approved 6,624-row Seamless test manifest; no media export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from .reader import DATA_ROOT,SCHEMA,FPS,backend,reader_contract,frame_count
from ..video.paired_metrics.media import sha256
from ..responsenet_pairs.prepare import write_json,write_jsonl

DEFAULT_SOURCE=DATA_ROOT/'experiments/listen_j_test_response_prefixfix_20261102/current/test.jsonl'


def prepare(source,output):
    if output.exists():raise FileExistsError(output)
    source=source.resolve();folder=source.parent
    summary=json.loads((folder/'summary.json').read_text())
    expected_hash=summary['manifests']['test.jsonl']['sha256']
    if sha256(source)!=expected_hash:raise ValueError('Approved source manifest hash mismatch')
    if summary['counts']['keep']!=6624:raise ValueError('Expected the user-selected 6,624-row release')
    cfg=json.loads((folder/'crop_config.json').read_text())
    data=backend();code=reader_contract()
    output.mkdir(parents=True)
    shutil.copyfile(source,output/'source.test.jsonl')
    shutil.copyfile(folder/'crop_config.json',output/'crop_config.json')
    shutil.copyfile(folder/'summary.json',output/'source.summary.json')
    rows=[];failures=[];ids=set();offset=0;hours=0;formats=Counter();archives=set()
    with (output/'source.test.jsonl').open('rb') as f:
        for i,raw in enumerate(f):
            row=json.loads(raw);sid=row['sample_id']
            try:
                if sid in ids or row['split']!='test':raise ValueError('Duplicate/non-test row')
                ids.add(sid)
                result=data.resolve_example(row,cfg)
                src=result['source'];duration=float(row['duration_s'])
                if abs(src['duration_s']-duration)>1e-7:raise ValueError('Duration mismatch')
                archive=Path(src['source']['archive'])
                if not archive.is_file():raise FileNotFoundError(archive)
                if not Path(result['speaker_audio']['path']).is_file():raise FileNotFoundError('Speaker audio source missing')
                cap=data.checked_json(row['target']['source_file'],row['target']['source_sha256'])
                recipe=cap['audio_recipe']
                if recipe['listener_id']!=src['listener_id']:raise ValueError('Listener recipe identity mismatch')
                if not Path(recipe['source']['path']).is_file():raise FileNotFoundError('Listener audio source missing')
                reference=dict(schema=SCHEMA,sample_id=sid,manifest=str((output/'source.test.jsonl').resolve()),
                    offset=offset,bytes=len(raw),row_sha256=hashlib.sha256(raw).hexdigest(),
                    crop_config=str((output/'crop_config.json').resolve()),crop_config_sha256=sha256(output/'crop_config.json'),
                    reader_contract=code)
                rows.append(dict(id=sid,dataset='seamless',split='test',identity='listener',gt_reference=reference,
                    duration_s=duration,listening_intervals=[[0.,duration]],fps=FPS,width=512,height=512,
                    frames=frame_count(duration),source_start=src['source_start_s'],source_end=src['source_end_s'],
                    speaker_id=src['speaker_id'],listener_id=src['listener_id'],
                    face_p10_px=row['face']['short_side_p10_px'],caption_format=row['target']['format'],
                    annotation_crop_differs=result['annotation_crop_differs'],crop_revision=row['inputs']['listener_first_frame']['crop_revision']))
                hours+=duration/3600;formats[row['target']['format']]+=1;archives.add(str(archive))
            except Exception as exc:failures.append(dict(id=sid,error=f'{type(exc).__name__}: {exc}'))
            offset+=len(raw)
            if (i+1)%500==0:print(f'validated {i+1}/6624',flush=True)
    if len(ids)!=6624:failures.append(dict(error=f'ID count {len(ids)} != 6624'))
    if not failures and abs(hours-summary['clip_hours']['keep'])>1e-8:failures.append(dict(error='Duration total mismatch'))
    audit=dict(dataset='seamless',split='test',source=str(source),source_sha256=expected_hash,
        clips=len(rows),hours=hours,expected_clips=6624,failures=failures,
        status='failed' if failures else 'ready_for_generation_mapping',caption_formats=dict(formats),
        source_archives=len(archives),minimum_face_p10_px=min((r['face_p10_px'] for r in rows),default=None),
        rejected_not_included=summary['counts']['reject'],review_not_included=summary['counts']['needs_review'],
        additional_face_threshold=False,new_media_exports=0,generation_available=False,
        validation='all rows resolved against provenance, caption, source pair, final crop; media existence only, no full media decoding')
    write_json(output/'audit.json',audit)
    if failures:return 2
    if sha256(output/'source.test.jsonl')!=expected_hash:raise ValueError('Frozen source copy mismatch')
    write_jsonl(output/'references.jsonl',rows)
    write_jsonl(output/'generation_map.template.jsonl',[
        dict(id=r['id'],dataset='seamless',generated=None,generated_audio=None,target_role='listener',timeline='same_pair_window',
             expected_fps=FPS,expected_width=512,expected_height=512,expected_frames=r['frames'],
             pair_duration_s=r['duration_s']) for r in rows])
    write_json(output/'seal.json',dict(dataset='seamless',source_sha256=expected_hash,
        references_sha256=sha256(output/'references.jsonl'),crop_config_sha256=sha256(output/'crop_config.json'),reader_contract=code))
    print(json.dumps(audit,ensure_ascii=False,indent=2));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raise SystemExit(prepare(a.source,a.output))


if __name__=='__main__':main()
