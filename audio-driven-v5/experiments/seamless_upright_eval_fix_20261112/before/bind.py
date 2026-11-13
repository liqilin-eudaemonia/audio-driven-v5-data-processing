"""Bind real same-window listener outputs to the frozen Seamless test set."""
import argparse
import json
from pathlib import Path
import cv2
from .reader import SCHEMA
from ..video.paired_metrics.media import sha256
from ..responsenet_pairs.prepare import probe_video,write_json,write_jsonl
from ..responsenet_pairs.bind import read_rows


def load_references(prepared):
    seal=json.loads((prepared/'seal.json').read_text())
    if seal['dataset']!='seamless':raise ValueError('Wrong prepared dataset')
    for filename,key in [('references.jsonl','references_sha256'),('source.test.jsonl','source_sha256'),('crop_config.json','crop_config_sha256')]:
        if sha256(prepared/filename)!=seal[key]:raise ValueError(f'Prepared {filename} changed')
    rows=read_rows(prepared/'references.jsonl')
    if seal.get('schema')=='seamless-approved-subset-v1':
        expected=seal['expected_clips']
        if type(expected) is not int or not 0<expected<6624:raise ValueError('Invalid approved subset size')
        parent=Path(seal['parent_prepared'])
        if parent.resolve()==prepared.resolve():raise ValueError('Subset parent cannot be itself')
        if sha256(parent/'seal.json')!=seal['parent_seal_sha256']:raise ValueError('Subset parent seal changed')
        if sha256(parent/'references.jsonl')!=seal['parent_references_sha256']:raise ValueError('Subset parent references changed')
        original={r['id']:r for r in load_references(parent)}
        ids=sorted(r['id'] for r in rows)
        import hashlib
        if hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=seal['selected_ids_sha256']:
            raise ValueError('Subset membership changed')
        media_keys=('id','dataset','split','identity','duration_s','fps','width','height','frames',
                    'source_start','source_end','speaker_id','listener_id','face_p10_px','listening_intervals')
        for row in rows:
            previous=original.get(row['id'])
            if previous is None or any(row.get(k)!=previous.get(k) for k in media_keys):
                raise ValueError('Subset changed parent media identity or geometry')
            ref=row['gt_reference'];old=previous['gt_reference']
            if (ref['row_sha256']!=old['row_sha256'] or ref['crop_config_sha256']!=old['crop_config_sha256']
                    or ref['reader_contract']!=old['reader_contract']):
                raise ValueError('Subset changed parent GT provenance')
    else:expected=6624
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError('Wrong prepared row count')
    return rows


def match(references,mapping,root):
    known={r['id']:r for r in references};by_id={}
    for item in mapping:
        if item['id'] not in known or item['id'] in by_id:raise ValueError('Unknown/duplicate generated sample ID')
        by_id[item['id']]=item
    result=[];missing=[]
    for ref in references:
        item=by_id.get(ref['id'])
        if not item or not item.get('generated'):
            missing.append(ref['id']);continue
        if item.get('dataset')!='seamless' or item.get('target_role')!='listener' or item.get('timeline')!='same_pair_window':
            raise ValueError('Require dataset=seamless, target_role=listener, timeline=same_pair_window')
        path=(root/Path(item['generated'])).resolve()
        if not path.is_file():missing.append(ref['id']);continue
        result.append((ref,path))
    if missing:raise ValueError(f'{len(missing)} generated videos missing; no silent data reduction. Examples: {missing[:4]}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--generated-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1)
    references=load_references(a.prepared)
    bindings=match(references,read_rows(a.generated_map),a.generated_map.resolve().parent)
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];failures=[]
    for ref,path in bindings:
        try:
            video=probe_video(path)
            if abs(video['fps']-ref['fps'])>1e-3:raise ValueError('Expected 25fps')
            if (video['width'],video['height'])!=(512,512):raise ValueError('Expected final-crop 512x512 listener video')
            if abs(video['frames']-ref['frames'])>1:raise ValueError('Generated duration differs by more than one frame')
            rows.append(dict(id=ref['id'],dataset='seamless',identity='listener',gt_reference=ref['gt_reference'],
                             generated=str(path),listening_intervals=ref['listening_intervals'],
                             source_start=ref['source_start'],source_end=ref['source_end'],timeline='same_pair_window'))
        except Exception as exc:failures.append(dict(id=ref['id'],error=f'{type(exc).__name__}: {exc}'))
    audit=dict(dataset='seamless',expected_pairs=len(references),valid_pairs=len(rows),failures=failures,
               status='failed' if failures else 'ready_for_scoring',mapping_sha256=sha256(a.generated_map),
               references_sha256=sha256(a.prepared/'references.jsonl'))
    write_json(a.output/'binding_audit.json',audit)
    if failures:raise SystemExit(2)
    write_jsonl(a.output/'pairs.jsonl',rows)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
