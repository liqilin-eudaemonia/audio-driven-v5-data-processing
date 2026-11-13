#!/usr/bin/env python3
"""Audit identities, timing, raw captions, and eight-GPU coverage; no quality claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha,digest
from listen_audio_postprocess.recipes import read_listener_audio


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    rows=json.loads((run/'manifest.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
    assert sha(run/'manifest.json')==protocol['prepared_manifest_sha256']
    ids={r['sample_id'] for r in rows};assert len(ids)==len(rows)==50
    assert {f.stem for f in (run/'results').glob('*.json')}==ids
    results=[];frame_matches=[]
    for row in rows:
        sid=row['sample_id'];r=json.loads((run/'results'/(sid+'.json')).read_text())
        recipe=json.loads(Path(row['listener_audio_recipe']).read_text());wave,sr,meta=read_listener_audio(recipe)
        assert r['sample_id']==sid and r['recipe_sha256']==digest(recipe)==row['recipe_sha256']
        assert sha(row['target_video'])==row['media_hashes']['target_video']
        assert r['input_evidence']['audio_identity']==meta
        assert abs(len(wave)/sr-row['duration_s'])<1e-6
        assert r['input_evidence']['audio_encoder_samples']==round(row['duration_s']*16000)
        assert r['input_evidence']['fps']==8 and r['input_evidence']['second_per_grid_ts']==[.25]
        assert r['input_evidence']['audio_origin']=='virtual_postprocessed_listener_recipe'
        assert r['finish_reason']=='stop' and re.fullmatch(r'[A-Za-z][^\n\r]*',r['caption']) and len(r['caption'].split())>=10
        assert r['prompt']==protocol['prompt'].format(duration=row['duration_s'])
        prior=json.loads((ROOT/'data_previews/listen_benchmark_review/native_simple/raw'/sid/'native_baseline.json').read_text())
        before=prior['input_evidence']['video']['frames'];after=r['input_evidence']['video_evidence']['frames']
        frame_matches.append(dict(sample_id=sid,frame_pts_equal=[x['time_s'] for x in before]==[x['time_s'] for x in after],
                                  frame_jpeg_hashes_equal=[x['jpeg_sha256'] for x in before]==[x['jpeg_sha256'] for x in after]))
        assert frame_matches[-1]['frame_pts_equal']
        results.append(r)
    workers=[json.loads(f.read_text()) for f in (run/'workers').glob('*.json')]
    assert len(workers)==8 and {w['gpu'] for w in workers}==set(range(8))
    assert all(w['stage']=='complete' and w['completed']==w['total'] for w in workers)
    gpu_counts=Counter(r['gpu'] for r in results);assert set(gpu_counts)==set(range(8))
    assert sum(gpu_counts.values())==50
    report=dict(passed=True,samples=50,nonempty_natural_language_captions=50,normal_finish=50,
                per_gpu=dict(sorted(gpu_counts.items())),input_identity_checks=True,preserved_lengths=True,
                matched_historical_frame_pts=sum(x['frame_pts_equal'] for x in frame_matches),
                matched_historical_frame_jpeg_hashes=sum(x['frame_jpeg_hashes_equal'] for x in frame_matches),
                full_silence=sum(row['audio_preflight']['full_silence'] for row in rows),
                inference_seconds=sum(r['seconds'] for r in results),
                worker_wall_seconds_including_preparation=[w['seconds'] for w in sorted(workers,key=lambda w:w['gpu'])],
                word_count_range=[min(len(r['caption'].split()) for r in results),max(len(r['caption'].split()) for r in results)],
                semantic_quality_verified=False,human_ground_truth=False,training_admitted=False)
    atomic_json(run/'audit.json',report);atomic_json(run/'frame_audit.json',frame_matches)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
