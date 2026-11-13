"""Four real reference clips, explicitly GT self-comparison, never model scores."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import soundfile as sf
from ..seamless.reader import load_listener_audio
from ..responsenet_pairs.bind import read_rows
from ..responsenet_pairs.prepare import write_json,write_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parents[1];rows=[]
    refs={r['id']:r for r in read_rows(root/'responsenet_pairs/prepared_20261102/references.jsonl')}
    for r in read_rows(root/'responsenet_pairs/self_check_padded_20261102/self_check_pairs.jsonl'):
        ref=refs[r['id']];rows.append(dict(r,dataset='responsenet',gt_audio=ref['listener']['audio'],gt_asr=ref['listener']['asr'],
            generated_audio=ref['listener']['audio'],pair_duration_s=ref['duration_s']))
    for r in read_rows(root/'seamless/self_check_20261102/self_check_pairs.jsonl'):
        wave,sr,_=load_listener_audio(r['gt_reference']);path=out/(r['id']+'.self_check_only.wav');sf.write(path,wave,sr,subtype='FLOAT')
        rows.append(dict(r,generated_audio=str(path),pair_duration_s=len(wave)/sr))
    write_jsonl(out/'pairs.jsonl',rows);write_json(out/'purpose.json',dict(self_check_only=True,generated_model_results=False,datasets=['responsenet','seamless'],pairs=len(rows)))
    code=subprocess.run([sys.executable,'-m','evaluation.listener_metrics.run','--manifest',str(out/'pairs.jsonl'),
        '--output',str(out/'metrics'),'--batch-size','8','--keep-media']).returncode
    report=json.loads((out/'metrics/summary.json').read_text())
    assert report['audio']['valid_pairs']==len(rows)
    for key in ['acoustic','speech']:
        counts=report['audio']['activity'][key]['strict'];assert counts['fp']==counts['fn']==0
    write_json(out/'assertions.json',dict(passed=True,audio_all_read=True,self_check_only=True,worker_exit_code=code,
        reaction_valid_pairs=report['reaction']['valid_pairs'],reaction_detection_failures_preserved=True))
    raise SystemExit(code)


if __name__=='__main__':main()
