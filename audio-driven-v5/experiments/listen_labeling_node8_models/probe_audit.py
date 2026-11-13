from pathlib import Path
import sys
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,digest,atomic_json
from listen_labeling.client import call,prompt
from listen_labeling.pipeline import load_manifest,branch_inputs
R=ROOT/'experiments/listen_labeling_node8_models'
c=read(R/'minicpm_r3.json');row=load_manifest(R/'smoke1_manifest.json')[0]
for branch in ['audio','visual']:
 content,meta=branch_inputs(row,branch,c['media'])
 fingerprint=digest(dict(prompt=prompt(branch,row['duration_s'],'audit'),media=row['media_hashes'],cfg=c['endpoints'][branch]))
 out=R/'coverage_probe'/branch/fingerprint
 result=call(c['endpoints'][branch],branch,row['duration_s'],content,out,fingerprint,meta,stage='audit')
 atomic_json(out/'result.json',result)
 print(branch,result['parsed'],flush=True)
