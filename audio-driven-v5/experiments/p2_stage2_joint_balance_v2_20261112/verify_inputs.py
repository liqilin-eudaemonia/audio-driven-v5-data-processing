from pathlib import Path
import sys,time,torch
from unittest.mock import patch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_stage2_joint_v2.config import validate,source_identity
from p2_stage2_joint_v2.data import Dataset
from p2_listen.contracts import read,atomic_json,sha
c=validate(read(W/'candidate_full6000/config.json'));torch.set_num_threads(2);d=Dataset(c);records=[]
with patch('p2_j_current_qwen_hidden.Reader.load',side_effect=AssertionError('Frozen Qwen payload must never be loaded')):
 for r in read(c['calibration_manifest'])['samples']:
  x=d.sample(r['index'])
  assert x['sample_id']==r['sample_id'] and x['condition']is None and x['planning_valid'] and x['planning_tokens']>0
  assert x['text']['positive']['prompt']==x['generation_prompt']
  records.append(dict(sample_id=x['sample_id'],planning_tokens=x['planning_tokens'],voiced=r['voiced'],duration_s=r['duration_s'],frozen_hidden_loaded=False))
assert len(records)==16 and not torch.cuda.is_initialized()
atomic_json(W/'CALIBRATION_INPUTS_ACCEPTED.json',dict(success=True,at=time.time(),source=source_identity(),records=records,manifest_sha256=sha(c['calibration_manifest']),gpu_used=False))
print('16 real calibration inputs passed; frozen H load forbidden',flush=True)
