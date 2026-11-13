import sys,json,os
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(R/'src'))
from p2_listen.contracts import read,atomic_json
from p2_listen.config import extractor_config
from p2_autonomous_eval.planning import generate_attempt,DEFAULTS
E=Path(__file__).resolve().parent;c=read(R/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json');cfg=extractor_config(c);from omni_av_p2.data import load_omni;model,processor=load_omni(cfg,'cuda:0');obs=sorted(read(R/'experiments/p2_reintegrated_final_eval_20261028/test24/observations.json'),key=lambda o:o['requested_duration_s']);results=[]
for o in [obs[0],obs[12],obs[-1]]:
 result,audit=generate_attempt(model,processor,o,DEFAULTS,1,"Zero-duration events are invalid: end must be strictly greater than start. Never output start=0.0 and end=0.0. Use positive duration, e.g. start=0.2 and end=0.7.")
 results.append(dict(sample_id=o['sample_id'],result=result,audit=audit));atomic_json(E/'planner_probe_retry_v2.json',results);print(json.dumps(results[-1],ensure_ascii=False),flush=True)
