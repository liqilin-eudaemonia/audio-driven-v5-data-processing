import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
source=ROOT/'experiments/lm_listener_readiness_20261109/gt_self_check'
base=ROOT/'evaluation/lm-listener'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',DECA_PATH=str(base/'third_party/DECA'),
    PYTHONPATH=str(base)+':'+str(base/'third_party/emoca'))
results=[]
for name,script in [('baseline',base/'evaluate_listener.py'),('fast',ROOT/'compare/deployment/lm_listener_eval_fast.py')]:
    dest=OUT/('lm_scorer_'+name)
    shutil.copytree(source,dest,dirs_exist_ok=True)
    command=[sys.executable,'-c',"import runpy,sys,numpy as np; np.random.seed(137); path=sys.argv.pop(1); runpy.run_path(path,run_name='__main__')",str(script),
        '--output_dir',str(dest),'--segments_path',str(source/'segments.pth'),'--mean_std_path',str(base/'dataset/trevor')]
    with (OUT/('lm_scorer_'+name+'.log')).open('w') as log:
        p=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,(name,p.returncode)
    score=json.loads((dest/'talkshow_scores.json').read_text());score.pop('paths');results.append(score)
assert results[0]==results[1],results
(OUT/'LM_LISTENER_SCORER_PARITY.json').write_text(json.dumps(dict(passed=True,exact=True,metrics=results[1],
    scope='Official scorer on existing 60-frame GT self-check, fixed NumPy seed137; no generated model performance claim'),indent=2)+'\n')
print('Official and fast scorer outputs exactly equal')
