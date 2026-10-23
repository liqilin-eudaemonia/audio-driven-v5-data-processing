"""Resumable manifest/statistics workflow. Does not export crop videos."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'code'/name),*args],env=env,check=True)
if not (ROOT/'index.json').exists():run('index_dataset.py')
run('pipeline.py','--workers','32');run('audit_geometry.py');run('report.py');run('validate_all.py');run('package.py')
