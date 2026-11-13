import subprocess,sys,time
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
last=EXP/'localbc/improvised_V00_S0862_I00000794/result.json'
deadline=time.monotonic()+1800
while not last.exists():
 if time.monotonic()>deadline:raise RuntimeError('QC did not complete within 30 minutes; inspect run.log')
 time.sleep(5)
for script in ['compare.py','export_pairs.py','publish.py']:
 subprocess.run([sys.executable,str(EXP/script)],cwd=ROOT,check=True)
print('READY FOR BROWSER',flush=True)
