import json,subprocess,time,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
first=OUT/'audit_jobs/e77fb562224e4e23b2f38a845b2c53d5/status.json'
while not first.exists():
 response=OUT/'production/preflight_sessions/e77fb562224e4e23b2f38a845b2c53d5/result.json'
 if response.exists() and json.loads(response.read_text())['status']=='failed':raise RuntimeError(response.read_text())
 time.sleep(5)
assert json.loads(first.read_text())['returncodes']==[0,0,0]
print('Full audit passed; starting a new 24-rank launch to verify receipt reuse',flush=True)
subprocess.run([sys.executable,str(OUT/'audit_cluster.py')],check=True)
subprocess.run([sys.executable,str(OUT/'verify_all.py')],check=True)
