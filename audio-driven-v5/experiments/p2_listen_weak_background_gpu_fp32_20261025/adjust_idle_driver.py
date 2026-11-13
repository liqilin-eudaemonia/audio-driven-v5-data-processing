from pathlib import Path
import json,os,signal,shutil
p=Path(__file__).resolve().parent
s=json.loads((p/'job_status.json').read_text());assert s['jobs']=={},'Driver has started tasks; do not stop it'
needle='experiments/p2_listen_weak_background_gpu_fp32_20261025/run_gpu_checks.py'
found=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:args=(d/'cmdline').read_bytes().decode().split('\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if needle in args or str(p/'run_gpu_checks.py') in args:found.append(int(d.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
shutil.copy2(p/'job_status.json',p/'initial_idle_status.json')
f=p/'run_gpu_checks.py';s=f.read_text();assert '65*1024' in s;f.write_text(s.replace('65*1024','60*1024'))
print('Stopped only idle validation driver',found,'new minimum free GPU memory: 60 GiB')
