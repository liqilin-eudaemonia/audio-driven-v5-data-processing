"""Complete validation and launch only after all CPU/GPU/audit gates succeed."""
import os,subprocess,time
from control import *
start=time.monotonic()
while True:
 suite=read(OUT/'gpu_suite_status.json')
 if suite['state']=='failed':raise RuntimeError(suite)
 audits=list((OUT/'audit_jobs').glob('*/status.json'))
 if suite['state']=='complete' and audits:
  if read(audits[-1])['returncodes']!=[0,0,0]:raise RuntimeError('Audit cluster failed')
  break
 if 'Traceback' in (OUT/'audit_cluster.log').read_text():raise RuntimeError('Audit cluster failed; inspect log')
 if time.monotonic()-start>3600:raise TimeoutError('Validation did not finish in one hour')
 time.sleep(5)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
for name,log in [('compare_gpu.py','gpu_comparison.log'),('finalize_validation.py','validation.log')]:
 with (OUT/log).open('w') as stream:
  subprocess.run([str(PYTHON),str(OUT/name)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run([str(PYTHON),str(OUT/'controller.py'),'--detach'],cwd=ROOT,check=True)
print('VERIFIED_AND_LAUNCHED',now(),flush=True)
