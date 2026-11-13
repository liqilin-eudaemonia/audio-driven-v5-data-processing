from pathlib import Path
import json,subprocess,sys,time
p=Path(__file__).resolve().parent;r=p.parents[1];py='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
start=time.time()
while not (p/'validation_queue_completed.json').exists():
 if time.time()-start>1200:raise TimeoutError('Validation queue did not finish')
 time.sleep(2)
base=json.loads((p/'s0_1gpu_E1/spec.json').read_text())
for arm,gpu in [('E0','4'),('E1','5')]:
 ck=p/f'optimized_1gpu/runs/s0_20261026/{arm}';d=p/f'optimized_replay_{arm}';d.mkdir(exist_ok=False)
 s={**base,'argv':[py,str(r/'scripts/check_p2_listen_gpu_replay.py'),'--checkpoint',str(ck/'step_000006.pt'),'--expected',str(ck/'step_000007.pt'),'--output',str(p/f'optimized_replay_{arm}.json')],'env':{**base['env'],'CUDA_VISIBLE_DEVICES':gpu}}
 (d/'spec.json').write_text(json.dumps(s,indent=2));q=subprocess.Popen([sys.executable,str(p/'run_job.py'),str(d/'spec.json')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);print(arm,q.pid,flush=True)
with (p/'optimized_pair_comparison.log').open('w') as f:
 subprocess.run([py,str(r/'scripts/check_p2_listen_gpu_pair.py'),'--single',str(p/'optimized_1gpu'),'--distributed',str(p/'optimized_2gpu'),'--seed','20261026','--steps','7','--output',str(p/'optimized_pair_comparison.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
