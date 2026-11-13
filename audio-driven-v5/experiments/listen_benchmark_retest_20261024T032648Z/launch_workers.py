from pathlib import Path
import json,sys,os,subprocess
root=Path(sys.argv[1]);node=sys.argv[2];plan=json.loads((root/'full50_plan.json').read_text());cfg=json.loads(Path(plan['config']).read_text());systems={s['name']:s for s in cfg['systems']};phases=json.loads((root/'selected_phases.json').read_text());folders=json.loads((root/'phases.json').read_text())
joblist=plan['nodes'][node]['jobs'] if len(sys.argv)<4 else json.loads(sys.argv[3]);procs=[]
for job in joblist:
 names=job['systems'];s=systems[names[0]];gpu=job['gpu'];engine=s.get('runtime_engine','').startswith('vllm');worker=f'{node}-g{gpu}-'+job.get('suffix','initial')
 output=root/folders[phases[names[0]]];script='run_listen_benchmark_vllm.py' if engine else 'run_listen_benchmark_queue.py'
 cmd=[s['python'],str(Path(plan['code'])/'scripts'/script),'--config',plan['config'],'--manifest',plan['manifest'],'--output',str(output),'--systems',','.join(names),'--worker-id',worker]
 if engine:cmd+=['--batch-size','1' if names==['qwen3_omni_native'] else '8','--clients','1' if names==['qwen3_omni_native'] else '16']
 elif gpu=='cpu':cmd+=['--device','cpu']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu=='cpu' else str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4')
 if engine:env['LD_LIBRARY_PATH']='/private/software/conda/envs/vllm/lib:'+env.get('LD_LIBRARY_PATH','')
 logfile=root/(worker+'.log')
 with logfile.open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 procs.append(dict(pid=p.pid,worker_id=worker,node=node,gpu=gpu,systems=names,command=cmd,output=str(output),log=str(logfile)))
 dest=root/('launch-'+node+'-'+str(procs[0]['pid'])+'.json');dest.write_text(json.dumps(procs,indent=2));print(json.dumps(procs))
