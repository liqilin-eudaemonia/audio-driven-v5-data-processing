from pathlib import Path
import json,subprocess,sys,time
p=Path(__file__).resolve().parent;r=p.parents[1];py='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
base=dict(cwd=str(r),timeout=1800,env=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='expandable_segments:True'))
def launch(name,folder,arm,gpus,world=1,resume=False,stop=False):
 d=p/name;d.mkdir(exist_ok=False)
 argv=[py]+(['-m','torch.distributed.run','--standalone','--nproc-per-node=2'] if world==2 else [])
 argv += [str(r/'scripts/p2_listen.py'),'train','--config',str(p/folder/'config.json'),'--arm',arm,'--train-seed','20261026','--s0','--steps','5']
 if stop:argv+=['--stop-after','3']
 if resume:argv+=['--resume',str(p/folder/f'runs/s0_20261026/{arm}/step_000003.pt')]
 spec={**base,'argv':argv,'env':{**base['env'],'CUDA_VISIBLE_DEVICES':gpus}};(d/'spec.json').write_text(json.dumps(spec,indent=2))
 proc=subprocess.Popen([sys.executable,str(p/'run_job.py'),str(d/'spec.json')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);print(name,proc.pid,flush=True)
def wait(name):
 start=time.time()
 while True:
  f=p/name/'status.json'
  if f.exists():
   s=json.loads(f.read_text())
   if s['state']!='running':
    assert s['state']=='completed',s
    return
  if time.time()-start>1800:raise TimeoutError(name)
  time.sleep(2)
launch('single_E1','single','E1','0',stop=True)
launch('disabled_E1','disabled','E1','1')
launch('ddp_E1','ddp','E1','2,3',world=2,stop=True)
launch('single_E0','single','E0','4')
for folder,name,gpus,world in [('single','single_E1','0',1),('ddp','ddp_E1','2,3',2)]:
 wait(name)
 out=p/folder/'runs/s0_20261026/E1'
 # Simulate valid log entries from an update after the saved checkpoint.
 for file in [*out.glob('train_rank*.jsonl'),out/'metrics.jsonl']:
  row=json.loads(file.read_text().splitlines()[-1]);row['step']=4
  with file.open('a') as stream:stream.write(json.dumps(row)+'\n')
 code='from torch.utils.tensorboard import SummaryWriter; w=SummaryWriter('+repr(str(out/'tensorboard'))+'); w.add_scalar("loss/av",999.,4); w.close()'
 subprocess.run([py,'-c',code],check=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
 launch(name+'_resume',folder,'E1',gpus,world=world,resume=True)
for name in ['single_E1_resume','ddp_E1_resume','disabled_E1','single_E0']:wait(name)
(p/'jobs_completed.json').write_text(json.dumps(dict(time=time.time())))
