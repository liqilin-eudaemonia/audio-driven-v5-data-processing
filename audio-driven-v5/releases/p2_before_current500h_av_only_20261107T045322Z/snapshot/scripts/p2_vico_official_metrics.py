"""Execute original ViCo scripts; CPBD per-clip parallelism, frame-weighted merge."""
import argparse,concurrent.futures,fcntl,json,os,re,runpy,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EVAL=ROOT/'evaluation/vico_challenge_baseline/evaluations';ARM='base_event_feature_whisper_8000'
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def child(work,metric,index=None):
 import numpy as np,scipy.ndimage,imageio.v2
 # CPBD's imread symbol is only used by its command-line image loader, not compute().
 scipy.ndimage.imread=imageio.v2.imread
 for name,value in [('float',float),('int',int)]:
  if name not in np.__dict__:setattr(np,name,value)
 p=json.loads((work/'protocol.json').read_text());spec=p['datasets']['vico'];pred=work/'scoring'/ARM;anno=Path(spec['annotation']);target=EVAL/f'compute_{metric}.py'
 args=['--pd_video_folder',str(pred),'--anno_file',str(anno),'--task','listener']
 if metric in ['base_metrics','fid','csim']:args+=['--gt_video_folder',spec['gt_dir']]
 if metric=='fid':args+=['--batch_size','512']
 if metric=='csim':target=EVAL/'arcface_torch/compute_csim.py';args+=['--batch_size','512','--weight','/public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth']
 if metric=='cpbd':
  import pandas as pd
  frame=pd.read_csv(anno);row=frame.iloc[[index]];anno=work/'metrics'/f'cpbd_anno_{index:02d}.csv';row.to_csv(anno,index=False);args[args.index('--anno_file')+1]=str(anno)
 sys.path.insert(0,str(target.parent));sys.argv=[str(target),*args];os.chdir(target.parent);runpy.run_path(str(target),run_name='__main__')
def execute_unlocked(work,metric,index=None):
 dest=work/'metrics'/(metric+(f'_{index:02d}' if index is not None else '')+'.log');receipt=dest.with_suffix('.json')
 if receipt.exists():return json.loads(receipt.read_text())
 argv=[sys.executable,str(Path(__file__).resolve()),'--work',str(work),'--child',metric]
 if index is not None:argv+=['--index',str(index)]
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 if metric=='cpbd':env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 start=time.time()
 with dest.open('w') as f:code=subprocess.call(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if code:raise RuntimeError(f'Official {metric} exit {code}; see {dest}')
 text=dest.read_text();patterns={'base_metrics':{'psnr':r'^psnr:\s*([\d.eE+\-]+)','ssim':r'^ssim:\s*([\d.eE+\-]+)'},'fid':{'fid':r'^fid:\s*([\d.eE+\-]+)'},'cpbd':{'cpbd':r'^cpbd:\s*([\d.eE+\-]+)'},'csim':{'csim':r'^cosine similarity:\s*([\d.eE+\-]+)'}}[metric]
 values={k:float(re.findall(expr,text,re.M)[-1]) for k,expr in patterns.items()};report=dict(metric=metric,index=index,values=values,seconds=time.time()-start,log=str(dest));save(receipt,report);return report
def execute(work,metric,index=None):
 (work/'metrics').mkdir(exist_ok=True)
 name=metric+(f'_{index:02d}' if index is not None else '')
 with (work/'metrics'/(name+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return execute_unlocked(work,metric,index)
def run(work):
 work=work.resolve();(work/'metrics').mkdir(exist_ok=True);reports=[]
 # CPBD uses CPUs only and can overlap GPU-based FID/ArcFace on physical GPU7.
 with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
  base=pool.submit(execute,work,'base_metrics')
  pending=[pool.submit(execute,work,'cpbd',i) for i in range(43)]
  for metric in ['fid','csim']:reports.append(execute(work,metric))
  reports.append(base.result())
  cpbd=[f.result() for f in pending]
 pre=json.loads((work/'INPUT_PREFLIGHT.json').read_text());frames={x['id']:x['original_gt']['frames'] for x in pre['samples_audit']}
 import pandas as pd
 anno=pd.read_csv(json.loads((work/'protocol.json').read_text())['datasets']['vico']['annotation']);total=sum(frames[x] for x in anno.uuid)
 values={k:v for r in reports for k,v in r['values'].items()};values['cpbd']=sum(r['values']['cpbd']*frames[anno.iloc[r['index']].uuid] for r in cpbd)/total
 if not all(__import__('math').isfinite(v) for v in values.values()):raise ValueError('Nonfinite official score')
 result=dict(dataset='ViCo paper test43',model='AV+Event+Feature+Whisper step8000',clips=43,frames=total,metrics=values,generation=dict(width=512,height=512,fps=25),evaluation=dict(width=256,height=256,fps=30),aggregation='Official global frame weighting; global FID features; CPBD per-clip official script with exact frame-weighted merge',compatibility=['cpbd scipy.ndimage.imread import alias -> imageio (unused by compute)','cpbd legacy numpy.float/int aliases -> Python float/int'],reports=reports,cpbd_reports=cpbd)
 save(work/'RESULTS.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--child');p.add_argument('--index',type=int);a=p.parse_args()
 if a.child:child(a.work,a.child,a.index)
 else:run(a.work)
