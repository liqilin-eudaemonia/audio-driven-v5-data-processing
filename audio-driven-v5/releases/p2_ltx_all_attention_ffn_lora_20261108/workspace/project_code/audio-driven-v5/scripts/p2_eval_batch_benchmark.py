"""Temporarily pause exactly one owned worker, benchmark, always resume it."""
import argparse,os,signal,sys,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_testsets_eval.protocol import read,save,sha


def execute(args):
 import torch
 from p2_testsets_eval.runtime import data_config,cached_loaders,observation
 from p2_listen.cache import tree_to,cpu,atomic_torch
 from p2_listen.features import make_observation
 from p2_matrix16.checkpoint import checked
 from p2_matrix16.model import MatrixModel
 from p2_listen.runtime import guided_sample,append_negative
 from p2_testsets_eval.batch_sampling import sample,pack
 d,cfg=data_config()
 from omni_av_p2.ltx import load_renderer,append_conditions
 from omni_av_p2.connector import ConditionInputs
 from omni_av_p2.inference import reference_latents
 from omni_av.ltx_runtime import with_text
 work=args.work;out=args.output;out.mkdir(parents=True,exist_ok=True);p=read(work/'protocol.json')
 torch.set_num_threads(4);torch.cuda.set_device(0)
 renderer,strategy,sampler=load_renderer(cfg,'cuda:0');renderer.set_gradient_checkpointing(False)
 arm=p['arms'][0];saved=checked(arm['checkpoint']);model=MatrixModel(saved['identity']['experiment'],cfg,renderer).cuda()
 model.load_state_dict(saved['model'],strict=True);del saved
 renderer.eval().requires_grad_(False);model.eval().requires_grad_(False)
 negative=tree_to(torch.load(work/'negative/00/payload.pt',map_location='cpu',weights_only=True),'cuda:0')
 items=read(args.samples);prepared=[];negs=[]
 with cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
  for item in items:
   folder=work/'conditions'/item['key'];g=item['geometry']
   payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True)
   text=tree_to(torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True),'cuda:0')
   inputs=ConditionInputs(**tree_to(payload['condition'],'cuda:0'))
   with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
   native=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
   av=reference_latents(cfg,make_observation(observation(item,folder)),native,'cuda:0')
   torch.manual_seed(p['seed']);current=strategy.prepare_training_inputs(with_text(av,text),sampler)
   prepared.append(append_conditions(current,u));negs.append(append_negative(negative,u))
   del payload,inputs,text,u,av,current
 gc.collect();torch.cuda.empty_cache()
 packed,negative_packed=pack(prepared,negs)
 reports=[];baseline=None
 def measure(mode):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  if mode in ['serial','serial_repeat']:
   outputs=[]
   for current,neg in zip(prepared,negs):
    torch.manual_seed(p['seed']);outputs.append(guided_sample(renderer,current,neg,**p['generation']))
   result={k:torch.cat([x[k] for x in outputs]) for k in outputs[0]}
  elif mode=='cfg':
   outputs=[sample(renderer,current,neg,**p['generation'],seed=p['seed'],combine_cfg=True) for current,neg in zip(prepared,negs)]
   result={k:torch.cat([x[k] for x in outputs]) for k in outputs[0]}
  else:result=sample(renderer,packed,negative_packed,**p['generation'],seed=p['seed'],combine_cfg=mode=='both')
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  return cpu(result),dict(mode=mode,seconds=elapsed,seconds_per_sample=elapsed/len(items),
      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
 # Warm the renderer with the actual shapes; measured runs all use the full 30 steps.
 with torch.inference_mode():
  torch.manual_seed(p['seed']);warm=guided_sample(renderer,prepared[0],negs[0],steps=1,video_cfg=3,audio_cfg=7,sigmas=torch.tensor([1.,0.]));del warm
 for mode in args.modes:
  save(out/'status.json',dict(stage='running',mode=mode,at=time.time()))
  try:
   result,report=measure(mode)
   if baseline is None:baseline=result
   report['difference']={}
   for key in result:
    a=result[key].float();b=baseline[key].float();delta=a-b
    report['difference'][key]=dict(exact=torch.equal(a,b),rmse=float(delta.square().mean().sqrt()),
       relative_l2=float(delta.norm()/b.norm()),max_abs=float(delta.abs().max()),
       cosine=float(torch.nn.functional.cosine_similarity(a.flatten()[None],b.flatten()[None])[0]))
   report['first_frame_exact']=torch.equal(result['video'][:,:256],baseline['video'][:,:256])
   atomic_torch(out/(mode+'.pt'),result);del result
  except torch.OutOfMemoryError as e:report=dict(mode=mode,error=str(e))
  reports.append(report);save(out/'results.json',dict(items=[x['key'] for x in items],geometry=[x['geometry'] for x in items],
       checkpoint=arm['checkpoint'],checkpoint_sha256=sha(arm['checkpoint']),generation=p['generation'],seed=p['seed'],results=reports))
  gc.collect();torch.cuda.empty_cache()
 save(out/'status.json',dict(stage='complete',at=time.time()))


def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--samples',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--production-pid',type=int,required=True);p.add_argument('--modes',nargs='+',default=['serial','cfg','samples','both','serial_repeat']);a=p.parse_args()
 command=Path(f'/proc/{a.production_pid}/cmdline').read_bytes()
 if b'p2_testsets_eval.py' not in command or str(a.work).encode() not in command or b'--stage\x00render' not in command:raise ValueError('Not the owned production render worker')
 a.output.mkdir(parents=True,exist_ok=False);save(a.output/'paused_worker.json',dict(pid=a.production_pid,command=command.decode().replace('\x00',' '),at=time.time()))
 os.kill(a.production_pid,signal.SIGSTOP)
 def terminate(signum,frame):raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 try:execute(a)
 except BaseException as e:
  save(a.output/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  os.kill(a.production_pid,signal.SIGCONT);save(a.output/'worker_resumed.json',dict(pid=a.production_pid,at=time.time()))
if __name__=='__main__':main()
