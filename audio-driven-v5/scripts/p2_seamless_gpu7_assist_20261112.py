"""Use only the GPU7 capacity explicitly released after ResponseNet completion."""
import os,sys,json,subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_lora6000_seamless_full_gpu4_20261112 as native
from p2_testsets_eval.protocol import read,save
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
RN=ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111'


def eligible():
 if not (RN/'COMPLETE.json').exists():return False
 if not read(RN/'COMPLETE.json').get('success'):return False
 return all((RN/f'step_{s:06d}/FINAL_AUDIT.json').exists() and read(RN/f'step_{s:06d}/FINAL_AUDIT.json').get('success') for s in [6000,6500])


def gpu():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
 r=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
 assert r[0].strip()==UUID
 return dict(free_gib=float(r[1])/1024,total_gib=float(r[2])/1024,util=float(r[3]))


def main():
 assert eligible(),'ResponseNet6000 and6500 remain the first priority'
 auth=read(native.WORK/'ASSIST_GPU7.json');assert auth['authorized'] and auth['safe_shared_claims_verified']
 state=read(native.WORK/'devices/render_0.json');assert state.get('claim_protocol')=='per-sample-flock-v1'
 original_save=native.save
 def record(path,value):
  path=Path(path)
  if path.name=='MODEL_ACCEPTANCE.json':path=path.with_name('MODEL_ACCEPTANCE_GPU7.json')
  if isinstance(value,dict) and 'physical_gpu' in value:value=dict(value,physical_gpu=7)
  return original_save(path,value)
 from p2_testsets_eval import runtime as rt
 old_progress=rt.progress
 def progress(work,stage,rank,**kw):return old_progress(work,stage,7 if stage=='render' else rank,**kw)
 with patch.object(native,'gpu',gpu),patch.object(native,'save',record),patch.object(rt,'progress',progress):
  native.worker('render',7)
 save(native.WORK/'GPU7_GENERATION_FINISHED.json',dict(at=__import__('time').time(),passed=True,physical_gpu=7))


if __name__=='__main__':main()
