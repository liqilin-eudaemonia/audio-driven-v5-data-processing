"""Use otherwise idle GPU7 space for Whisper and SFT2752; defer Gemma.

The normal full586 controller resumes afterward. No model or prompt changes.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import p2_responsenet_full586_all_lora_20261111 as task
from p2_testsets_eval.protocol import read,save,sha,PYTHON


class GemmaDeferred(Exception):pass


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    p=task.verify(task.COMMON)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20/(torch.cuda.get_device_properties(0).total_memory/2**30),0)
    path=task.COMMON/'locks/cache_0.lock';path.parent.mkdir(exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resources=task.gpu();assert resources['free_gib']>=22
        from p2_testsets_eval import runtime as rt,whisper_runtime as wr
        from p2_listen.config import extractor_config
        from p2_responsenet_random200 import rn_speaker_wave
        from p2_ltx6000_latest_sft_dual200_20261107 import cached_conditioning
        from p2_ltx6000_sft_eval_20261107 import SFTInferenceSession
        d=read(task.COMMON/'DATA_CONFIG.json');cfg=extractor_config(d)
        items=read(task.COMMON/'shared_items.json')
        save(task.WORK/'SMALL_CONDITIONS_POLICY.json',dict(at=time.time(),pid=os.getpid(),wrapper_sha256=sha(SELF),
            protocol_sha256=sha(task.COMMON/'protocol.json'),memory_cap_gib=20,required_free_gib=22,
            planner_batch=p['planner_batch'],whisper_batch=p['whisper_batch'],gemma_deferred=True,
            models_prompts_and_encoding_unchanged=True))
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(wr,'data_config',lambda:(d,cfg)),\
             patch.object(wr,'speaker_wave',rn_speaker_wave):
            save(task.WORK/'status.json',dict(stage='small_memory_Whisper_cache',at=time.time(),physical_gpu=7))
            wr.cache(task.COMMON,items,0,p)
            assert all(rt.ready(task.COMMON/'whisper'/x['key']) for x in items)
            save(task.COMMON/'WHISPER_COMPLETE.json',dict(success=True,at=time.time(),pairs=586))
            save(task.WORK/'status.json',dict(stage='small_memory_SFT2752_cache',at=time.time(),physical_gpu=7))
            import omni_av.ltx_runtime as ltx
            def defer(*args,**kwargs):raise GemmaDeferred()
            with patch.object(rt,'conditioning',cached_conditioning),patch('p2_j_sft_hidden.Reader',SFTInferenceSession),\
                 patch.object(ltx,'NativeTextConditioner',defer):
                try:rt.cache(task.COMMON,items,0,p)
                except GemmaDeferred:pass
            assert all(rt.ready(task.COMMON/'conditions'/x['key']/'qwen') for x in items)
            save(task.COMMON/'SFT2752_COMPLETE.json',dict(success=True,at=time.time(),pairs=586,wrapper_sha256=sha(SELF)))
    stop=read(task.WORK/'STOP');assert stop['owner_script']==SELF.name
    (task.WORK/'STOP').unlink()
    command=[str(PYTHON),'-u',str(task.SELF),'controller']
    log=task.WORK/'small_conditions_native_resume.log'
    with log.open('ab',buffering=0) as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save(task.WORK/'LAUNCH.json',dict(pid=proc.pid,at=time.time(),command=command,physical_gpus=[7],reason='Whisper_and_SFT2752_prefill_finished'))
    save(task.WORK/'SMALL_CONDITIONS_COMPLETE.json',dict(success=True,at=time.time(),pairs=586,native_resume_pid=proc.pid))


if __name__=='__main__':main()
