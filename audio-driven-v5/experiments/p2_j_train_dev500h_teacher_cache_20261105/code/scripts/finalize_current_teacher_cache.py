#!/usr/bin/env python3
"""Fresh CPU-only finalization; initialize one Torch thread before fork."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import p2_j_current_teacher_cache as task
from p2_listen.contracts import atomic_json,sha
if __name__=='__main__':
    w=task.WORK;r=w/'incidents/media_timeout_retry'
    try:
        result=task.finalize(str(w))
        atomic_json(r/'RECOVERY_COMPLETE.json',dict(success=True,resolved_blocks=3,new_samples=64,final_mixed_complete_sha256=sha(w/'MIXED_COMPLETE.json'),audit_execution='fresh-process; Torch intra/inter-op one thread before 32-process fork',time=time.time()))
        atomic_json(w/'chain_status.json',dict(stage='complete',time=time.time(),result=result));print(json.dumps(result),flush=True)
    except BaseException:
        atomic_json(r/'FINAL_AUDIT_ERROR.json',dict(error=traceback.format_exc(),time=time.time()));raise
