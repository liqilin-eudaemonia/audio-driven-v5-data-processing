"""CPU-only residency checks: same-process pause/resume and original-backend reuse keys."""
from runtime import *
from controller import launch
from resource_control import pause_owned,resume_owned,defer_owned
checks=[];d=WORK/'checks';counter=d/'pause_counter.txt';counter.unlink(missing_ok=True)
r=launch('test_resident_pause','checks/pause_probe.py',[str(counter)],sys.executable,7,0)
try:
    for _ in range(100):
        if counter.exists() and int(counter.read_text() or 0)>=3:break
        time.sleep(.1)
    before_pid=read(Path(r['directory'])/'running.json')['pid']
    assert pause_owned(r,'CPU residency regression test')
    time.sleep(.2);a=counter.read_text();time.sleep(.4);assert counter.read_text()==a
    resume_owned(r);time.sleep(.5);assert int(counter.read_text())>int(a)
    assert read(Path(r['directory'])/'running.json')['pid']==before_pid
    checks.append('Pause/resume retains PID and process state; no restart')
finally:defer_owned(r,'CPU test cleanup')
activate_readers()
import metric_launcher as adapter
from evaluation.video.paired_metrics import backends
calls=[]
class Dummy:
    def __init__(self,*args,**kwargs):calls.append((args,kwargs))
backends.Networks=Dummy;backends.Faces=Dummy
adapter.WORK=WORK/'checks/cache_constructor';adapter.WORK.mkdir(exist_ok=True)
adapter.install_backend_cache()
x=backends.Networks({'asset':'same'},'cuda:0',64)
for _ in range(10):assert backends.Networks({'asset':'same'},'cuda:0',64) is x
assert len(calls)==1
assert backends.Networks({'asset':'different'},'cuda:0',64) is not x
assert len(calls)==2
checks.append('Same backend configuration reuses one constructor; changed assets get a different instance')
import torch
before=(torch.backends.cudnn.allow_tf32,torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32)
with adapter.original_metric_precision():
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
assert before==(torch.backends.cudnn.allow_tf32,torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32)
checks.append('Resident scorer uses and restores the original standalone precision flags')
assert read(WORK/'RESOURCE_POLICY.json')['time_slice_enabled'] is False
source=(WORK/'controller.py').read_text();assert "'--limit'" not in source and "'--limit-batches'" not in source
checks.append('No clip quotas and no periodic model teardown in the resident controller')
save(WORK/'RESIDENT_CPU_CHECK.json',dict(passed=True,checks=checks,cuda_allocated=False,at=time.time()))
print('Passed',len(checks),'CPU-only residency checks')
