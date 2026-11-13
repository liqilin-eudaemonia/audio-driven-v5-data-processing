from pathlib import Path
import sys,os,json,time
from unittest.mock import patch
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(r/'src'),str(r/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch,av
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
assert '/paired_video_metrics/' not in av.__file__,av.__file__
assert '172.29.189.119' in os.popen('hostname -I').read().split()
torch.set_num_threads(4);torch.cuda.set_device(0)
import p2_all_lora6000_eval_20261109 as e
from p2_testsets_eval import runtime as rt,shared_items as shared
from p2_listen.config import extractor_config
w=e.WORK;p=e.read(w/'protocol.json');assert p['execution_code']==e.execution_code()
items=shared.all_items(w,[],p);assert len(items)==200
pending=[x for x in items if rt.ready(w/'latents'/e.ARM/x['key']) and not rt.ready(w/'generated'/e.ARM/x['key'])]
assert pending
selected=pending[:1];d=e.read(w/'DATA_CONFIG.json');cfg=extractor_config(d)
with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(shared,'all_items',lambda *args:selected):
    rt.decode(w,selected,24,p,p['arms'])
f=w/'generated'/e.ARM/selected[0]['key'];assert rt.ready(f)
m=e.read(f/'metadata.json');assert m['seed']==137 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
e.save(w/'recovery_decode_environment/CANARY_ACCEPTED.json',dict(success=True,at=time.time(),sample=selected[0]['key'],native_av=av.__file__,physical_gpu=0,export=m['export'],production_output_committed=True))
print(json.dumps(dict(success=True,sample=selected[0]['key'],native_av=av.__file__)),flush=True)
