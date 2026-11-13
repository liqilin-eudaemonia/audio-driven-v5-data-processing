"""Apply the same official EMOCA readout to completed ListenFormer RGB videos."""
import run_emoca as common
from pathlib import Path
import copy
from vico_more_common import read,save,sha
ROOT=common.ROOT;OUT=common.OUT
SOURCE=ROOT/'experiments/p2_vico_additional_baselines_20261107/listenformer'
folder=OUT/'emoca/listenformer';folder.mkdir(parents=True,exist_ok=True)
parent=common.SOURCE/'emoca_eval/protocol.json';p=copy.deepcopy(read(parent))
r=read(SOURCE/'RENDER_COMPLETE.json');assert r['success'] and r['outputs']==43
p['tasks']=[dict(id=x['id'],role='listenformer',video=x['path'],video_sha256=x['sha256'],expected_frames=x['frames']) for x in r['records']]
p['parent_protocol_sha256']=sha(parent);p['render_manifest_sha256']=sha(SOURCE/'RENDER_COMPLETE.json')
p['source_files'].update({str(Path(__file__).resolve()):sha(__file__),str(Path(common.__file__).resolve()):sha(common.__file__)})
path=folder/'protocol.json'
if path.exists():assert read(path)==p
else:save(path,p)
for name in ['assets','LICENSE_CONFIRMATION.json']:
 path=folder/name
 if not path.exists():path.symlink_to(common.SOURCE/'emoca_eval'/name,target_is_directory=name=='assets')
import vico_literature_emoca as ex
import vico_more_common
ex.WORK=folder;original=vico_more_common.gpu_guard

def guard(n):
 info=original(n)
 import torch
 torch.cuda.set_per_process_memory_fraction(.055,0)
 return dict(info,allocator_fraction=.055)

vico_more_common.gpu_guard=guard
ex.run(False)
