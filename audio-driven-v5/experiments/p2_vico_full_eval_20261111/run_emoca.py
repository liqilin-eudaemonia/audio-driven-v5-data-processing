"""Run the existing official EMOCA path in independent person shards."""
import argparse,os,sys,copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
# Optional experiment logging is unused in coefficient extraction. The shared
# wandb install currently requires newer protobuf; do not change that environment.
sys.modules['wandb']=None
from vico_more_common import read,save,sha
SOURCE=ROOT/'experiments/p2_l2l_public_renderer_20261110'

def prepare(name):
    source=SOURCE/'emoca_eval/protocol.json';p=copy.deepcopy(read(source))
    p['tasks']=[t for t in p['tasks'] if t['role']=='l2l_gpavatar_'+name]
    assert len(p['tasks'])==43
    p['parent_protocol_sha256']=sha(source)
    p['source_files'][str(Path(__file__).resolve())]=sha(__file__)
    folder=OUT/'emoca'/name;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'protocol.json'
    if dest.exists():assert read(dest)==p
    else:save(dest,p)
    for item in ['assets','LICENSE_CONFIRMATION.json']:
        target=folder/item
        if not target.exists():target.symlink_to(SOURCE/'emoca_eval'/item,target_is_directory=item=='assets')
    return folder

def main():
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=['conan','fallon','stephen','trevor']);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    folder=prepare(a.name)
    import vico_literature_emoca as ex
    import vico_more_common as common
    ex.WORK=folder;original=common.gpu_guard
    def guard(n):
        device=original(n)
        import torch
        torch.cuda.set_per_process_memory_fraction(.055,0)
        return dict(device,allocator_fraction=.055)
    common.gpu_guard=guard
    ex.run(a.smoke)

if __name__=='__main__':main()
