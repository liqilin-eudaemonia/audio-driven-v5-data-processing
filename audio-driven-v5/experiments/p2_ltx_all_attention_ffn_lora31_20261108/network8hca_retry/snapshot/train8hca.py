"""Run the accepted expanded-LoRA model on the current 8+8+8+7 allocation."""
import hashlib,json
from pathlib import Path
import sys
sys.path[:0]=[str(Path('/private/liqi_workspace/project_code/audio-driven-v5/scripts')),str(Path('/private/liqi_workspace/project_code/audio-driven-v5/src'))]
import p2_ltx_all_lora_train as native
import p2_current500h_component_ablations as placement
SELF=Path(__file__).resolve()
WORK=native.DEFAULT_OUTPUT

def layout(cluster):
 if [n['name'] for n in cluster['nodes']]!=['node12','node8','node4','node11']:
  raise ValueError('Unexpected node order')
 result=[];offset=0
 for n in cluster['nodes']:
  g=list(range(1,8)) if n['name']=='node11' else list(range(8))
  if n['gpus']!=g:raise ValueError('Reserve node11 GPU0; require current approved allocation')
  result.append(dict(node=n['name'],gpus=g,rank_offset=offset,ranks=list(range(offset,offset+len(g)))))
  offset+=len(g)
 assert offset==31
 return result

def verify():
 receipt=json.loads((WORK/'network8hca_retry/NETWORK_RECEIPT.json').read_text())
 for key,p in [('wrapper_sha256',SELF),('cluster_sha256',WORK/'cluster.json'),('config_sha256',WORK/'config.json')]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=receipt[key]:raise ValueError('Launch identity changed: '+key)
 layout(json.loads((WORK/'cluster.json').read_text()))

def main():
 verify();placement.layout=layout;native.SELF=SELF;native.main()
if __name__=='__main__':main()
