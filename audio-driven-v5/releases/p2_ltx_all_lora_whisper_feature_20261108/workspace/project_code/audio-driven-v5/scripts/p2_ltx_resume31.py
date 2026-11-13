"""Reuse the proven explicit-rank 8+8+8+7 launcher for the native LTX joint model."""
from pathlib import Path
import argparse,sys,runpy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read
import p2_whisper_event_feature_train31 as launcher


def layout(cluster):
    if [n['name'] for n in cluster['nodes']]!=['node12','node8','node4','node11']:
        raise ValueError('Expected the existing four-node order')
    result=[];offset=0
    for i,node in enumerate(cluster['nodes']):
        gpus=list(range(7 if i==3 else 8))
        if node['gpus']!=gpus:raise ValueError('Reserve the whole node11 GPU7')
        result.append(dict(node=node['name'],gpus=gpus,rank_offset=offset,
                           ranks=list(range(offset,offset+len(gpus)))))
        offset+=len(gpus)
    if offset!=31:raise ValueError('Expected31 ranks')
    return result


launcher.layout=layout
launcher.SELF=Path(__file__).resolve()


def commands(cluster,config,resume=None,attempt=None):
    return launcher.commands(cluster,config,resume,attempt)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='train':
        native=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='native_ltx_resume31')
        return native['main']()
    p=argparse.ArgumentParser();p.add_argument('command',choices=['node'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--node-index',type=int,required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();return launcher.node(read(a.config),a.node_index,a.resume,a.attempt)


if __name__=='__main__':main()
