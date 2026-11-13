"""Transport-only node17 -> node11 adapter; native training/source identity unchanged."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def layout(cluster):
    if [n['name'] for n in cluster['nodes']]!=['node11','node12','node4','node8']:
        raise ValueError('Expected approved node11/node12/node4/node8 order')
    result=[];offset=0
    for i,n in enumerate(cluster['nodes']):
        g=list(range(7 if i==3 else 8))
        if n['gpus']!=g:raise ValueError('Keep8+8+8+7 and reserve node8 GPU7')
        result.append(dict(node=n['name'],gpus=g,rank_offset=offset,ranks=list(range(offset,offset+len(g)))))
        offset+=len(g)
    return result


import p2_whisper_event_feature_train31 as launcher
spec=importlib.util.spec_from_file_location('unchanged_ablation_cli',ROOT/'scripts/p2_whisper_ablation.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
cli.SELF=Path(__file__).resolve()
config_path=cli.config_path
def commands(*args,**kwargs):
    from unittest.mock import patch
    with patch.object(launcher,'layout',layout):return cli.commands(*args,**kwargs)

if __name__=='__main__':
    from unittest.mock import patch
    with patch.object(launcher,'layout',layout):cli.main()
