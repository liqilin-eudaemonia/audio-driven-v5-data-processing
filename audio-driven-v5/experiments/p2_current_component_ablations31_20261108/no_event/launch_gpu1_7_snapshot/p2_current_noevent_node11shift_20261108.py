"""Launch the reviewed no-Event run with node11 physical GPUs1..7.

Only physical device placement changes. Global ranks, batch, source identity,
sample/noise schedule and the frozen training entry remain unchanged.
"""
import hashlib
import json
from pathlib import Path

import p2_current500h_component_ablations as native

SELF=Path(__file__).resolve()
WORK=native.DEFAULT/'no_event'


def layout(cluster):
    if [n['name'] for n in cluster['nodes']]!=['node12','node8','node4','node11']:
        raise ValueError('Expected the approved four-node order')
    result=[];offset=0
    for n in cluster['nodes']:
        expected=list(range(1,8)) if n['name']=='node11' else list(range(8))
        if n['gpus']!=expected:raise ValueError('Unexpected physical GPU allocation')
        result.append(dict(node=n['name'],gpus=expected,rank_offset=offset,
                           ranks=list(range(offset,offset+len(expected)))))
        offset+=len(expected)
    assert offset==31
    return result


def verify():
    receipt=json.loads((WORK/'PHYSICAL_LAYOUT_AUTHORIZED.json').read_text())
    assert receipt['wrapper_sha256']==hashlib.sha256(SELF.read_bytes()).hexdigest()
    assert receipt['cluster_sha256']==hashlib.sha256((WORK/'cluster.json').read_bytes()).hexdigest()
    assert receipt['config_sha256']==hashlib.sha256((WORK/'config.json').read_bytes()).hexdigest()
    assert receipt['user_authorized_node11_gpus']==list(range(1,8))
    layout(json.loads((WORK/'cluster.json').read_text()))


def main():
    verify()
    native.layout=layout
    native.SELF=SELF
    native.main()


if __name__=='__main__':main()
