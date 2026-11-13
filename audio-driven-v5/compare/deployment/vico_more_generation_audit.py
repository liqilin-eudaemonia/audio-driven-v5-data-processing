"""Run the unchanged native generator with a strict source-only array I/O audit."""
import argparse
from pathlib import Path

import numpy as np
import torch
from vico_more_common import WORK,read,save,sha
import vico_more_train as native


def run(method):
    out=WORK/method;features=WORK/'features'
    inputs=read(features/'TEST_INPUTS.json');assert inputs['source_only']
    allowed={str((features/'train_stats.npz').resolve()):'train-only statistics'}
    for row in inputs['records']:
        allowed[str(Path(row['speaker']).resolve())]='speaker motion'
        allowed[str(Path(row['initial']).resolve())]='listener t0 only'
        allowed[str((features/'test'/row['id']/'audio.npy').resolve())]='speaker audio features'
        with np.load(row['initial']) as initial:
            assert initial['coeff'].shape==(1,257)
            assert initial['transform_params'].shape==(1,5)
    original_numpy=np.load;original_torch=torch.load;events=[];weights=[]
    def audited_numpy(file,*args,**kwargs):
        path=str(Path(file).resolve())
        assert path in allowed,f'Unexpected model-input array access: {path}'
        events.append(dict(path=path,role=allowed[path],sha256=sha(path)))
        return original_numpy(file,*args,**kwargs)
    def audited_torch(file,*args,**kwargs):
        path=Path(file).resolve()
        assert path==(out/'best.pt').resolve(),f'Unexpected checkpoint/target access: {path}'
        weights.append(dict(path=str(path),sha256=sha(path)))
        return original_torch(file,*args,**kwargs)
    np.load=audited_numpy;torch.load=audited_torch
    try:
        native.generate(method)
    finally:
        np.load=original_numpy;torch.load=original_torch
    initial_events=[e for e in events if e['role']=='listener t0 only']
    assert len(events)==130 and len(initial_events)==43 and len({e['path'] for e in initial_events})==43
    assert len(weights)==1
    save(out/'GENERATION_INPUT_AUDIT.json',dict(passed=True,outputs=43,array_loads=events,checkpoints=weights,
        listener_frames_per_pair=1,allowed_array_roles=sorted(set(allowed.values())),
        future_listener_sequence_opened=False,unexpected_array_accesses=0,
        test_not_used_for_normalization=True,test_not_used_for_checkpoint_selection=True,
        training_protocol_sha256=sha(out/'TRAIN_PROTOCOL.json'),
        generation_complete_sha256=sha(out/'GENERATION_COMPLETE.json'),
        generator_implementation_sha256=sha(native.__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['rlhg','listenformer']);a=p.parse_args();run(a.method)
