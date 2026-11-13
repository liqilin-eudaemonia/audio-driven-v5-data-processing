"""Final CPU review of the requested checkpoint, coverage and changed Var readout."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=read(WORK/'protocol.json');complete=read(WORK/'COMPLETE.json');audit=read(WORK/'FINAL_AUDIT.json')
    assert complete['passed'] and complete['outputs']==43 and sha(WORK/'protocol.json')==complete['protocol_sha256']
    for path,value in p['execution_code'].items():assert sha(path)==value,path
    mp=read(WORK/'METRICS_PROTOCOL.json')
    for path,value in mp['code'].items():assert sha(path)==value,path
    assert sha(WORK/'ALL_RESULTS.json')==audit['all_results_sha256'] and audit['passed']
    assert p['arms'][0]['original_checkpoint']==str(ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109/step_006500.pt')
    assert p['arms'][0]['original_sha256']==read(ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109/step_006500.complete.json')['sha256']
    assert read(WORK/'MODEL_ACCEPTANCE.json')['targets']==1632
    rows=read(WORK/'DIM_RESULTS.json')['coverage'];dim=read(WORK/'DIM_RESULTS.json')
    sources={}
    for folder in (WORK/'emoca').glob('shard_*'):
        done=read(folder/'EXTRACTION_COMPLETE.json');assert done['success']
        for record in done['records']:
            assert sha(record['path'])==record['sha256'];sources[record['id']]=record
    assert len(rows)==len(sources)==43
    pred=[];gt=[]
    for row in rows:
        indices=np.asarray(row['listener_frame_indices'])
        with np.load(sources[row['id']]['path']) as values:pred.append(values['motion'][indices])
        with np.load(ROOT/'experiments/p2_vico_literature_protocol_20261108/features/gt'/(row['id']+'.npz')) as values:gt.append(values['motion'][indices])
        assert np.isfinite(pred[-1]).all() and np.isfinite(gt[-1]).all()
    a=np.concatenate(pred);g=np.concatenate(gt);assert len(a)==dim['frames'] and len(a)>0
    independent_var={}
    for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
        value=float(np.var(a[:,sl],axis=0,ddof=0).mean())
        expected=dim['models']['ours']['primary']['Var'][group]
        assert abs(value-expected)<1e-9
        independent_var[group]=value
        assert abs(float(np.var(g[:,sl],axis=0,ddof=0).mean())-dim['models']['ours']['primary']['Var_GT'][group])<1e-9
    bp=read(WORK/'bfm/protocol.json');independent_FD={};all_L1={name:[] for name in ['exp','angle','trans']}
    for sample in bp['samples']:
        uid=sample['id'];path=WORK/'bfm/features/prediction'/(uid+'.npz');receipt=read(path.with_suffix('.json'))
        assert sha(path)==receipt['feature_sha256'] and sha(sample['streams']['prediction']['path'])==receipt['input_sha256']
        with np.load(path) as values:pred=values['motion'].astype(np.float64)
        with np.load(WORK/'bfm/features/gt'/(uid+'.npz')) as values:gt=values['motion'].astype(np.float64)
        assert pred.shape==gt.shape==(sample['streams']['gt']['frames'],70)
        for name,sl in [('exp',slice(0,64)),('angle',slice(64,67)),('trans',slice(67,70))]:all_L1[name].append(float(np.abs(pred[:,sl]-gt[:,sl]).mean()))
    bfm=read(WORK/'bfm/RESULTS.json')
    for name,values in all_L1.items():
        value=float(np.mean(values));assert abs(value-bfm['metrics_raw']['FD'][name])<1e-12;independent_FD[name]=value
    assert dim['original_code_parity']['passed'] and dim['original_code_parity']['max_abs']==0
    assert read(WORK/'IMAGE_RESULTS.json')['clips']==43 and audit['decoded_videos']==86
    result=dict(passed=True,requested_checkpoint_verified=True,strict_LoRA_targets=1632,clips=43,
        fully_decoded_videos=86,DIM_common_valid_frames=dim['frames'],independent_requested_Var=independent_var,
        independent_BFM_FD_raw=independent_FD,complete_protocol_and_code_hashes_verified=True,
        DIM_published_code_parity_max_abs=0.,paper_private_alignment_claim=False,at=time.time())
    (WORK/'FINAL_USER_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
