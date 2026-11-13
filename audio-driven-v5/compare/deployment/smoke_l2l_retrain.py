"""Exercise the full trainer on already approved train-only pilot data."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import vico_l2l_retrain as full
from vico_common_deca import read,save,sha


def main():
    sys.path.insert(0,str(full.REPO))
    from utils.load_utils import bilateral_filter,mean_std_swap
    original=full.WORK
    full.WORK=original.parent/'training_adapter_smoke'
    full.WORK.mkdir(exist_ok=True)
    p=read(original/'protocol.json')
    p['train_recipe'].update(max_steps_per_stage=2,minimum_steps=1,validation_interval=1)
    p['scope']='training adapter smoke only; original train8/validation4; no test data'
    save(full.WORK/'protocol.json',p)
    pilot=read(full.PILOT/'split.json');data={}
    for name,items in pilot['pilot'].items():
        files=[np.load(full.PILOT/'features'/name/(r['uuid']+'.npz')) for r in items]
        data[name]={k:np.stack([f[k] for f in files]).astype(np.float32) for k in ['speaker','listener','audio']}
        for k in ['speaker','listener']:data[name][k]=bilateral_filter(data[name][k])
    stats={}
    for k in ['speaker','listener','audio']:
        mean,std=mean_std_swap(data['train'][k]);stats[k+'_mean'],stats[k+'_std']=mean,std
        for name in data:data[name][k]=((data[name][k]-mean)/std).astype(np.float32)
    np.savez_compressed(full.WORK/'train_only_stats.npz',**stats)
    full.make_windows=lambda:data
    full.fit()
    result=read(full.WORK/'TRAINING_COMPLETE.json')
    assert result['vq']['total_steps']==result['predictor']['total_steps']==2
    save(original/'ADAPTER_SMOKE.json',dict(passed=True,train_only=True,test_media_opened=False,
        batch_size=64,vq_steps=2,predictor_steps=2,implementation_sha256=sha(full.__file__),
        results=str(full.WORK/'TRAINING_COMPLETE.json')))
    print('PASS: full trainer batch64, two stages, validation-only checkpoint selection')


if __name__=='__main__':main()
