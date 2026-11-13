"""Freeze approved ViCo splits and upstream window-first preprocessing."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(os.environ.get('L2L_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
SOURCE=ROOT/'experiments/common_method_comparison_20261106/l2l_retrained'
UPSTREAM=ROOT/'compare/learning2listen/src'
DEFAULT_WORK=ROOT/'experiments/p2_l2l_reliable_retrain_20261107'


def prepare(work):
    from l2l_reliable_core import read,save,sha
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if (work/'DATA_COMPLETE.json').exists():
        receipt=read(work/'DATA_COMPLETE.json')
        for path,expected in receipt['files'].items():assert sha(work/path)==expected,path
        return receipt
    sys.path.insert(0,str(UPSTREAM))
    from utils.load_utils import bilateral_filter,mean_std_swap
    split=read(SOURCE/'split.json');origins=read(SOURCE/'WINDOWS.json')
    assert len(split['train'])==351 and len(split['validation'])==46 and len(split['quarantined_training_rows'])==1
    source_hashes={};features={}
    for name in ['train','validation']:
        allowed={r['uuid'] for r in split[name]}
        assert not allowed.intersection(split['held_out_test_ids']+split['held_out_ood_ids'])
        features[name]={}
        for uid in sorted(allowed):
            path=SOURCE/'features'/name/(uid+'.npz');receipt=read(path.with_suffix('.json'))
            assert sha(path)==receipt['feature_sha256'],uid
            source_hashes[str(path)]=receipt['feature_sha256']
            features[name][uid]=path
    arrays={};coverage={}
    for name in ['train','validation']:
        values={k:[] for k in ['speaker','listener','audio']};cache={}
        for row in origins[name]['window_origins']:
            uid=row['id'];start=row['start_frame'];assert uid in features[name]
            if uid not in cache:cache[uid]=np.load(features[name][uid])
            f=cache[uid]
            for key in ['speaker','listener']:values[key].append(f[key][start:start+64])
            values['audio'].append(f['audio'][4*start:4*(start+64)])
        arrays[name]={k:np.stack(v).astype(np.float32) for k,v in values.items()}
        for key in ['speaker','listener']:
            # Match upstream load_data: filter each64-frame window, THEN normalize.
            arrays[name][key]=bilateral_filter(arrays[name][key])
        assert arrays[name]['listener'].shape[1:]==(64,56)
        assert arrays[name]['audio'].shape[1:]==(256,128)
        assert all(np.isfinite(v).all() for v in arrays[name].values())
        coverage[name]={k:v for k,v in origins[name].items() if k!='window_origins'}
    stats={}
    for key in ['speaker','listener','audio']:
        mean,std=mean_std_swap(arrays['train'][key]);assert np.isfinite(std).all() and (std>0).all()
        stats[key+'_mean']=mean;stats[key+'_std']=std
        for name in arrays:arrays[name][key]=((arrays[name][key]-mean)/std).astype(np.float32)
    folder=work/'data';folder.mkdir(exist_ok=True);files={}
    for name in arrays:
        for key,value in arrays[name].items():
            path=folder/(name+'_'+key+'.npy');np.save(path,value,allow_pickle=False);files[str(path.relative_to(work))]=sha(path)
    path=folder/'train_only_stats.npz';np.savez_compressed(path,**stats);files[str(path.relative_to(work))]=sha(path)
    save(work/'split.json',split);files['split.json']=sha(work/'split.json')
    save(work/'window_origins.json',{k:v['window_origins'] for k,v in origins.items()});files['window_origins.json']=sha(work/'window_origins.json')
    save(work/'source_features.json',source_hashes);files['source_features.json']=sha(work/'source_features.json')
    receipt=dict(schema='l2l-upstream-window-preprocessing-v1',files=files,coverage=coverage,
        sources=len(source_hashes),seed_for_split=137,statistics_scope='train windows only',
        preprocessing='raw cached motion ->64frame windows ->upstream bilateral_filter ->upstream mean_std_swap',
        changed_from_v1_v2='smoothing per window matches upstream, instead of smoothing whole clips',
        upstream_helpers_sha256=sha(UPSTREAM/'utils/load_utils.py'),test_media_opened=False)
    save(work/'DATA_COMPLETE.json',receipt)
    return receipt


def load(work):
    work=Path(work);result={}
    for split in ['train','validation']:
        result[split]={key:np.load(work/'data'/(split+'_'+key+'.npy'),mmap_mode='r') for key in ['speaker','listener','audio']}
        y=result[split]['listener']
        result[split]['vq']=np.concatenate([y[:,:32],y[:,32:]],axis=0)
    stats=dict(np.load(work/'data/train_only_stats.npz'))
    return result,stats


def freeze(work):
    from l2l_reliable_core import read,save,sha
    work=Path(work)
    assert not any(work.glob('seed_*/*/*.pth')),'Formal training exists; use a NEW run instead of refreezing'
    target=work/'code_snapshot';target.mkdir(exist_ok=True)
    source_root=ROOT/'compare/deployment'
    for name in ['l2l_reliable_core.py','l2l_reliable_data.py','l2l_reliable.py','l2l_reliable_preflight.py','l2l_reliable_boundary_checks.py','run_l2l_preflight.py','vico_common_deca.py']:
        shutil.copyfile(source_root/name,target/name)
    metric_source=ROOT/'scripts/p2_vico_motion_metrics.py'
    shutil.copyfile(metric_source,target/metric_source.name)
    # The copied scorer imports its copied metric definitions, not mutable scripts.
    scorer=target/'vico_common_deca.py'
    scorer.write_text(scorer.read_text().replace("sys.path.insert(0, str(ROOT / 'scripts'))", "sys.path.insert(0, str(Path(__file__).resolve().parent))"))
    frozen_repo=target/'l2l_src';frozen_repo.mkdir(exist_ok=True)
    skipped={'data','outputs','models','.git','__pycache__','.venv'}
    for base,dirs,names in os.walk(UPSTREAM):
        dirs[:]=[d for d in dirs if d not in skipped]
        for name in names:
            path=Path(base)/name
            if path.suffix not in ['.py','.json']:continue
            dest=frozen_repo/path.relative_to(UPSTREAM);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    files={str(p.relative_to(work)):sha(p) for p in target.rglob('*') if p.is_file()}
    versions={}
    for name in ['torch','torchvision','numpy','opencv-python','scipy','librosa','Pillow']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    save(work/'CODE_FROZEN.json',dict(files=files,versions=versions))
    return frozen_repo
