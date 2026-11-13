"""L2L official forward/sampling with GPU-resident inputs and model-level resume."""
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import numpy as np
from vico_more_common import ROOT, save, read, sha
from vico_fast_registry import BASE as ARCHIVE_BASE, NAMES, node8_guard, assets_match, l2l_group
from vico_l2l import REPO, aligned_window
ARCHIVE = ARCHIVE_BASE / 'learning2listen'
BASE = ARCHIVE_BASE / 'dystream'
WORK = ROOT / 'experiments/p2_vico_inference_fast_20261110/learning2listen'


def install_resident_inputs(module):
    """Reuse the official create_data_vq body; replace its three transfer statements only."""
    import torch
    import utils.load_utils as source
    text = (REPO / 'utils/load_utils.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'create_data_vq')
    targets = {'speakerData': 'speakerData_np', 'listenerData': 'listenerData_np', 'audioData': 'audioData_np'}
    changed = []
    for statement in node.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in targets:
                statement.value = ast.parse('_resident(' + targets[name] + ')', mode='eval').body
                changed.append(name)
    assert set(changed) == set(targets)
    def resident(value):
        if isinstance(value, torch.Tensor):
            # Match the contiguous layout produced by the original H2D copies.
            return value.detach().contiguous()
        return torch.from_numpy(value).cuda()
    env = dict(source.__dict__, _resident=resident)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(REPO / 'utils/load_utils.py'), 'exec'), env)
    module.create_data_vq = env['create_data_vq']


def audio_features(item):
    # Same mel function; new caches are written only inside the new work directory.
    import vico_l2l as original
    previous = original.WORK
    original.WORK = WORK
    try:
        return original.audio_features(item)
    finally:
        original.WORK = previous


def prepare(out):
    global WORK
    WORK = Path(out).resolve()
    assert WORK != ARCHIVE
    WORK.mkdir(parents=True, exist_ok=True)
    protocol = read(ARCHIVE / 'protocol.json')
    assert assets_match(protocol)
    for name in ['DECA_COMPLETE.json', 'DECA_PROTOCOL.json']:
        shutil.copyfile(ARCHIVE / name, WORK / name)
    link = WORK / 'features'
    if not link.exists():link.symlink_to(ARCHIVE / 'features', target_is_directory=True)
    (WORK / 'audio_features').mkdir(exist_ok=True)
    for sample in protocol['samples']:
        src = ARCHIVE / 'audio_features' / (sample['id'] + '.npy')
        dest = WORK / 'audio_features' / src.name
        if src.exists() and not dest.exists():dest.symlink_to(src)
    for name in NAMES:
        if l2l_group(WORK, name, protocol) is not None:
            continue
        # An invalid reused group must never be regenerated through an archive symlink.
        dest = WORK / 'generated' / name
        if dest.is_symlink():dest.unlink()
        if (dest / 'COMPLETE.json').exists():(dest / 'COMPLETE.json').unlink()
        if l2l_group(ARCHIVE, name, protocol) is not None:
            src = ARCHIVE / 'generated' / name
            dest = WORK / 'generated' / name
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                dest.symlink_to(src, target_is_directory=True)
    # Only complete person groups can be skipped: changing the batch changes RNG sampling.

def generate(smoke=False):
    if not smoke:
        protocol=read(ARCHIVE/'protocol.json')
        groups=[l2l_group(WORK,n,protocol) for n in NAMES]
        if all(group is not None for group in groups):
            save(WORK/'protocol.json',protocol)
            save(WORK/'GENERATION_COMPLETE.json',dict(models=NAMES,outputs=sum(map(len,groups)),reused=True,gpu_requested=False))
            return
    import torch
    node8_guard()
    assert (WORK/'DECA_COMPLETE.json').exists()
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    from modules.fact_model import setup_model
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    import test_vq_decoder
    install_resident_inputs(test_vq_decoder)
    generate_prediction = test_vq_decoder.generate_prediction
    from utils.load_utils import bilateral_filter
    p=json.loads((BASE/'protocol.json').read_text());items=p['samples']
    if smoke:items=[min(items,key=lambda x:x['duration'])]
    torch.set_num_threads(2);args=types.SimpleNamespace(sample_idx=None)
    configs={n:REPO/'configs/vq'/('delta_v6.json' if n=='conan' else n+'_delta_v6.json') for n in NAMES}
    code=[Path(__file__),REPO/'test_vq_decoder.py',REPO/'utils/load_utils.py',REPO/'vqgan/vqmodules/gan_models.py',REPO/'modules/fact_model.py']
    protocol=dict(model='Learning2Listen four person-specific pretrained weights',dataset='ViCo paper test43',fps=30,
        feature_space='official DECA expression50 + global head rotation3 + jaw rotation3; not BFM64/translation',
        audio='author issue2 corrected melspectrogram: sr16000 win400 hop160 nfft512 n_mels128 power2; PIL Lanczos resize to4T; no log/MFCC',
        audio_reference='https://github.com/evonneng/learning2listen/issues/2',
        model_names=NAMES,seed=137,batch_size=len(items),
        adaptation='user-approved corrected audio slice4*t:4*(t+40); unchanged official predictor, VQ and nucleus sampling',
        history='32 standardized zero frames before clip origin for listener/speaker/audio; no target listener input at any time',
        tail='speaker edge pad; zero raw mel pad; target length padded to multiple32 for VQ decoding; only original frames scored',
        loop='8-frame/one-token AR; past4 tokens decoded/requantized by original create_data_vq; decode generated tokens in32-frame blocks',
        source_only=True,listener_future_input=False,photorealistic_renderer=False,
        deca_protocol_sha256=sha(WORK/'DECA_PROTOCOL.json'),code_sha256={str(q):sha(q) for q in code},samples=items,models=json.loads((ARCHIVE/'protocol.json').read_text())['models'])
    mel=[audio_features(s) for s in items]
    # Read ONLY speaker features in this generation process. Listener GT arrays
    # are opened later by a separate CPU scoring command.
    speaker=[bilateral_filter(np.load(WORK/'features/speaker'/(s['id']+'.npz'))['motion'].astype(np.float32)[None])[0] for s in items]
    length=max(s['frames'] for s in items);padded=(length+31)//32*32;batch=len(items)
    start=time.time()
    for name in (['conan'] if smoke else NAMES):
        if not smoke and (WORK/'generated'/name/'COMPLETE.json').exists():continue
        torch.manual_seed(137);np.random.seed(137)
        config=json.loads(configs[name].read_text());vqconfig=json.loads((REPO/config['l_vqconfig']).read_text())
        checkpoint=REPO/config['model_path']/(config['tag']+config['pipeline']+'_best.pth')
        vqcheckpoint=REPO/'vqgan'/vqconfig['model_path']/(vqconfig['tag']+vqconfig['pipeline']+'_best.pth')
        xstats_path=REPO/config['model_path']/(config['tag']+config['pipeline']+'_preprocess_core.npz')
        ystats_path=REPO/'vqgan'/vqconfig['model_path']/(vqconfig['tag']+vqconfig['pipeline']+'_preprocess_core.npz')
        xs=np.load(xstats_path);ys=np.load(ystats_path)
        mean_x,std_x=xs['body_mean_X'].reshape(56),xs['body_std_X'].reshape(56)
        mean_a,std_a=xs['body_mean_audio'].reshape(128),xs['body_std_audio'].reshape(128)
        for stat in [xs['body_std_X'],xs['body_std_audio'],ys['body_std_Y']]:assert np.isfinite(stat).all() and (stat>0).all()
        protocol['models'][name]=dict(config=str(configs[name]),vq_config=config['l_vqconfig'],files={str(q):sha(q) for q in [checkpoint,vqcheckpoint,xstats_path,ystats_path]})
        save(WORK/('SMOKE_PROTOCOL.json' if smoke else 'protocol.json'),protocol)
        vq,vopt,_=setup_vq_transformer(args,vqconfig,load_path=str(vqcheckpoint),test=True);vq.eval()
        generator,gopt,_=setup_model(config,vqconfig,mask_index=0,test=True,load_path=str(checkpoint));generator.eval()
        x=np.zeros((batch,padded+32,56),np.float32);a=np.zeros((batch,(padded+32)*4,128),np.float32)
        for i,s in enumerate(items):
            n=s['frames'];sx=speaker[i]
            xn=(sx-mean_x)/std_x;an=(mel[i]-mean_a)/std_a
            x[i,32:32+n]=xn;x[i,32+n:]=xn[-1]
            a[i,128:128+n*4]=an
            a[i,128+n*4:]=(0-mean_a)/std_a
        x=torch.from_numpy(x).cuda();a=torch.from_numpy(a).cuda()
        history=torch.zeros((batch,32,56),device='cuda');tokens=None;shape=None;alignment=[]
        with torch.no_grad():
            for t in range(0,padded,8):
                past=history if tokens is None else tokens[:,-4:]
                xw,aw=aligned_window(x,a,t)
                prediction,prob,inputs,shape=generate_prediction(config,args,vq,generator,xw,past,aw,32,8,t//8,4,btc=shape)
                if tokens is None:tokens=inputs['listener_past']
                tokens=torch.cat((tokens,prediction[:,0]),dim=1)
                alignment.append(dict(speaker_start=t-32,audio_mel_start=4*(t-32),speaker_frames=40,audio_mel_frames=160))
                if t % 64 == 0 or t+8 >= padded:save(WORK/'status.json',dict(stage='generating',model=name,generated_frames=min(t+8,padded),padded_frames=padded,samples=batch,elapsed_s=time.time()-start))
            code=tokens[:,4:];assert code.shape==(batch,padded//8)
            decoded=[]
            for start_token in range(0,code.shape[1],4):decoded.append(vq.module.decode_to_img(code[:,start_token:start_token+4],shape).cpu().numpy())
        pred=np.concatenate(decoded,axis=1)*ys['body_std_Y']+ys['body_mean_Y']
        assert pred.shape==(batch,padded,56) and np.isfinite(pred).all()
        folder=WORK/('smoke' if smoke else 'generated')/name;folder.mkdir(parents=True,exist_ok=True)
        for i,s in enumerate(items):
            values=pred[i,:s['frames']].astype(np.float32);dest=folder/(s['id']+'.npz')
            np.savez_compressed(dest,motion=values,token_indices=code[i].cpu().numpy())
            save(dest.with_suffix('.json'),dict(id=s['id'],model=name,frames=len(values),fps=30,space='DECA50+3+3',
                source_only=True,listener_future_input=False,feature_sha256=sha(dest)))
        save(folder/'alignment.json',alignment)
        save(folder/'COMPLETE.json',dict(model=name,samples=batch,frames=sum(s['frames'] for s in items),padded_frames=padded,seed=137))
        del vq,generator,vopt,gopt,tokens,code,x,a,history;torch.cuda.empty_cache()
    if smoke:save(WORK/'SMOKE_COMPLETE.json',dict(model='conan',samples=1,frames=items[0]['frames'],finite=True))
    else:save(WORK/'GENERATION_COMPLETE.json',dict(models=NAMES,outputs=172,frames_per_model=sum(s['frames'] for s in items),seconds=time.time()-start))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=WORK)
    a=p.parse_args();prepare(a.work);generate()
