"""Measure the same official separator on CPU without using any GPU."""
import importlib.util
import os
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK/'python_deps'),str(WORK/'vendor/IIANet'),str(ROOT/'scripts')]
from p2_avatarforcing_table4_generate import read,save,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from look2hear.models import IIANet
    from look2hear.videomodels import ResNetVideoModel
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic=True
    torch.manual_seed(20)
    conf=yaml.safe_load((WORK/'released_checkpoints/vox2/conf.yml').read_text())
    audio=IIANet.from_pretrain(str(WORK/'released_checkpoints/vox2/best_model.pth'),sample_rate=16000,**conf['audionet']['audionet_config']).cpu().eval()
    video=ResNetVideoModel(pretrain=str(WORK/'issue6_bundle/IIANetv2/weights/lrw_resnet18_mstcn_adamw_s3.pth.tar')).cpu().eval()
    assert all(p.device.type=='cpu' for model in [audio,video] for p in model.parameters())
    uid='ClNUxPsUByg_000210_000221.0'
    folder=WORK/'execution/pilot_a/separated'/uid/'speaker'
    receipt=read(folder/'complete.json')
    for path,value in receipt['files'].items():assert sha(path)==value
    sample=next(s for s in read(WORK/'PROTOCOL.json')['samples'] if s['id']==uid)
    mix,sr=sf.read(sample['mix'],dtype='float32');assert sr==16000
    spec=importlib.util.spec_from_file_location('official_iianet_transform_cpu_probe',WORK/'vendor/IIANet/look2hear/datas/transform.py')
    transform=importlib.util.module_from_spec(spec);spec.loader.exec_module(transform)
    roi=np.load(folder/'mouthroi/speaker1.npz')['data']
    normalized=transform.get_preprocessing_pipelines()['val'](roi)
    inputs=torch.from_numpy(normalized[None,None]).float()
    wave=torch.from_numpy(mix[None,None])
    started=time.time()
    with torch.inference_mode():
        embedding=video(inputs)
        output=audio(wave,embedding)
        forward_seconds=time.time()-started
        repeated=audio(wave,embedding)
    assert output.shape==(1,1,len(mix)) and torch.isfinite(output).all()
    error=float((output-repeated).abs().max());assert error<=1e-6
    assert not torch.cuda.is_initialized()
    result=output[0,0].numpy().copy();gpu=np.load(folder/'estimate.npy')
    np.save(WORK/'CPU_AUDIO_PROBE.npy',result)
    save(WORK/'CPU_AUDIO_PROBE.json',dict(passed=True,id=uid,role='speaker',threads=8,
        CPU_only=True,CUDA_initialized=False,forward_seconds=forward_seconds,
        repeated_forward_max_abs=error,GPU_comparison_max_abs=float(np.abs(result-gpu).max()),
        GPU_comparison_correlation=float(np.corrcoef(result,gpu)[0,1]),
        sample_count=len(result),weights_source_audit_sha256=sha(WORK/'IIANET_MODEL_ASSEMBLY_AUDIT.json'),
        implementation_sha256=sha(__file__),input_ROI_sha256=sha(folder/'mouthroi/speaker1.npz'),
        output_sha256=sha(WORK/'CPU_AUDIO_PROBE.npy')))
    print('CPU separator probe',forward_seconds,'seconds; repeat',error,flush=True)


if __name__=='__main__':main()
