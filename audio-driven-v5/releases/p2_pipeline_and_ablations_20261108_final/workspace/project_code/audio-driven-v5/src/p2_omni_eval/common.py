import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
from p2_glare_eval.common import ROOT,read,sha,digest,atomic_json,checked_files,artifact_valid,complete,job_key

METRICS=('fvd','fd','utmos','sync','text')
DEFAULTS=dict(fvd_frames=32,fvd_stride=1,crop_seed=42,metric_seed=42,
    silence_rms=1e-5,sync_min_track=5,sync_vshift=15,sync_batch_size=20,bert_layers=17)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def code_identity():
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/p2_glare_eval').glob('*.py'))
    files += [ROOT/'scripts'/n for n in ('p2_omni_eval.py','p2_eval_generate.py')]
    return {str(p):sha(p) for p in sorted(files)}


def environment(python):
    script="import importlib.metadata as m,json,sys; names=['torch','torchvision','torchaudio','numpy','scipy','opencv-python','av','bert-score','nltk','timm','transformers','python-speech-features','scenedetect']; installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps(dict(python=sys.version,packages={n:installed.get(n) for n in names})))"
    return json.loads(subprocess.check_output([str(python),'-c',script],text=True))


def media_tools():
    result={}
    for name in ('ffmpeg','ffprobe'):
        path=shutil.which(name)
        if not path:raise FileNotFoundError(name)
        result[name]=dict(path=path,sha256=sha(path),version=subprocess.check_output([path,'-version'],text=True).splitlines()[0])
    return result


def build_runtime(catalog_path):
    catalog=read(catalog_path);a=catalog['metrics']
    assets={k:a[k] for k in ('i3d','hf_home','nltk_data','utmos_code','utmos_checkpoint','bert_model','sync_root','sync_python')}
    assets['metrics_python']=catalog['metrics_python'];assets['reference_root']=str(ROOT.parent/'OmniResponse/evaluation')
    reference=Path(assets['reference_root'])
    files=[reference/p for p in ('video/fvd/frechet_video_distance.py','video/fd/compute_fd_features.py',
        'audio/audio_quality/utmosv2_avg.py','audio/audio_sync/SyncNetInstance_calc_scores.py',
        'text/compute_meteor.py','text/compute_rouge-l.py','text/compute_distinct.py','text/compute_bertscore.py')]
    files+=list((Path(a['utmos_code'])/'utmosv2').rglob('*.py'))
    files+=list(Path(a['sync_root']).glob('*.py'))+list((Path(a['sync_root'])/'detectors').rglob('*.py'))
    source={str(p):sha(p) for p in sorted(set(files))}
    weights=[Path(a['i3d']),Path(a['utmos_checkpoint']),Path(a['sync_root'])/'data/syncnet_v2.model',Path(a['sync_root'])/'detectors/s3fd/weights/sfd_face.pth']
    # Include auxiliary wav2vec/EfficientNet/tokenizer/config and NLTK resources.
    for base in (Path(a['hf_home'])/'hub',Path(a['nltk_data'])):
        weights += [p for p in base.rglob('*') if p.is_file() and '.locks' not in p.parts and '.cache' not in p.parts and '__pycache__' not in p.parts]
    models={str(p):sha(p) for p in sorted(set(weights))}
    return dict(schema='p2-omni-runtime-v1',paths=assets,source=source,models=models,
        environments={k:environment(assets[k]) for k in ('metrics_python','sync_python')},media_tools=media_tools())


def check_runtime(runtime,full=False):
    if runtime['schema']!='p2-omni-runtime-v1':raise ValueError('Invalid runtime')
    if runtime['media_tools']!=media_tools():raise ValueError('ffmpeg/ffprobe changed')
    checked_files(runtime['source'])
    if full:checked_files(runtime['models'])
    for key,expected in runtime['environments'].items():
        if environment(runtime['paths'][key])!=expected:raise ValueError('Evaluation environment changed: '+key)


def subprocess_env(runtime,device='cpu'):
    a=runtime['paths'];env=dict(os.environ,HF_HOME=a['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NLTK_DATA=a['nltk_data'],TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('LD_LIBRARY_PATH',None)
    if device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    return env


def check_device(device):
    if device=='cpu':return
    if not device.startswith('cuda:') or not device[5:].isdigit():raise ValueError('Use cpu or explicit cuda:N')
    index=int(device[5:]);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=visible.split(',')[index] if visible else str(index)
    query=['nvidia-smi','-i',physical,'--query-gpu=memory.used','--format=csv,noheader,nounits']
    if int(subprocess.check_output(query,text=True).strip())>=1024:raise RuntimeError('GPU busy; leave training undisturbed')
    query=['nvidia-smi','-i',physical,'--query-compute-apps=pid','--format=csv,noheader,nounits']
    if subprocess.check_output(query,text=True).strip():raise RuntimeError('GPU has another compute process')
