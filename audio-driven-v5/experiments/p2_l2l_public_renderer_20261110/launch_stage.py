"""Launch one owned integration stage on the already allocated node11 GPU7."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from listen_cluster import transport

stage=sys.argv[1]
assert stage in ['cuda_probe','track_smoke','smoke','pilot','track','generate','controller']
cfg=json.loads((ROOT/'experiments/p2_current500h_av_only31_20261107/cluster.json').read_text())
node=next(n for n in cfg['nodes'] if n['name']=='node11')
script=ROOT/'scripts/p2_l2l_gpavatar.py'
args=[stage]
if stage=='cuda_probe':script=WORK/'cuda_probe.py';args=[]
if stage=='pilot':script=WORK/'pilot_video.py';args=[]
if stage=='controller':script=WORK/'controller.py';args=[]
command=['env','CUDA_VISIBLE_DEVICES=7','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',
    'MAX_JOBS=2','TORCH_CUDA_ARCH_LIST=9.0','TORCH_EXTENSIONS_DIR='+str(WORK/'torch_extensions'),
    'TOKENIZERS_PARALLELISM=false','PYTHONUNBUFFERED=1',
    'PATH=/private/software/conda/envs/lm-listener/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
    str(ROOT/'compare/.envs/vico_more/bin/python'),str(script),*args]
runner=('import subprocess,json,time,fcntl; from pathlib import Path; '
    f'w=Path({str(WORK)!r}); stage={stage!r}; lock=open(w/(stage+".launch.lock"),"a"); '
    'fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); '
    f'p=subprocess.Popen({command!r},cwd={str(ROOT)!r},stdout=open(w/(stage+".log"),"ab"),stderr=subprocess.STDOUT); '
    '(w/(stage+"_launch.json")).write_text(json.dumps(dict(pid=p.pid,at=time.time()))); '
    'code=p.wait(); (w/(stage+"_exit.json")).write_text(json.dumps(dict(returncode=code,at=time.time())))')
launcher=('import subprocess; '
    f'p=subprocess.Popen(["python","-c",{runner!r}],cwd={str(ROOT)!r},stdin=subprocess.DEVNULL,'
    f'stdout=open({str(WORK/(stage+'_supervisor.log'))!r},"ab"),stderr=subprocess.STDOUT,start_new_session=True); print(p.pid,flush=True)')
try:
    result=subprocess.run(transport(cfg,node,['python','-c',launcher]),capture_output=True,text=True,timeout=30)
    print(result.returncode,result.stdout,result.stderr[-1000:])
except subprocess.TimeoutExpired:
    print('Launch observation timed out; inspect stage launch receipt before retrying')
