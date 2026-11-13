import sys,subprocess,shlex,concurrent.futures,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_audio_full.cluster import HOSTS,KEY,TRAIN_QUEUE,PYTHON
(OUT/'logs').mkdir(exist_ok=True)
def launch(rank):
 cmd=[str(PYTHON),str(OUT/'node.py'),'detach',str(rank)]
 if rank:cmd=['ssh','-i',KEY,'-o','UserKnownHostsFile='+str(TRAIN_QUEUE/'queue_known_hosts'),'-o','StrictHostKeyChecking=yes','root@'+HOSTS[rank],shlex.join(cmd)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;return p.stdout
for v in concurrent.futures.ThreadPoolExecutor().map(launch,range(3)):print(v)
for name in ['gt_detect','evaluate','finish_scores']:
 with (OUT/f'{name}.log').open('a') as log:p=subprocess.Popen([str(PYTHON),str(OUT/f'{name}.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 print(name,p.pid)
