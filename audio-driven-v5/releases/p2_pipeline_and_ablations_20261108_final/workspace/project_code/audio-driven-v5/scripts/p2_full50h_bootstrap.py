#!/usr/bin/env python3
from pathlib import Path
import sys,subprocess,time,fcntl,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.append(str(ROOT/'scripts'))
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import transport


def run(root):
 c=read(root/'cluster.json');spec=read(root/'pipeline.json')
 with (root/'bootstrap.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  while not (root/'RECOVERY_COMPLETE.json').exists():
   with (root/'recovery_build.log').open('a') as log:
    result=subprocess.run([c['python'],str(ROOT/'scripts/p2_restore_full50h.py'),'--old',spec['prior_reduced_run'],'--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    atomic_json(root/'bootstrap_error.json',dict(at=time.time(),returncode=result.returncode,log=str(root/'recovery_build.log')));time.sleep(30)
  records=[]
  for n in c['nodes']:
   cmd=[c['python'],str(ROOT/'scripts/p2_guided_chain_director.py'),'--root',str(root)];log=str(root/'pool'/('director_'+n['name']+'.log'));code='import subprocess;f=open('+repr(log)+',"a");p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
   x=subprocess.run(transport(c,n,[c['python'],'-c',code]),capture_output=True,text=True,check=True);records.append(dict(node=n['name'],pid=int(x.stdout.strip().splitlines()[-1])))
  atomic_json(root/'director/controllers.json',records);atomic_json(root/'BOOTSTRAP_COMPLETE.json',dict(at=time.time(),recovery_sha256=sha(root/'RECOVERY_COMPLETE.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();run(a.root.resolve())
