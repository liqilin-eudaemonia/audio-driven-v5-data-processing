"""Keep the explicitly authorized evaluation alive after transport failures."""
import os,time,subprocess,fcntl
from pathlib import Path
from eval_common import HERE,PY,read,write

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'

def main():
    work=Path(os.environ.get('P2_EVAL_CLUSTER_WORK',str(HERE/'cluster24')))
    record=Path(os.environ.get('P2_EVAL_CONTROLLER_RECORD',str(HERE/'cluster24_controller.json')))
    script=Path(os.environ.get('P2_EVAL_CONTROLLER_SCRIPT',str(HERE/'cluster_eval.py')))
    lock=(work/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restarts=0
    while not (work/'STOP').exists():
        if (work/'AVAILABLE_COMPLETE.json').exists():
            value=read(work/'AVAILABLE_COMPLETE.json')
            write(HERE/'eval_status.json',dict(phase='available_final_weights_complete',**value,at=time.time()))
            write(work/'watcher_status.json',dict(status='complete',restarts=restarts));return
        pid=read(record)['pid']
        if not alive(pid):
            bad=[p.name for p in (work/'states').glob('*.json') if read(p).get('failures',0)>=3]
            if bad or restarts>=3:
                write(work/'watcher_status.json',dict(status='failed_requires_review',failed_tasks=bad,restarts=restarts));return
            with (HERE/'cluster24_controller.log').open('a') as log:
                child=subprocess.Popen([str(PY),'-u',str(script),'controller'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            restarts+=1;write(record,dict(pid=child.pid,restarts=restarts))
        write(work/'watcher_status.json',dict(status='watching',restarts=restarts,at=time.time()))
        time.sleep(10)

if __name__=='__main__':main()
