from pathlib import Path
import sys,fcntl,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extension import activate,WORK,ORIGINAL
activate()
from p2_loss_search28 import queue
from p2_listen.contracts import atomic_json
try:
    with (ORIGINAL/'queue.lock').open('a') as allocation:
        fcntl.flock(allocation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (WORK/'promotion_3000.json').exists()
        assert not (WORK/'promotion_6000.json').exists()
        queue.run()
except Exception as exc:
    atomic_json(WORK/'RESUME1500_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
    raise
