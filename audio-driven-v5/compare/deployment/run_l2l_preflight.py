import os
from pathlib import Path
import sys
import traceback
from l2l_reliable_core import backend,save
from l2l_reliable_data import prepare,load,DEFAULT_WORK,UPSTREAM
from l2l_reliable_preflight import run

if __name__=='__main__':
    try:
        backend();prepare(DEFAULT_WORK);data,stats=load(DEFAULT_WORK)
        run(UPSTREAM,DEFAULT_WORK,data,stats)
    except Exception:
        save(DEFAULT_WORK/'preflight/FAILED.json',dict(traceback=traceback.format_exc()))
        raise
