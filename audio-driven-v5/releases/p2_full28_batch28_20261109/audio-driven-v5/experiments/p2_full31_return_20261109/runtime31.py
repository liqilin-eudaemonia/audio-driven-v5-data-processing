"""Resume the running full-LoRA Event/Feature/Whisper model on31 ranks."""
from pathlib import Path
import sys,importlib.util
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime as old
from p2_listen.contracts import sha

def source():
 d=old.source_identity('full')
 for name in ['runtime31.py','worker.py','training.py','canary.py']:d[str(W/name)]=sha(W/name)
 return d

def load():
 spec=importlib.util.spec_from_file_location('p2_ltx_all_lora._return31',W/'training.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from p2_ltx_all_lora.schedule import Schedule
 m.Schedule=Schedule;m.source_identity=source
 spec=importlib.util.spec_from_file_location('return31_canary',W/'canary.py');check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
 m.split_canary=lambda **kw:check.longest_canary(training=m,arm='full',**kw)
 return m
