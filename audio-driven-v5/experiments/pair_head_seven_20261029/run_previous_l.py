"""Fill the missing two-source L baseline with the same inputs and QC cache."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import importlib.util
from pathlib import Path
EXP=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('seven_run',EXP/'run.py')
job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
import listen_local_recovery_localbc as previous
job.repair=previous;job.run.repair=previous
job.run.PAIRS=job.PAIRS[:2];job.run.OUT=EXP/'previous_l'
job.run.EXTRA_CACHE_DIRS.append(EXP/'results')
job.run.main()
