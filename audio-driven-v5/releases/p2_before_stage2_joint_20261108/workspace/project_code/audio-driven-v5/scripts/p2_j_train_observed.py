"""Unchanged duration trainer, with signal-triggered Python stack diagnostics."""
import faulthandler
from pathlib import Path
import runpy
import signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
runpy.run_path(str(Path(__file__).with_name('p2_j_duration_train.py')), run_name='__main__')
