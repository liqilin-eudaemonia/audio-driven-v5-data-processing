"""Explicit profile entrypoint for same-loss LTX-hidden Event + Feature training."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_ltx_recompute_profile_20261106 import install
install()
if __name__=='__main__':runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='__main__')
