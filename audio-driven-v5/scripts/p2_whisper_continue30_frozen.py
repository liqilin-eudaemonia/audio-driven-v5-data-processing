"""Transport entry for pinned-source30-rank resume; original training remains unchanged."""
import sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_frozen_resume_20261106 import install
install()
spec=importlib.util.spec_from_file_location('native30_cli',ROOT/'scripts/p2_whisper_continue30.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
cli.SELF=Path(__file__).resolve()
if __name__=='__main__':cli.main()
