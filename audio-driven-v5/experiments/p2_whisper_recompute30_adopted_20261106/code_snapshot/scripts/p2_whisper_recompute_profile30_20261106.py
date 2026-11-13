"""Same30-rank sequential queue, Event three_quarters; Feature remains all."""
import sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_recompute_profile_20261106 import install,BASE
install()
spec=importlib.util.spec_from_file_location('profile30_cli',ROOT/'scripts/p2_whisper_continue30.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
cli.SELF=Path(__file__).resolve();cli.BASE=BASE
if __name__=='__main__':cli.main()
