"""Exercise the file-path worker entry used by real SSH launchers."""
import subprocess
import sys
from pathlib import Path


def test_worker_file_entry_does_not_shadow_standard_queue():
    root=Path(__file__).resolve().parents[1]
    worker=root/'src/p2_loss_search28/worker.py'
    # Reproduce Python's sys.path[0] for a script, rather than importing the
    # worker only as a package (which did not expose the shadowing regression).
    program="""import sys,runpy,pathlib
worker=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(worker.parent))
runpy.run_path(str(worker),run_name='entrypoint_import_check')
import torch,queue
assert pathlib.Path(queue.__file__).name=='queue.py'
assert pathlib.Path(queue.__file__).parent!=worker.parent
from p2_loss_search28.config import validate_cluster
print('file entry torch/queue/config imports accepted')
"""
    subprocess.run([sys.executable,'-c',program,str(worker)],cwd=root,check=True,timeout=60)
