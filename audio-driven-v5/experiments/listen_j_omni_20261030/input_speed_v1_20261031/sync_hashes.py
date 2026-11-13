from pathlib import Path
import sys,hashlib
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked
with locked(W/'config-edit.lock'):
 c=read(W/'config.json')
 for rel in [*c['code_hashes'],'src/listen_benchmark/shared_visual_decode.py','src/listen_benchmark/shared_raw_decode.py','src/listen_benchmark/frame_crop.py','src/listen_benchmark/j_omni_stream.py','src/listen_benchmark/j_omni_input_benchmark.py']:
  c['code_hashes'][rel]=hashlib.sha256((R/rel).read_bytes()).hexdigest()
 write(W/'config.json',c)
