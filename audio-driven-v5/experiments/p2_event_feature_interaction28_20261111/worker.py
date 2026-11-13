"""Use the native node/rank worker with the two-candidate configuration extension."""
from pathlib import Path
from extension import activate, ROOT

activate()
original = ROOT / 'src/p2_loss_search28/worker.py'
exec(compile(original.read_text(), str(original), 'exec'), globals())

