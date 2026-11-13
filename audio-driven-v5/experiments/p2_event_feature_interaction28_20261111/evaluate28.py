"""Native accelerated inference with the interaction training identity installed."""
from pathlib import Path
from extension import activate, LEGACY

activate()
import eval_runtime
original = LEGACY / 'evaluate28.py'
exec(compile(original.read_text(), str(original), 'exec'), globals())

