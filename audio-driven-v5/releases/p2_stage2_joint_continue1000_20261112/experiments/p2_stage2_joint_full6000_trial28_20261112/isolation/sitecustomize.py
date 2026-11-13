"""Scoped to this trial through its explicit child PYTHONPATH."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import isolate_events
isolate_events.activate()
