from pathlib import Path
import sys
w=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(w),str(w.parent/"p2_stage2_joint_full6000_trial28_20261112")]
import isolate_events
isolate_events.activate()
