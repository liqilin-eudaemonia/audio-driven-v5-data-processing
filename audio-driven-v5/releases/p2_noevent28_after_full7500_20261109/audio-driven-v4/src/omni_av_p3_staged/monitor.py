"""Detect stalled phases even when no worker has published its first heartbeat."""
from pathlib import Path
from .orchestration import read

ACTIVE = {'training','testing_24gpu','extracting_full_cache','sealing_cache','preparing_input_integrity'}


def progress_issue(root, state, history, now, timeout=1200):
    if state.get('status') not in ACTIVE:return None
    folder=Path(state.get('folder') or root)
    phase=(state['status'],str(folder),state.get('phase_started',state.get('time',now)))
    if history.get('phase')!=phase:
        history.clear();history.update(phase=phase,last_change=phase[2],signature=None)
    paths=(list(folder.glob('progress_rank*.json')) if state['status'] in ('training','testing_24gpu')
           else list((folder/'progress').glob('*.json')) + list(folder.glob('progress.json')))
    if state['status']=='sealing_cache':paths=list(folder.glob('seal_progress.json'))
    rows=[]
    for path in sorted(paths):
        value=read(path)
        # Only advancing work counts; rewriting a timestamp does not hide a stall.
        rows.append((path.name,value.get('step'),value.get('micro'),value.get('phase'),value.get('completed')))
    signature=tuple(rows)
    if rows and signature!=history['signature']:
        history['signature']=signature
        history['last_change']=min(now,max(path.stat().st_mtime for path in paths))
    if now-history['last_change']>=timeout:
        return dict(kind='no_training_progress_20min',status=state['status'],folder=str(folder),
                    seconds_without_progress=now-history['last_change'],heartbeat_files=len(paths))
    return None
