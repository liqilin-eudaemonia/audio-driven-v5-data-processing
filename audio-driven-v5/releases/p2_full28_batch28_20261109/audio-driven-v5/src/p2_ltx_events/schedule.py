"""Use the measured duration grouping with CURRENT train+dev membership."""
import csv
from pathlib import Path
from p2_listen.contracts import read,sha
from p2_j_duration_execution import Schedule as OriginalSchedule,make_pairs


class CurrentSchedule(OriginalSchedule):
    def __init__(self,c):
        spec=c['duration_sampling'];t=c['training'];root=Path(c['current_source']);path=root/'sample_list.csv'
        if sha(path)!=spec['sample_list_sha256'] or sha(spec['manifest'])!=spec['sha256']:
            raise ValueError('Current duration plan/source changed')
        with path.open() as stream:source=list(csv.DictReader(stream))
        if [r['sample_id'] for r in source]!=read(root/'manifest.json')['training_ids']:
            raise ValueError('Current train+dev duration order differs')
        rows=[dict(requested_duration_s=float(r['duration_s'])) for r in source]
        plan=read(spec['manifest']);self.pairs=[tuple(p) for p in plan['pairs']]
        if (self.pairs!=make_pairs(rows,t['steps'],t['global_batch'],t['seed'],spec['start_step'],spec['policy'],spec['window'])
                or plan['sample_list_sha256']!=spec['sample_list_sha256']):
            raise ValueError('Current duration plan is not deterministic')
        self.size=len(rows);self.batch=t['global_batch'];self.seed=t['seed']
        self.durations=[r['requested_duration_s'] for r in rows]


def Schedule(c):
    if c.get('resume_execution',{}).get('world')==24:
        from .resume24 import Resume24Schedule
        return Resume24Schedule(c)
    if c.get('resume_execution'):
        from .resume31 import CurrentResumeSchedule
        return CurrentResumeSchedule(c)
    return CurrentSchedule(c) if c.get('current_cache_index') else OriginalSchedule(c)
