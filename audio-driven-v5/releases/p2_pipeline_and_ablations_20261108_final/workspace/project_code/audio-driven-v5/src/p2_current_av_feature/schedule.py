"""Deterministic one-epoch duration groups for the final train+dev sample order."""
from pathlib import Path
import csv,math
from p2_listen.contracts import read,sha
from p2_j_duration_execution import make_pairs,Schedule as Native


class Schedule(Native):
 def __init__(self,c):
  spec=c['duration_sampling'];t=c['training'];path=Path(c['sample_list'])
  if sha(path)!=spec['sample_list_sha256'] or sha(spec['manifest'])!=spec['sha256']:raise ValueError('Schedule changed')
  with path.open()as f:rows=list(csv.DictReader(f))
  self.size=len(rows);self.batch=t['global_batch'];self.seed=t['seed']
  if t['steps']!=math.ceil(self.size/self.batch):raise ValueError('Budget must cover exactly one epoch plus tail padding')
  protocol=read(Path(c['j_cache'])/'protocol.json')
  if [r['sample_id']for r in rows]!=protocol['sample_ids']:raise ValueError('Current schedule sample order differs')
  self.durations=[float(r['duration_s'])for r in rows];plan=read(spec['manifest'])
  expected=make_pairs([dict(requested_duration_s=v)for v in self.durations],t['steps'],self.batch,self.seed,
   spec['start_step'],spec['policy'],spec['window'])
  self.pairs=[tuple(p)for p in plan['pairs']]
  if self.pairs!=expected or plan['sample_list_sha256']!=spec['sample_list_sha256']:raise ValueError('Non-deterministic current AV schedule')
