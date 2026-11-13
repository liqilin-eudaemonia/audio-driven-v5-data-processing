from p2_listen.contracts import read,sha
from .config import validate
class Schedule:
 def __init__(self,c):
  validate(c);s=c['loss_search'];p=read(s['schedule'])
  if sha(s['schedule'])!=s['schedule_sha256']:raise ValueError('Schedule digest changed')
  self.pairs=[tuple(x)for x in p['pairs']];self.size=p['size'];self.seed=c['training']['seed'];self.batch=28
  if len(self.pairs)!=c['training']['steps']*28:raise ValueError('Schedule budget mismatch')
 def indices(self,size,step,batch,seed,rank=0,world=1):
  if(size,batch,seed)!=(self.size,28,self.seed)or world not in (1,28)or not 0<=rank<world or not 1<=step<=len(self.pairs)//28:raise ValueError('Search sampler geometry mismatch')
  return self.pairs[(step-1)*28:step*28][rank::world]
