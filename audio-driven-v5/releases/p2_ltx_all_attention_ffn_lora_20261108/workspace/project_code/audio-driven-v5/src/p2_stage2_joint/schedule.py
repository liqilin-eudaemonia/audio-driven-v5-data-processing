"""Reuse the exact parent31-card sample/noise plan; take its first300 updates."""
from p2_current_av_feature.schedule import Schedule as Parent

class Schedule(Parent):
 def __init__(self,c):
  super().__init__(c['generator'])
  self.pairs=self.pairs[:c['steps']*c['global_batch']]
