"""Unchanged native Whisper branches/forward; zero inactive loss coefficients."""
from p2_whisper_av.model import QwenWhisperAV
from p2_j_av_only import av_objective
from .config import native_view

class Model(QwenWhisperAV):
 def __init__(self,c,cfg,renderer):
  super().__init__(native_view(c),cfg,renderer)
  self.spec=c
  self.assert_ownership()
 def forward(self,bundle,strategy,sampler):
  if any(k in bundle for k in ('events','hidden_event_layout')):raise ValueError('Event labels forbidden')
  return super().forward(bundle,strategy,sampler)
