from .config import normalized
from p2_ltx_all_lora.model import Model as Native
class Model(Native):
 def __init__(self,c,cfg,renderer):
  # Loss multipliers are applied by the executor objective using actual c.
  super().__init__(normalized(c),cfg,renderer)
