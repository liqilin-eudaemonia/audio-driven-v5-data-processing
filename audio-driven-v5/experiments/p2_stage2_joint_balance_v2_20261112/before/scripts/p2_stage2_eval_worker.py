"""One explicit evaluation phase in an isolated process; never starts training."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_stage2_joint.config import validate
from p2_listen.contracts import read

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--step',type=int,choices=[0,100,300],required=True)
 p.add_argument('--phase',choices=['qwen','gemma','whisper','render','decode','references','plan-evidence'],required=True);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();c=validate(read(a.config))
 # The generator reuse path activates the original native helpers.
 from p2_stage2_joint.data import data_config
 from p2_listen.config import extractor_config
 extractor_config(data_config(c['generator']))
 if a.phase=='plan-evidence':
  from p2_stage2_joint.plan_evidence import extract
  extract(c,a.step)
 else:
  from p2_stage2_joint.eval_worker import PHASES
  PHASES[a.phase](c,a.step,a.device)

if __name__=='__main__':main()
