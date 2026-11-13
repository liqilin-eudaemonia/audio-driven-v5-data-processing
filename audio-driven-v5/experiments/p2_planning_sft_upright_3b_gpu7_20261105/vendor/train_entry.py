import argparse
from p2_planning_sft.train import train
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--stop-after',type=int);a=p.parse_args()
train(a.config,a.resume,a.smoke,a.output,a.stop_after)
