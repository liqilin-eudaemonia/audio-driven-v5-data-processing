import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
from p2_testsets_eval.protocol import DEFAULT,prepare,read

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','queue','bind','merge','metrics','metrics_queue'])
 p.add_argument('--work',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['cache','render','decode'])
 p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--arms',nargs='+');p.add_argument('--metric-smoke',action='store_true')
 a=p.parse_args()
 if a.action=='prepare':print(prepare(a.work));return
 if a.action=='worker':
  from p2_testsets_eval.runtime import worker
  worker(a.work,a.stage,a.rank,a.limit,a.arms)
 elif a.action=='metrics':
  from p2_testsets_eval.metrics import metrics_worker
  metrics_worker(a.work,a.rank,a.arms,a.metric_smoke)
 elif a.action=='metrics_queue':
  from p2_testsets_eval.metrics import queue
  queue(a.work)
 else:
  from p2_testsets_eval import cluster
  getattr(cluster,a.action)(a.work)
if __name__=='__main__':main()
