"""CPU-only official CPBD, consuming only verified finished exports."""
from pathlib import Path
import json,time,concurrent.futures,sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p2_vico_official_metrics import execute
ROOT=Path(__file__).resolve().parents[1];w=ROOT/'experiments/p2_vico256_combined8000_20261106'
import pandas as pd
p=json.loads((w/'protocol.json').read_text());df=pd.read_csv(p['datasets']['vico']['annotation']);jobs={}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 while len(jobs)<43:
  state=json.loads((w/'status.json').read_text())
  if state['stage']=='failed':break
  for i,row in df.iterrows():
   if i not in jobs and (w/'conversion'/f'{row.uuid}.verified.json').exists():jobs[i]=pool.submit(execute,w,'cpbd',i)
  for f in jobs.values():
   if f.done():f.result()
  if len(jobs)<43:time.sleep(2)
 for f in jobs.values():f.result()
(w/'CPBD_OVERLAP_COMPLETE.json').write_text(json.dumps({'clips':len(jobs),'at':time.time()}))
