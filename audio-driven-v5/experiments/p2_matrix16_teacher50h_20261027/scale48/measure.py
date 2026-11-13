from pathlib import Path
import os,json,time,sys,collections
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');folder=Path(__file__).parent;begin=time.time()
obs=json.loads((root/'experiments/p2_train50h_postprocessed_reselected_20261027/manifests/observations.json').read_text());ids=sorted(o['sample_id'] for o in obs);rank={sid:i%24 for i,sid in enumerate(ids)};counts=[0]*24
for item in os.scandir(folder.parent/'cache'):
 if item.is_dir() and os.path.isfile(item.path+'/complete.json'):counts[rank[item.name]]+=1
value=dict(at=(begin+time.time())/2,scan_seconds=time.time()-begin,total=sum(counts),counts24=counts)
(folder/(sys.argv[1]+'.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
print('states',dict(collections.Counter(json.loads(p.read_text())['state'] for p in folder.glob('state-*.json'))))
