"""CPU migration regression: no inference/data/sampling changes on resource expansion."""
import importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
spec=importlib.util.spec_from_file_location('expand32',ROOT/'scripts/p2_responsenet_expand32.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from p2_testsets_eval.protocol import save,read,partition


def test_safe_migration_retains_fixed200_weights_and_scores(tmp_path):
 items=[dict(key='responsenet/'+str(i),id=str(i),geometry=dict(native_frames=8*i+1,width=512,height=512)) for i in range(200)]
 p=dict(workers=28,datasets=dict(responsenet=dict(count=200,references_sha256='fixed')),
  selection_sha256='fixed-selection',generation=dict(steps=30,video_cfg=3,audio_cfg=7),seed=137,
  arms=[dict(name='fixed',checkpoint='unchanged.pt')],total_samples=200)
 save(tmp_path/'protocol.json',p);save(tmp_path/'groups.json',[])
 save(tmp_path/'CONTROLLER.json',dict(pid=99999999))
 save(tmp_path/'STOP',dict(reason='owned coordinator safe-boundary expansion from28 to32 GPUs'))
 save(tmp_path/'cluster.json',dict(nodes=[dict(name='node8',host='172.29.104.237',gpus=[0,1,2,3])]))
 for i,s in enumerate(partition(items,28)):save(tmp_path/'shards'/f'{i:02d}.json',s)
 old=tmp_path/'metric_shards/old.json';save(old,dict(value=123))
 class Process:pid=42
 with patch('p2_seamless200_whisper_eval.idle_snapshot',return_value=dict(all_idle=True)),patch.object(m.subprocess,'Popen',return_value=Process()):
  assert m.migrate(tmp_path)==42
 new=read(tmp_path/'protocol.json')
 for key in ['datasets','selection_sha256','generation','seed','arms']:assert new[key]==p[key]
 assert new['workers']==32 and new['shared_item_queue']
 assert read(tmp_path/'cluster.json')['nodes'][0]['gpus']==list(range(8))
 assert not (tmp_path/'STOP').exists()
 assert read(tmp_path/'expansion32/original28/metric_shards/old.json')['value']==123
 selected=[x['key'] for i in range(32) for x in read(tmp_path/'shards'/f'{i:02d}.json')]
 assert len(selected)==len(set(selected))==200
