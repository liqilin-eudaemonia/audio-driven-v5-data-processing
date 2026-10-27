import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import control as c
@pytest.fixture
def pair(monkeypatch):
 original=c.read(c.E1/'run.json');second=copy.deepcopy(original);second['arm']='E0'
 second['config']['output']=str(c.TRAIN);second['source']=c.approved_source(original['source'])
 monkeypatch.setattr(c,'read',lambda p:original if p==c.E1/'run.json' else second)
 return original,second

def test_approved_preflight_only_pair(pair):
 assert c.verify_pair()['status']=='passed'

@pytest.mark.parametrize('key',['initialization','observations','targets','labels','formal_statistics','seed','budget','world','source'])
def test_pair_rejects_other_changes(pair,key):
 pair[1][key]='unexpected'
 with pytest.raises(ValueError):c.verify_pair()

def test_pair_rejects_configuration_change(pair):
 pair[1]['config']['training']['connector_lr']*=2
 with pytest.raises(ValueError):c.verify_pair()

@pytest.mark.parametrize('memory,apps,expected',[(1,'',True),(526,'123, unrelated-job, 526 MiB',False),(2000,'',False)])
def test_gpu_admission_rejects_live_context_even_with_low_memory(monkeypatch,memory,apps,expected):
 import run_node
 def query(argv,**kwargs):
  if any('--query-gpu=' in v for v in argv):return '\n'.join(f'{i}, {memory}' for i in range(8))
  return apps
 monkeypatch.setattr(run_node.subprocess,'check_output',query)
 assert run_node.gpu_state()['idle']==expected
