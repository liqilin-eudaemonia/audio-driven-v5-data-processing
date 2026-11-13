import copy,json
from pathlib import Path
from unittest.mock import patch
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from p2_guided_chain.data import training_record,finish_view,publish_task
from p2_guided_chain.pool import old_reservations,worker_command
from p2_guided_chain.director import process_semantic
from p2_guided_chain.prepare import marker
from p2_guided_chain.training_queue import commands,advance
from p2_listen.contracts import atomic_json,digest,read
from p2_plan_cache.protocol import render_clean
from p2_plan_cache.runtime import identity
from listen_benchmark.semantic_prompt import materialize
from test_semantic_prompt import fixture


def published():
 r,a=fixture();return materialize(r,a,dict(accepted=True,issues=[]),dict(protocol_sha256='test'))


def test_audio_verbatim_and_prior_template():
 result=published();original=copy.deepcopy(result);record=training_record(result);text=render_clean(record)
 assert result==original
 assert [e for e in record['events'] if e['modality']=='audio']==[e for e in result['source_record']['events'] if e['modality']=='audio']
 assert 'Visual response: Smiles broadly.' in text['prompt']
 assert '[3–3.2 seconds]' in text['prompt'] and '[3.3–3.5 seconds]' in text['prompt']
 assert text['policy']['template']=='existing_english_guidance'


@pytest.mark.parametrize('change',[lambda r:r['semantic_review'].update(accepted=False),lambda r:r['events'][-1].update(text='invented')])
def test_rejected_or_changed_audio_never_accepted(change):
 r=published();change(r)
 with pytest.raises(ValueError):training_record(r)


def test_chunk_and_merged_cache_identity_same():
 item=dict(guidance=render_clean(training_record(published())))
 p=dict(code={'x':'same'},plan_source='postprocessed_train_annotation',max_text_tokens=8192,max_qwen_tokens=32768,sample_ids=['a'],sources={'chunk':'sha'})
 merged=dict(p,sample_ids=['a','b'],sources={'chunk':'sha','other':'sha'})
 assert identity(p,item,'text',{})==identity(merged,item,'text',{})
 with patch('p2_plan_cache.runtime.input_identity',return_value={'input':'same'}):
  item['observation']={}
  assert identity(p,item,'input_plan',{})==identity(merged,item,'input_plan',{})
 item2=copy.deepcopy(item);item2['guidance']['prompt']+=' changed'
 assert identity(p,item,'text',{})!=identity(p,item2,'text',{})


def test_legacy_recovery_has_priority(tmp_path):
 atomic_json(tmp_path/'parallel_queue/jobs.json',{'a':dict(status='running',slots=[0]),'b':dict(status='running',slots=[1],expanding_to=[1,2]),'c':dict(status='complete',slots=[3])})
 active,priority=old_reservations(tmp_path)
 assert set(active)==set(range(6)) and priority=={2,3,4,5}


def test_worker_uses_unchanged_native_entrypoint(tmp_path):
 atomic_json(tmp_path/'pipeline.json',dict(python='trainpy',vllm_python='vpy',semantic_model='model'))
 task=dict(id='chunk',kind='semantic',inputs='input',output='out')
 cmd=worker_command(task,'host',3,True,1,tmp_path)
 assert cmd[cmd.index('--memory')+1]=='0.46' and cmd[cmd.index('--attempts')+1]=='3'
 assert any(x.endswith('/run_listener_semantic_prompt.py') for x in cmd)
 assert 'CUDA_VISIBLE_DEVICES=3' in cmd


def test_partial_chunk_only_retries_unpublished(tmp_path):
 task=dict(id='x',kind='semantic',inputs=str(tmp_path/'chunks/x/inputs.jsonl'),output='run');folder=Path(task['inputs']).parent;folder.mkdir(parents=True)
 jobs=[dict(job_id=str(i),record={'sample_id':str(i)}) for i in range(3)]
 Path(task['inputs']).write_text(''.join(json.dumps(j)+'\n' for j in jobs))
 export=dict(accepted=['0'],rejected=[],unresolved_ids=['1','2'],work='plans')
 atomic_json(folder/'export.json',export)
 atomic_json(tmp_path/'pipeline.json',dict(semantic_model='model'))
 with patch('p2_guided_chain.partition.groups',return_value=[[jobs[1]],[jobs[2]]]):
  process_semantic(tmp_path,task,dict(status='data_error',error='Context overflow; no truncation allowed'))
 children=[read(p) for p in (tmp_path/'tasks').glob('*.json')]
 assert len(children)==4
 retried=[json.loads(line)['job_id'] for t in children if t['kind']=='semantic' for line in Path(t['inputs']).read_text().splitlines()]
 assert sorted(retried)==['1','2']
 process_semantic(tmp_path,task,dict(status='data_error',error='same'))
 assert len(list((tmp_path/'tasks').glob('*.json')))==4


def test_system_failure_cannot_filter_members(tmp_path):
 task=dict(id='x',kind='semantic',inputs=str(tmp_path/'chunks/x/inputs.jsonl'),output='run');folder=Path(task['inputs']).parent;folder.mkdir(parents=True)
 Path(task['inputs']).write_text(json.dumps(dict(job_id='s',record={'sample_id':'s'}))+'\n')
 atomic_json(folder/'export.json',dict(accepted=[],rejected=[],unresolved_ids=['s'],work='plans'))
 with pytest.raises(ValueError,match='omitted'):process_semantic(tmp_path,task,dict(status='succeeded'))
 assert not (tmp_path/'context_rejections').exists()


def test_marker_requires_matching_contract_and_payload(tmp_path):
 expected={'actual':'contract'};atomic_json(tmp_path/'complete.json',dict(identity=digest(expected),contract=expected,payload_sha256='x'))
 with pytest.raises(FileNotFoundError):marker(tmp_path,expected)
 (tmp_path/'payload.pt').write_bytes(b'x');marker(tmp_path,expected)
 with pytest.raises(ValueError):marker(tmp_path,{'actual':'changed'})


def test_distributed_stage2_has_parent_only_without_resume():
 cluster=dict(nodes=[dict(host='master'),dict(host='worker'),dict(host='third')],network={},python='python')
 _,cmd=commands(cluster,[0],'config',None,'parent',32000)
 assert all('--parent' in c and '--resume' not in c for c in cmd)
 assert '--nproc_per_node=2' in cmd[0]
 _,cmd=commands(cluster,[0,1],'config','resume','parent',32000)
 assert all('--resume' in c and '--parent' not in c and '--nproc_per_node=4' in c for c in cmd)
 with pytest.raises(ValueError):commands(cluster,[0,1,2],'config',None,None,32000)


def test_stage1_advances_to_paired_stage2(tmp_path):
 path=tmp_path/'config.json';atomic_json(path,dict(training={'steps':10}))
 j=dict(config=str(path),stage=1,slots=[0],status='running')
 with patch('p2_matrix16.checkpoint.checked',return_value={'step':10}),patch('p2_guided_chain.training_queue.prepare_run',return_value='s2_config') as prep:
  advance(tmp_path,'av_event',j,'final.pt')
  prep.assert_called_once_with(tmp_path,'av_event',2,[0])
 assert j['stage']==2 and j['status']=='pending' and j['parent']=='final.pt'
 assert j['completed']=={'1':'final.pt'}


def test_incomplete_parent_cannot_advance(tmp_path):
 path=tmp_path/'config.json';atomic_json(path,dict(training={'steps':10}));j=dict(config=str(path),stage=1,slots=[0])
 with patch('p2_matrix16.checkpoint.checked',return_value={'step':9}):
  with pytest.raises(ValueError):advance(tmp_path,'av',j,'unfinished.pt')


def test_topology_migration_preserves_stage1_parent_and_optimizer(tmp_path):
 import torch
 from p2_matrix16 import VERSION
 from p2_listen.cache import atomic_torch
 from p2_listen.contracts import sha
 from p2_guided_chain.training_queue import migration
 source=tmp_path/'old/step_000005.pt';old=dict(output=str(source.parent),training={'global_batch':24});new=dict(old,output=str(tmp_path/'new'))
 saved=dict(schema=VERSION,step=5,model={'weight':torch.tensor([1.,2.])},optimizer={'state':{'exp_avg':torch.tensor([3.,4.])}},rank_rng=[{'rank':i} for i in range(6)],identity=dict(experiment=old,world=6,parent=None,source={'code':'same'},data_identity={'data':'same'}))
 atomic_torch(source,saved);atomic_json(source.with_suffix('.complete.json'),dict(sha256=sha(source),step=5))
 with patch('p2_guided_chain.backend.source_identity',return_value={'code':'same'}),patch('p2_guided_chain.backend.data_identity',return_value={'data':'same'}):
  dest=migration(source,new,12)
 loaded=torch.load(dest,weights_only=True)
 assert loaded['identity']['world']==12 and loaded['identity']['parent'] is None
 assert torch.equal(loaded['model']['weight'],saved['model']['weight'])
 assert torch.equal(loaded['optimizer']['state']['exp_avg'],saved['optimizer']['state']['exp_avg'])
 assert loaded['rank_rng']==saved['rank_rng']*2
 assert read(dest.parent/'topology_migration.json')['model_and_optimizer_tensors_unmodified']


def test_all_runs_share_exact_filtered_members(tmp_path):
 from p2_guided_chain.data import finish_view
 from p2_plan_cache import VERSION
 data={'output':str(tmp_path/'original')}
 rows=[dict(sample_id=s,requested_duration_s=2.) for s in ('a','b','c')]
 atomic_json(tmp_path/'original/manifests/observations.json',rows)
 atomic_json(tmp_path/'pipeline.json',dict(data_config='cfg',policy='strict'))
 atomic_json(tmp_path/'PREPARED.json',dict(samples=3))
 work=tmp_path/'chunks/x/plans';p=dict(sources={'source':'sha'},entries={'a':'entry-a','c':'entry-c'})
 atomic_json(work/'protocol.json',p)
 atomic_json(work.parent/'export.json',dict(accepted=['a','c'],rejected=[dict(sample_id='b',error='rejected')],work=str(work)))
 with patch('p2_listen.config.load',return_value=data):result=finish_view(tmp_path)
 assert result['samples']==2 and result['rejected']==1 and result['steps_per_stage']==1
 assert [r['sample_id'] for r in read(tmp_path/'view/observations.json')]==['a','c']
 assert read(tmp_path/'plans/protocol.json')['sample_ids']==['a','c']
 assert read(tmp_path/'view/rejected.json')[0]['sample_id']=='b'
 # A missing outcome can never freeze a smaller set silently.
 atomic_json(work.parent/'export.json',dict(accepted=['a','c'],rejected=[],work=str(work)))
 with patch('p2_listen.config.load',return_value=data),pytest.raises(ValueError,match='coverage'):finish_view(tmp_path)


def test_context_probe_partitions_without_rejecting_or_rewriting():
 from p2_guided_chain.partition import groups
 jobs=[{'job_id':str(i)} for i in range(5)]
 before=copy.deepcopy(jobs)
 with patch('p2_guided_chain.partition.initial_overflow',side_effect=lambda j,m:j['job_id'] in ('1','4')):
  result=groups(jobs,'model')
 assert [[j['job_id'] for j in g] for g in result]==[['0','2','3'],['1'],['4']]
 assert jobs==before and sorted(j['job_id'] for g in result for j in g)==['0','1','2','3','4']


def test_stage2_requires_matching_final_parent_and_zero_lora(tmp_path):
 import torch
 from types import SimpleNamespace
 from p2_guided_chain.backend import initialize_stage2
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.connector=torch.nn.Linear(2,2);adapter=torch.nn.Module();adapter.b=torch.nn.Linear(2,2,bias=False);torch.nn.init.zeros_(adapter.b.weight);self.qwen_lora=torch.nn.ModuleList([adapter])
 model=Model();c=dict(stage=2,parent_experiment='guided_av_s1',text_condition='guided',loss='av',loss_weights={},text_lora={},data_config='data',plan_cache='plan',teacher_cache='teacher',guided_view='view')
 pc=dict(c,stage=1,id='guided_av_s1',training={'steps':10})
 saved=dict(step=10,identity=dict(experiment=pc,source={'same':True},data_identity={'same':True}),model={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('qwen_lora.')})
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 with patch('p2_matrix16.checkpoint.checked',return_value=saved),patch('p2_guided_chain.backend.source_identity',return_value={'same':True}):
  meta=initialize_stage2(parent,model,c,{'same':True});assert meta['step']==10
  saved['step']=9
  with pytest.raises(ValueError,match='full budget'):initialize_stage2(parent,model,c,{'same':True})
  saved['step']=10
  with pytest.raises(ValueError,match='identity'):initialize_stage2(parent,model,c,{'different':True})


def test_director_preloads_package_before_legacy_script_path():
 import subprocess,sys
 root=Path(__file__).resolve().parents[1]
 code="import sys;sys.path.insert(0,"+repr(str(root/'src'))+");import p2_guided_chain.director;sys.path.insert(0,"+repr(str(root/'scripts'))+");import p2_guided_chain.canary;import p2_matrix16.config;assert hasattr(sys.modules['p2_matrix16'],'__path__')"
 result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
 assert result.returncode==0,result.stderr


def test_replay_conflict_reprocesses_singleton_without_rejection(tmp_path):
 task=dict(id='x',kind='semantic',inputs=str(tmp_path/'chunks/x/inputs.jsonl'),output='run');folder=Path(task['inputs']).parent;folder.mkdir(parents=True)
 job=dict(job_id='remaining',record={'sample_id':'remaining'});Path(task['inputs']).write_text(json.dumps(job)+'\n')
 atomic_json(folder/'export.json',dict(accepted=['published'],rejected=[],unresolved_ids=['remaining'],work='plans'))
 process_semantic(tmp_path,task,dict(status='replay_conflict',error='Cached response identity/request mismatch'))
 child=read(tmp_path/'tasks/xp000.json');assert json.loads(Path(child['inputs']).read_text())==job
 assert not (tmp_path/'context_rejections').exists()
 assert (tmp_path/'tasks/x_text.json').exists() and (tmp_path/'tasks/x_input_plan.json').exists()


def test_gpu_smoke_sets_cublas_before_cuda_access():
 import os,subprocess,sys
 root=Path(__file__).resolve().parents[1];env=dict(os.environ);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
 code='import runpy,os;runpy.run_path('+repr(str(root/'scripts/p2_guided_chain_s0.py'))+',run_name="smoke_import");assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"'
 result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env)
 assert result.returncode==0,result.stderr
