import copy,json,os
from pathlib import Path
import pytest,torch
from p2_listen.contracts import atomic_json,sha,digest
from p2_listen.cache import save_part
from listen_audio_full.indexing import array_rows
from listen_audio_full import preparation as prep,producer,worker,cluster

@pytest.mark.parametrize('chunk',[1,2,7,31,1024])
def test_bounded_json_array_unicode_and_chunk_edges(tmp_path,chunk):
 rows=[{'name':'你好, ] \\"','v':i,'nested':{'a':[1,2]}} for i in range(10)]
 p=tmp_path/'rows.json';p.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 assert list(array_rows(p,chunk))==rows

@pytest.mark.parametrize('value',['[{},]','[{}','[{}] false','{}'])
def test_array_parser_rejects_malformed_input(tmp_path,value):
 p=tmp_path/'bad.json';p.write_text(value)
 with pytest.raises(ValueError):list(array_rows(p,2))


def test_unlocalized_and_unmapped_are_preserved_separately(monkeypatch):
 visual=dict(event_id='v',modality='visual',description='原文微笑')
 a=dict(event_id='a',modality='audio',description='嗯',text='嗯',start=.1,end=.4)
 b=dict(event_id='b',modality='audio',description='哈',text='哈',start=.6,end=.8)
 missing=dict(event_id='missing',modality='audio',description='无法定位原文',text='你好',start=None,end=None)
 label=dict(events=[visual,a,b,missing],raw_issue_count=3,semantic_issue_count=1);original=copy.deepcopy(label)
 canonical={'events':[{'type':'listener_vocalization','source':{'event_id':'a'}}]}
 def derive(job,raw,provenance):
  assert job['events']==[a,b]
  return {'audio_events':copy.deepcopy(job['events']),'provenance':{}},{'excluded_audio_events':[]}
 monkeypatch.setattr(prep,'derive',derive)
 obs={'sample_id':'s','requested_duration_s':1};target={'target_audio':'test','media_hashes':{'target_audio':'a'*64}}
 record,audit,opened=prep.derive_sample(label,canonical,obs,target,0,{}, {},'b'*64,'c'*64)
 assert label==original
 assert record['audio_events']==[a]
 assert opened['events']==[visual,a,b]
 assert audit['unresolved_audio_events'][0]['original_event']==missing
 assert audit['canonical_excluded_unmapped'][0]['original_event']==b
 assert audit['raw_issue_count']==3 and audit['semantic_issue_count']==1


def test_queue_append_restart_and_torn_tail(tmp_path):
 task={'task_id':'a','rank':0,'samples':3};atomic_json(tmp_path/'ready/a.json',task)
 published={};assert producer.publish_ready(tmp_path,published,2)==1
 assert producer.publish_ready(tmp_path,published,2)==0
 p=tmp_path/'queues/000.jsonl'
 with p.open('ab') as f:f.write(b'{"incomplete":')
 with pytest.raises(ValueError):producer.queued_tasks(tmp_path,2)
 observed,queues,total=producer.queued_tasks(tmp_path,2,repair_tail=True)
 assert total==3 and observed==published and queues['1']['tasks']==0
 assert producer.publish_ready(tmp_path,observed,2)==0


def test_cache_corruption_quarantined_but_identity_mismatch_rejected(tmp_path):
 folder=tmp_path/'view/cache/audio_postprocess/s';identity={'test':'identity'};save_part(folder,identity,{'a':torch.ones(2)})
 (folder/'payload.pt').write_bytes(b'corrupt')
 worker.quarantine_corrupt(tmp_path,folder,identity)
 assert not folder.exists() and len(list((tmp_path/'quarantine').iterdir()))==1
 save_part(folder,identity,{'a':torch.ones(2)})
 with pytest.raises(ValueError,match='identity'):worker.quarantine_corrupt(tmp_path,folder,{'test':'different'})
 assert folder.exists()


def test_predecessor_requires_finished_training_and_verified_checkpoint(tmp_path,monkeypatch):
 monkeypatch.setattr(cluster,'TRAIN_QUEUE',tmp_path/'training');monkeypatch.setattr(cluster,'PREDECESSOR',tmp_path/'E1')
 atomic_json(tmp_path/'training/queue/state.json',{'stage':'E1','phase':'training'})
 atomic_json(tmp_path/'E1/status.json',{'step':1268,'stopped':False})
 p=tmp_path/'E1/step_001268.pt';p.write_bytes(b'checkpoint');atomic_json(p.with_suffix('.complete.json'),{'step':1268,'sha256':sha(p)})
 assert cluster.predecessor_ready() is None
 atomic_json(tmp_path/'training/queue/state.json',{'stage':'E1','phase':'complete'})
 assert cluster.predecessor_ready()['step']==1268
 p.write_bytes(b'broken')
 with pytest.raises(ValueError):cluster.predecessor_ready()


def fake_ready(run):
 atomic_json(run/'scope.json',{'samples':722706});atomic_json(run/'view/config.json',{});atomic_json(run/'producer_complete.json',{})


def test_no_gpu_launch_while_e1_running(tmp_path,monkeypatch):
 fake_ready(tmp_path);monkeypatch.setattr(cluster,'verify_code',lambda r:None);monkeypatch.setattr(cluster,'predecessor_ready',lambda:None)
 calls=[]
 def call(rank,args):
  calls.append(args);return dict(rank=rank,owned=[],idle=False,node={})
 monkeypatch.setattr(cluster,'call',call)
 state={};cluster.supervisor_tick(tmp_path,0,state)
 assert state['phase']=='cpu_prepare_waiting_E1'
 assert all(args[-1]=='status' for args in calls)


def test_stale_gpu_start_never_spawns(tmp_path,monkeypatch):
 monkeypatch.setattr(cluster,'verify_code',lambda r:None);monkeypatch.setattr(cluster,'inspect',lambda r:[])
 monkeypatch.setattr(cluster.socket,'gethostname',lambda:cluster.NAMES[0])
 monkeypatch.setattr(cluster.subprocess,'Popen',lambda *a,**kw:pytest.fail('Must not spawn'))
 atomic_json(tmp_path/'status.json',{'generation':'current','gpu_authorized':True})
 with pytest.raises(ValueError,match='Stale'):cluster.agent(tmp_path,0,'start-node',generation='old')
 atomic_json(tmp_path/'status.json',{'generation':'current','gpu_authorized':False})
 with pytest.raises(ValueError):cluster.agent(tmp_path,0,'start-node',generation='current')


def test_failed_final_audit_requeues_repair_instead_of_repeating_idle_audit(tmp_path,monkeypatch):
 fake_ready(tmp_path);monkeypatch.setattr(cluster,'verify_code',lambda r:None)
 monkeypatch.setattr(cluster,'call',lambda rank,args:dict(rank=rank,owned=[],idle=True,node={'finished':True,'generation':'old'}))
 atomic_json(tmp_path/'errors/audit-None.json',{'error':'missing cache payload'})
 state={'generation':'old','gpu_authorized':True};cluster.supervisor_tick(tmp_path,0,state)
 assert state['generation']!='old' and state['phase']=='gpu'
 assert not (tmp_path/'errors/audit-None.json').exists()
 assert len(list((tmp_path/'resolved_errors').iterdir()))==1


def test_final_audit_waits_for_active_repair_producer(tmp_path,monkeypatch):
 fake_ready(tmp_path);monkeypatch.setattr(cluster,'verify_code',lambda r:None)
 calls=[]
 def call(rank,args):
  calls.append(args)
  return dict(rank=rank,owned=([{'role':'prepare'}] if rank==0 else []),idle=True,node={'finished':True,'generation':'same'})
 monkeypatch.setattr(cluster,'call',call)
 state={'generation':'same','gpu_authorized':True};cluster.supervisor_tick(tmp_path,0,state)
 assert state['phase']=='waiting_preparation_settle'
 assert all(args[-1]=='status' for args in calls)


def test_unadmitted_can_be_prepared_without_admitting_training():
 from test_p2_listen_core import label,obs
 from listen_audio_postprocess.prepare import adapt_canonical
 from p2_listen.annotations import validate
 original=label();original['training_admitted']=False;original['provenance']['formal_training_admitted']=False
 cfg={'events':{k:original['provenance'][k] for k in ('mapping_version','admission_policy')}}
 before=copy.deepcopy(original)
 with pytest.raises(ValueError,match='not been admitted'):adapt_canonical(original,{'audio_events':[]},{},obs(),cfg)
 result=adapt_canonical(original,{'audio_events':[]},{},obs(),cfg,require_admitted=False)
 assert original==before and result['training_admitted'] is False and result['provenance']['formal_training_admitted'] is False
 validate(result,obs(),require_admitted=False)
 with pytest.raises(ValueError,match='not been admitted'):validate(result,obs())


def test_default_training_loader_keeps_admission_gate(monkeypatch):
 from listen_audio_postprocess import overlay
 def contract(*args,**kwargs):
  assert kwargs.get('require_admitted',True) is True
  raise ValueError('Candidate labels have not been admitted')
 monkeypatch.setattr(overlay,'sample_contract',contract)
 with pytest.raises(ValueError,match='not been admitted'):overlay.load_training_sample({}, {}, {})
