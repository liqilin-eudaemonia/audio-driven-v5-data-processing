import json,tempfile,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import atomic,cache_put,cached,semantic_job,lock,prepare_shard,fuse_shard,semantic_state,read,file_sha

class FullFusionTests(unittest.TestCase):
 def test_worker_batches_retries_without_changing_attempt_budgets(self):
  import importlib.util,types
  from unittest.mock import patch
  calls=[]
  class Params:
   def __init__(self,**kw):self.__dict__.update(kw)
  class Processor:
   @classmethod
   def from_pretrained(cls,*a,**k):return cls()
   def apply_chat_template(self,messages,**kw):return messages[0]['content'][0]['text']
  class LLM:
   def __init__(self,**kw):pass
   def generate(self,prompts,params,**kw):
    calls.append((len(prompts),[p.max_tokens for p in params]));out=[]
    for text,param in zip(prompts,params):
     fail=(param.max_tokens==768 and ('Smiles A.' in text or 'Smiles B.' in text)) or (param.max_tokens==1536 and 'Smiles A.' in text)
     answer=json.dumps(dict(item=dict(categories=['smile'],uncertain=False,reason='visible smile',evidence_source_ids=['raw'])))
     out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=answer,finish_reason='length' if fail else 'stop',token_ids=[1])]))
    return out
  with tempfile.TemporaryDirectory() as t:
   run=Path(t);atomic(run/'config.json',dict(model='stub',model_signature='weights',mapping_version='test'))
   jobs=[semantic_job('visual','Smiles '+c+'.','weights') for c in 'ABC'];p=run/'jobs/0000.jsonl';p.parent.mkdir();p.write_text(''.join(json.dumps(j)+'\n' for j in jobs));atomic(run/'state/ingest/0000.json',dict(done=True))
   spec=importlib.util.spec_from_file_location('semantic_worker_test',ROOT/'scripts/run_listen_full_fusion_semantic.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   modules={'vllm':types.SimpleNamespace(LLM=LLM,SamplingParams=Params),'vllm.sampling_params':types.SimpleNamespace(StructuredOutputsParams=Params),'transformers':types.SimpleNamespace(AutoProcessor=Processor)}
   with patch.dict(sys.modules,modules),patch.object(sys,'argv',['worker','--run',str(run),'--worker','test','--batch','2','--one-shard']):module.main()
   self.assertEqual(calls,[(2,[768,768]),(1,[768]),(2,[1536,1536]),(1,[2048])]);self.assertTrue(all(cached(run,j)['valid'] for j in jobs));self.assertTrue(read(run/'state/semantic/0000.json')['done'])
 def test_cache_key_preserves_exact_content_and_modality(self):
  a=semantic_job('audio','Yeah','weights');b=semantic_job('audio','yeah','weights');c=semantic_job('visual','Yeah','weights')
  self.assertNotEqual(a['key'],b['key']);self.assertNotEqual(a['key'],c['key'])
 def test_cache_immutable_first_complete_record(self):
  with tempfile.TemporaryDirectory() as t:
   job=semantic_job('visual','Smiles.','weights');r=dict(**job,valid=True,finish_reason='stop',answer=dict(categories=['smile'],uncertain=False,reason='visible smile',evidence_source_ids=['raw']))
   cache_put(t,r);cache_put(t,dict(r,worker='second'));self.assertEqual(cached(t,job),r)
 def test_duplicate_lock_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with lock(Path(t)/'x') as first:
    with lock(Path(t)/'x') as second:self.assertTrue(first);self.assertFalse(second)
 def test_full_shard_partial_raw_preserves_every_sample_and_resumes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/'raw_run';run=root/'fusion';atomic(run/'config.json',dict(raw_run=str(raw),model_signature='weights'))
   source=dict(sample_id='s',duration_s=2.,source_start_s=0.,source_end_s=2.,normalized_audio=dict(sample_rate=48000),source_line_sha256='identity')
   q=raw/'queue/shards/0000.jsonl';q.parent.mkdir(parents=True);q.write_text(json.dumps(source)+'\n')
   for b in ['visual','audio','asr']:
    r=dict(record_kind='result',sample_id='s',source=source,request_id='s@full',view='full',window_start_s=0.,window_end_s=2.,parent_duration_s=2.)
    if b=='visual':r.update(status='schema_pass',parsed=dict(events=[dict(start=.2,end=1.,modality='visual',behavior='Smiles softly.')]))
    elif b=='audio':r.update(status='validation_failed',error='bad interval')
    else:r.update(status='complete',raw_response=dict(text='Yeah',words=[dict(text='Yeah',start_time=.5,end_time=.8)]))
    p=raw/f'raw/{b}/results/0000.jsonl';p.parent.mkdir(parents=True);p.write_text(json.dumps(r)+'\n');atomic(raw/f'raw/{b}/state/0000.json',dict(done=True))
   marker=prepare_shard(run,'0000');self.assertEqual(marker['counts']['samples'],1);self.assertEqual(marker['counts']['samples_with_raw_issues'],1)
   job=json.loads((run/'jobs/0000.jsonl').read_text());cache_put(run,dict(**job,valid=True,finish_reason='stop',answer=dict(categories=['smile'],uncertain=False,reason='smile',evidence_source_ids=['raw'])))
   semantic_state(run,'0000','test');result=fuse_shard(run,'0000');output=json.loads((run/'labels/0000.jsonl').read_text())
   self.assertEqual(output['raw_completion_status'],'partial_raw');self.assertFalse(output['training_admitted']);self.assertEqual([e['description'] for e in output['events']],['Smiles softly.','Yeah'])
   before=file_sha(run/'labels/0000.jsonl');self.assertEqual(fuse_shard(run,'0000'),result);self.assertEqual(file_sha(run/'labels/0000.jsonl'),before)
 def test_incomplete_raw_not_ingested(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);atomic(root/'config.json',dict(raw_run=str(root/'raw'),model_signature='weights'));self.assertIsNone(prepare_shard(root,'0000'))
 def test_failed_semantic_not_fake_negative(self):
  with tempfile.TemporaryDirectory() as t:
   job=semantic_job('visual','ambiguous','weights');cache_put(t,dict(**job,valid=False,error='truncated'));self.assertFalse(cached(t,job)['valid']);self.assertIsNone(cached(t,job,allow_failure=False))
if __name__=='__main__':unittest.main()
