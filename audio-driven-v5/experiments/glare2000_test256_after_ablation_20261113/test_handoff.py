import fcntl,json,tempfile,unittest
from pathlib import Path
import run as trial
import handoff
class HandoffTests(unittest.TestCase):
 def test_exact_scope_and_cached_inputs(self):
  p=trial.read(trial.WORK/'protocol.json');old=trial.read(trial.PARENT/'step10618/protocol.json')
  for key in ['shared_items_sha256','references_sha256','seed','nfe','audio_guidance','metric_assets','metrics_gt_cache']:self.assertEqual(p[key],old[key])
  self.assertEqual(p['step'],2000);self.assertEqual(p['total'],256);self.assertEqual(p['reused_generated'],0)
  self.assertEqual([(s['id'],s['reference_sha256'],s['audio_sha256']) for s in p['samples']],[(s['id'],s['reference_sha256'],s['audio_sha256']) for s in old['samples']])
 def test_all24_shards_disjoint_and_complete(self):
  p=trial.read(trial.WORK/'protocol.json');ids=[]
  for i in range(24):
   gen=trial.read(trial.WORK/'generation_shards'/f'{i}.json');score=trial.read(trial.WORK/'blocks'/f'{i:03d}.json');self.assertEqual(gen,score);ids.extend(s['id'] for s in gen)
  self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{s['id'] for s in p['samples']})
 def test_incomplete_pause_and_exit_lock_gates(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);self.assertFalse(handoff.dependency_ready(path)[0])
   trial.save(path/'COMPLETE.json',dict(success=True,outputs=1536,combinations=6))
   with (path/'controller.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertEqual(handoff.dependency_ready(path),(False,'waiting_for_upstream_controller_exit'))
   self.assertTrue(handoff.dependency_ready(path)[0]);(path/'STOP').touch();self.assertEqual(handoff.dependency_ready(path),(False,'upstream_paused'))
 def test_wrong_completion_never_triggers(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);trial.save(path/'COMPLETE.json',dict(success=True,outputs=64,combinations=1))
   with self.assertRaises(ValueError):handoff.dependency_ready(path)
 def test_frozen_native_code_and_weight(self):
  trial.verify();p=trial.read(trial.BASE/'CHECKPOINT.json');self.assertEqual(p['step'],2000);self.assertTrue(p['same_native_parameter_shapes_and_assets_as_10618'])
 def test_two_nodes_cover_all_logical_shards(self):
  import run_two_nodes as dispatch
  slots=dispatch.slots();self.assertEqual(len(slots),16)
  self.assertEqual({n['name'] for _,n,_ in slots},{'node12','node11'})
  shards=[{str(i)} for i in range(24)]
  busy=set();chosen=[]
  for _ in range(16):
   job=dispatch.choose_job(shards,set(),set(),busy);chosen.append(job);busy.add(job[1])
  self.assertEqual(len({j[1] for j in chosen}),16)
  self.assertEqual(dispatch.choose_job(shards,{str(i) for i in range(16)},set(),set()),('generate',16))
  self.assertEqual(dispatch.choose_job(shards,{str(i) for i in range(24)},set(),set()),('score',0))
if __name__=='__main__':unittest.main()
