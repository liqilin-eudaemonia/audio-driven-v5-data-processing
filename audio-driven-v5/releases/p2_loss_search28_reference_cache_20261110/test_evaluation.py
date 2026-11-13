import unittest,tempfile,json,sys,copy
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(W),str(ROOT.parent/'OmniResponse')]
from p2_listen.contracts import read,sha,atomic_json
from p2_testsets_eval.protocol import partition
class EvaluationChecks(unittest.TestCase):
 def test_native_subset_seal(self):
  from evaluation.seamless.bind import load_references
  refs=load_references(W/'prepared64');self.assertEqual(len(refs),64)
  self.assertEqual({x['id']for x in refs},{x['id']for x in read(W/'tuning64.json')})
 def test_28_shards_exact_subset_with_sealed_templates(self):
  from binder import ShardBinder
  parent=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107';p=copy.deepcopy(read(parent/'protocol.json'))
  ids={x['id']for x in read(W/'tuning64.json')};items=[x for x in read(parent/'shared_items.json')if x['dataset']=='seamless'and x['id']in ids]
  p.update(workers=28,total_samples=64,datasets={'seamless':dict(p['datasets']['seamless'],prepared=str(W/'prepared64'),references_sha256=sha(W/'prepared64/references.jsonl'))})
  with tempfile.TemporaryDirectory()as folder:
   work=Path(folder);atomic_json(work/'protocol.json',p);atomic_json(work/'shared_items.json',items)
   for rank,rows in enumerate(partition(items,28)):atomic_json(work/'shards'/f'{rank:02d}.json',rows)
   b=ShardBinder(work);self.assertEqual(len(b.items),64);self.assertEqual(len(b.templates['seamless']),64)
   self.assertEqual({x['id']for x in b.templates['seamless']},ids);self.assertTrue(all(b.shards));self.assertEqual(sum(map(len,b.shards)),64)
 def test_no_train_overlap_at_three_levels(self):
  audit=read(W/'VALIDATION_MEMBERSHIP_ACCEPTED.json');self.assertEqual(audit['training_samples'],284218)
  self.assertTrue(all(not v for v in audit['train_tuning_overlaps'].values()))
  for f,h in audit['outputs'].items():self.assertEqual(sha(W/f),h)
 def test_requests_keep_real_steps_and_checkpoints(self):
  for target in (500,1500,3000,6000):
   for request in (W/'evaluations').glob(f'*_{target}/REQUEST.json'):
    r=read(request);self.assertTrue(r['checkpoint'].endswith(f'step_{target:06d}.pt'));self.assertEqual(r['samples'],64 if target==500 else 200)
if __name__=='__main__':unittest.main(verbosity=2)
