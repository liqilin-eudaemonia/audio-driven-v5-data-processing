import sys,pathlib,unittest
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from baseline_training.glare_batched import bucket_order,length_batches,PairMaskedReactionLoss
class BatchTests(unittest.TestCase):
 def test_all_members_and_reproducible_order(self):
  lengths=[50]*35+[100]*29+[200]*18
  a=bucket_order(lengths,137,16)
  self.assertEqual(sorted(a),list(range(len(lengths))))
  self.assertEqual(a,bucket_order(lengths,137,16))
  self.assertNotEqual(a,bucket_order(lengths,138,16))
 def test_migration_preserves_consumed_prefix(self):
  lengths=[50,100,200]*100
  old=torch.randperm(len(lengths),generator=torch.Generator().manual_seed(137)).tolist();cursor=97
  remaining=old[cursor:];order=old[:cursor]+[remaining[i] for i in bucket_order([lengths[i] for i in remaining],137,16)]
  self.assertEqual(order[:cursor],old[:cursor]);self.assertEqual(set(order[cursor:]),set(old[cursor:]));self.assertEqual(len(set(order)),300)
 def test_equal_pair_weight_and_unknown_pair(self):
  f=PairMaskedReactionLoss();p=torch.tensor([[[1.],[100.]],[[2.],[2.]]]);f.mask=torch.tensor([[[True],[False]],[[True],[True]]])
  self.assertEqual(f(p,torch.zeros_like(p)).item(),1.)
  f.mask[0]=False;self.assertEqual(f(p,torch.zeros_like(p)).item(),.75)
if __name__=='__main__':unittest.main()
