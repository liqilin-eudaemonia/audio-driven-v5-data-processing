import json
import pathlib
import sys
import tempfile
import unittest
import hashlib
import numpy as np
import torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baseline_training.glare_cached import load_row, sha, save_checkpoint

class CachedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name); n=5
        values={'lia':{'motion':torch.randn(n,512)}, 'audio':{'audio':torch.randn(n*640)},
          'frozen_audio':{'wav2vec':torch.randn(n-1,9216),'emotion':torch.rand(7)},
          'statistics':{'alpha_mean':torch.randn(20)}}
        self.row={'sample_id':'sample','glare_caches':{}}
        for name,value in values.items():
            p=self.root/(name+'.pt');torch.save(value,p);self.add(name,p)
        for name,value in [('reaction',dict(reaction=np.zeros((n,6),np.float32),reaction_valid=np.zeros((n,6),bool))),
                           ('prosody',dict(intensity=np.zeros((n,1),np.float32),intensity_valid=np.ones((n,1),bool)))]:
            p=self.root/(name+'.npz');np.savez(p,**value);self.add(name,p)
        c=self.row['glare_caches'];c['view']=dict(reference_frame=0,history_frames=0,frames=n,target_frames=n-1)
        c['frozen_audio'].update(pre_projection=True,waveform_samples=(n-1)*640)
    def add(self,name,p):
        r=p.with_suffix('.json');r.write_text('{}')
        self.row['glare_caches'][name]=dict(payload=str(p),receipt=str(r),payload_sha256=sha(p),receipt_sha256=sha(r))
    def test_zero_null_kept_and_missing_reaction_mask_preserved(self):
        v=load_row(self.row,True)
        self.assertFalse(v['reaction_valid'].any());self.assertEqual(v['intensity'].sum(),0)
        self.assertEqual(v['wav2vec'].shape,(4,9216))
    def test_changed_receipt_rejected(self):
        pathlib.Path(self.row['glare_caches']['lia']['receipt']).write_text('changed')
        with self.assertRaises(AssertionError):load_row(self.row)
    def test_changed_payload_rejected(self):
        p=pathlib.Path(self.row['glare_caches']['lia']['payload']);torch.save({'motion':torch.zeros(5,512)},p)
        with self.assertRaises(AssertionError):load_row(self.row,True)
    def test_window_contract_rejected(self):
        self.row['glare_caches']['frozen_audio']['waveform_samples']-=1
        with self.assertRaises(AssertionError):load_row(self.row)
    def test_atomic_checkpoint(self):
        p=self.root/'latest.pt';save_checkpoint(p,{'step':2,'x':torch.ones(2)})
        self.assertEqual(torch.load(p,weights_only=True)['step'],2)
        self.assertFalse(list(self.root.glob('*.tmp.*')))

if __name__=='__main__':unittest.main()
