import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_execution_opts import install_last_logits,install_masked_audio_prefix
from glare_prosody_subset import IntegrityError


class ExecutionOverlayTests(unittest.TestCase):
    def test_supported_last_logit_option_is_reversible_and_generation_only(self):
        def forward(num_logits_to_keep=0,labels=None):return num_logits_to_keep
        lm=SimpleNamespace(forward=forward)
        runtime=SimpleNamespace(analyzer=SimpleNamespace(model=SimpleNamespace(language_model=lm)))
        restore=install_last_logits(runtime)
        self.assertEqual(lm.forward(),1)
        with self.assertRaises(IntegrityError):lm.forward(labels=[1])
        restore();self.assertIs(lm.forward,forward);self.assertEqual(lm.forward(),0)

    def test_trim_preserves_valid_convolution_attention_pool_outputs(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(137)
        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__();self.config=SimpleNamespace(max_source_positions=32)
                self.conv1=torch.nn.Conv1d(4,8,3,padding=1).double()
                self.conv2=torch.nn.Conv1d(8,8,3,padding=1,stride=2).double()
                self.embed_positions=torch.nn.Embedding(32,8).double()
                self.norm=torch.nn.LayerNorm(8).double();self.seen=[]
            def forward(self,input_features,attention_mask=None):
                self.seen.append(input_features.shape[-1])
                self.assert_len=input_features.shape[-1]==self.config.max_source_positions*2
                x=F.gelu(self.conv2(F.gelu(self.conv1(input_features)))).transpose(1,2)
                x=x+self.embed_positions.weight
                for _ in range(3):
                    attn=F.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],attn_mask=attention_mask)[:,0]
                    x=self.norm(x+attn)
                return F.avg_pool1d(x.transpose(1,2),2,2).transpose(1,2)
        tower=Tower().eval();model=SimpleNamespace(audio_tower=tower)
        runtime=SimpleNamespace(analyzer=SimpleNamespace(model=model))
        features=torch.randn(2,4,64,dtype=torch.float64)
        mask=torch.zeros(2,1,32,32,dtype=torch.float64);mask[:,:,:,8:]=-float('inf')
        original_positions=tower.embed_positions
        with torch.inference_mode():baseline=tower(features,attention_mask=mask)
        restore=install_masked_audio_prefix(runtime)
        with torch.inference_mode():short=tower(features,attention_mask=mask)
        self.assertEqual(tower.seen,[64,20]);self.assertTrue(tower.assert_len)
        self.assertIs(tower.embed_positions,original_positions);self.assertEqual(tower.config.max_source_positions,32)
        torch.testing.assert_close(baseline[:,:4],short[:,:4],rtol=1e-12,atol=1e-12)
        bad=mask.clone();bad[:,:,:,3]=-float('inf')
        with self.assertRaises(IntegrityError):tower(features,attention_mask=bad)
        self.assertIs(tower.embed_positions,original_positions)
        restore()


if __name__=='__main__':unittest.main()
