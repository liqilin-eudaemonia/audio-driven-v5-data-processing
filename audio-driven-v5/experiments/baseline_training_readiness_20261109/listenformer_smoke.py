"""Native ListenFormer CPU forward/backward; synthetic tensors, no benchmark."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compare/ListenFormer'))
import torch
import yaml
from box import Box
from networks.listener_generator import ListenerGenerator_trans_ca

torch.set_num_threads(2)
torch.manual_seed(137)
config = Box(yaml.safe_load((ROOT / 'compare/ListenFormer/configs/configs.yaml').read_text()))
model = ListenerGenerator_trans_ca(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
audio = torch.randn(2, 8, 45)
speaker = torch.randn(2, 8, 73)
initial = torch.randn(2, 73)
target = torch.randn(2, 7, 73)
lengths = torch.tensor([8, 8])
before = model.driven_proj.weight.detach().clone()
loss, metrics, prediction = model(audio, speaker, initial, lengths, 0, 100, target=target)
assert torch.isfinite(loss) and torch.isfinite(prediction).all()
optimizer.zero_grad()
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().sum() > 0 for g in grads)
optimizer.step()
assert not torch.equal(before, model.driven_proj.weight)
result = {'passed': True, 'kind': 'native CPU synthetic forward/backward/Adam; not 500h training',
          'seed': 137, 'loss': float(loss.detach()), 'prediction_shape': list(prediction.shape),
          'parameters_with_gradient': len(grads), 'changed_parameter': 'driven_proj.weight',
          'gpu_used': False, 'model_source_sha256': hashlib.sha256(
              (ROOT / 'compare/ListenFormer/networks/listener_generator.py').read_bytes()).hexdigest()}
(Path(__file__).parent / 'LISTENFORMER_NATIVE_SMOKE.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
