"""CPU diagnostic of GLARE's unchanged FMT, with reduced width/depth only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'related/glare'))
import torch
from float.options.train_options import TrainOptions
from float.models.FMT import FlowMatchingTransformer

torch.set_num_threads(2)
torch.manual_seed(137)
options = TrainOptions().parse(['--dim_h', '64', '--fmt_depth', '2', '--num_heads', '4',
                               '--num_ref_frames', '2', '--num_prev_frames', '2',
                               '--wav2vec_sec', '0.16', '--use_alpha_mean', '--use_alpha_std',
                               '--use_speed_mean', '--use_speed_std', '--use_accel_mean', '--use_accel_std'])
options.rank = 'cpu'
model = FlowMatchingTransformer(options)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
before = model.decoder.linear.weight.detach().clone()
history = []
for step in range(2):
    b, frames = 2, 4
    stats = {k: torch.randn(b, 1, 20) for k in ['alpha_mean', 'alpha_std', 'speed_mean',
                                              'speed_std', 'accel_mean', 'accel_std']}
    velocity, hidden = model(torch.rand(b), torch.randn(b, frames, 512),
                             torch.randn(b, frames, 512), torch.randn(b, 512), torch.randn(b, 1, 7), stats,
                             prev_x=torch.randn(b, 2, 512), prev_wa=torch.randn(b, 2, 512),
                             ref_x=torch.randn(b, 2, 512), train=True,
                             intensity_scores=torch.randn(b, frames, 4),
                             prev_intensity_scores=torch.randn(b, 2, 4))
    assert velocity.shape == (b, 6, 512) and torch.isfinite(velocity).all()
    loss = (velocity - torch.randn_like(velocity)).square().mean()
    optimizer.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    optimizer.step()
    history.append(float(loss.detach()))
assert not torch.equal(before, model.decoder.linear.weight)
report = {'passed': True, 'kind': 'CPU synthetic native FMT diagnostic; not full GLARE or 500h training',
          'changes_for_diagnostic': {'dim_h': 64, 'fmt_depth': 2, 'num_heads': 4,
                                     'current_frames': 4, 'reference_frames': 2, 'history_frames': 2},
          'native_motion_dimension': 512, 'native_alpha_dimension': 20, 'alpha_conditions': 6,
          'updates': 2, 'losses': history, 'prediction_shape': list(velocity.shape),
          'changed_parameter': 'decoder.linear.weight', 'gpu_used': False,
          'source_sha256': hashlib.sha256((ROOT / 'related/glare/float/models/FMT.py').read_bytes()).hexdigest()}
(Path(__file__).parent / 'GLARE_FMT_SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
