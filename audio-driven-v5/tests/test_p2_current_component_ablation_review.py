"""Review gaps: each new arm's recomputation and actual CFG inference path."""
import pytest
import torch

from test_p2_current_component_ablation import ARMS, build, forward, objective
from test_p2_whisper_av import opened


@pytest.mark.parametrize('arm', ARMS)
def test_all_recomputation_matches_loss_and_every_gradient(arm):
    from p2_listen.performance import configure_renderer
    c, _, model, bundle, strategy, sampler = build(arm)
    if ARMS[arm]['whisper']:
        opened(model)
    model.train()
    results = []
    counts = {k: v['known'].sum() for k, v in bundle.get('events', {}).items()}
    for policy in ('none', 'all'):
        configure_renderer(model.renderer, policy)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(731)
        values = forward(model, bundle, strategy, sampler)
        loss = objective(values, c, counts)[0]
        loss.backward()
        results.append((loss.detach().clone(), {
            n: None if p.grad is None else p.grad.detach().clone()
            for n, p in model.named_parameters()
        }))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, grad in results[0][1].items():
        other = results[1][1][name]
        if grad is None:
            assert other is None, name
        else:
            torch.testing.assert_close(grad, other, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize('arm', ARMS)
def test_native_cfg_retains_enabled_conditions_and_reference(arm):
    from omni_av.ltx_runtime import with_text
    from p2_current_component_ablation.inference import sample
    _, _, model, bundle, strategy, sampler = build(arm)
    if ARMS[arm]['whisper']:
        opened(model)
    model.eval()
    prepared = strategy.prepare_training_inputs(
        with_text(bundle['av']['av'], bundle['text']['positive']), sampler)
    with torch.no_grad():
        u = model.connector(bundle['condition'])
    speaker = bundle.get('whisper')
    seen = []

    def capture(module, args, kwargs):
        seen.append((kwargs['video'], kwargs['audio']))

    handle = model.renderer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        out = sample(model, prepared, u, speaker, {
            'video': torch.zeros_like(prepared.video.context),
            'audio': torch.zeros_like(prepared.audio.context),
            'mask': prepared.video.context_mask,
        }, steps=2, sigmas=torch.tensor([1., .5, 0.]))
    finally:
        handle.remove()
    assert len(seen) == 4
    for video, audio in seen:
        for mod, value in [('video', video), ('audio', audio)]:
            assert (value.qwen_context is not None) == ARMS[arm]['whisper']
            if speaker is not None:
                assert value.qwen_time_positions is speaker.bounds
                assert torch.equal(value.qwen_context_mask, speaker.valid)
            condition = getattr(u, mod)
            assert torch.equal(value.context[:, -condition.shape[1]:], condition)
    keep = ~prepared.video_loss_mask
    assert torch.equal(out['video'][keep], prepared.video.latent.float()[keep])
    assert all(torch.isfinite(v).all() for v in out.values())
