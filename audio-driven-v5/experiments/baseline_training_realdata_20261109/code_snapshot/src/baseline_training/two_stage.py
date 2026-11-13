"""Unreduced native L2L and DIM codecs and predictors on final-pool clips."""
import gc
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from .data import ROOT, WORK, read, save, sha
from .engine import check_features, exercise, setup


def windows(work, method):
    protocol = read(Path(work) / 'DATA_PROTOCOL.json')
    data, origins, excluded, hashes = {}, {}, [], {}
    for split, ids in [('train', protocol['diagnostic_train']), ('validation', protocol['diagnostic_validation'])]:
        data[split], origins[split] = [], []
        for sid in ids:
            value = check_features(work, method, sid)
            hashes[sid] = sha(Path(work) / 'features' / method / (sid + '.pt'))
            n = len(value['listener'])
            if method == 'l2l' and n < 64:
                excluded.append(dict(sample_id=sid, frames=n, reason='native64-frame window; no padding'))
                continue
            starts = range(0, n - 63, 32) if method == 'l2l' else [0]
            for start in starts:
                length = 64 if method == 'l2l' else min(64, n) // 2 * 2
                item = {k: v[start * (4 if k == 'audio' and method == 'l2l' else 1):
                                  (start + length) * (4 if k == 'audio' and method == 'l2l' else 1)].float()
                        for k, v in value.items() if torch.is_tensor(v) and k in ['speaker', 'listener', 'audio']}
                data[split].append(item)
                origins[split].append(dict(sample_id=sid, start_frame=start, frames=length))
        assert data[split], 'No usable real ' + split + ' windows'
    groups = protocol['diagnostic_groups']
    assert not {groups[x['sample_id']] for x in origins['train']} & {groups[x['sample_id']] for x in origins['validation']}
    info = dict(features_sha256=hashes, origins=origins, excluded_short_clips=excluded,
                split_policy='group-disjoint diagnostic; official benchmark test unopened')
    if method == 'l2l':
        sys.path.insert(0, str(ROOT / 'compare/learning2listen/src'))
        from utils.load_utils import bilateral_filter, mean_std_swap
        arrays = {split: {key: np.stack([x[key].numpy() for x in items]) for key in ['speaker', 'listener', 'audio']}
                  for split, items in data.items()}
        statistics = {}
        for split in arrays:
            for key in ['speaker', 'listener']:
                arrays[split][key] = bilateral_filter(arrays[split][key])
        for key in ['speaker', 'listener', 'audio']:
            mean, std = mean_std_swap(arrays['train'][key])
            assert np.isfinite(std).all() and (std > 0).all()
            statistics[key + '_mean'], statistics[key + '_std'] = mean, std
            for split in arrays:
                arrays[split][key] = ((arrays[split][key] - mean) / std).astype(np.float32)
        data = {split: [{key: torch.from_numpy(value[i]) for key, value in arrays[split].items()}
                        for i in range(len(data[split]))] for split in data}
        folder = Path(work) / 'runs/l2l'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / 'train_only_statistics.npz', **statistics)
        info['statistics_sha256'] = sha(folder / 'train_only_statistics.npz')
        info['preprocessing'] = 'native64-frame windows -> native bilateral_filter -> native mean_std_swap on training windows'
    save(Path(work) / 'runs' / method / 'WINDOWS.json', info)
    return data, info


class L2L:
    def __init__(self, work, phase, data, info, device):
        self.work, self.phase, self.data, self.data_identity, self.device = Path(work), phase, data, info, device
        repo = ROOT / 'compare/learning2listen/src'
        sys.path[:0] = [str(ROOT / 'compare/deployment'), str(repo)]
        from l2l_reliable_core import factories
        self.model, self.optimizer, cfg, vqcfg = factories(repo, 'vq' if phase == 'codec' else 'predictor', 137)
        self.info = dict(native_repository=str(repo), config=cfg, codec_config=vqcfg, phase=phase,
                         official_optimizer_scheduler=True, full_native_dimensions=True,
                         native_source_sha256={str(p.relative_to(repo)): sha(p) for p in repo.rglob('*.py')})
        self.codec = None
        self.required_gradient_groups = ['encoder', 'decoder', 'quantize'] if phase == 'codec' else ['audio_projector', 'motion_projector', 'cross_modal_layer']
        if phase == 'predictor':
            from vqgan.vqmodules.gan_models import setup_vq_transformer
            from types import SimpleNamespace
            self.codec, _, _ = setup_vq_transformer(SimpleNamespace(), vqcfg, load_path=None)
            checkpoint = self.work / 'runs/l2l/codec_final.pt'
            self.codec.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            self.codec.eval().requires_grad_(False)
            self.info['codec_sha256'] = sha(checkpoint)

    def train(self):
        self.model.train()

    def loss(self, step, validation):
        split = 'validation' if validation else 'train'
        item = self.data[split][step % len(self.data[split])]
        origin = self.data_identity['origins'][split][step % len(self.data[split])]
        y = item['listener'][None].to(self.device)
        if self.phase == 'codec':
            from vqgan.vqmodules.gan_models import calc_vq_loss
            y = y[:, 32 * (step % 2):32 * (step % 2) + 32]
            pred, quant = self.model(y, None)
            loss = calc_vq_loss(pred, y, quant)
        else:
            from modules.fact_model import calc_logit_loss
            with torch.no_grad():
                _, past = self.codec.module.get_quant(y[:, :32])
                _, future = self.codec.module.get_quant(y[:, 32:64])
            inputs = dict(listener_past=past.reshape(1, 4),
                          speaker_full=item['speaker'][None, :40].to(self.device),
                          audio_full=item['audio'][None, :160].to(self.device))
            pred = self.model(inputs, 4, 4 if validation else -1)
            loss = calc_logit_loss(pred[:, :4], future.reshape(1, 4))
        assert torch.isfinite(pred).all()
        return loss, dict(**origin, native_phase=self.phase)


class DIM:
    def __init__(self, work, phase, data, info, device):
        self.work, self.phase, self.data, self.data_identity, self.device = Path(work), phase, data, info, device
        repo = ROOT / 'compare/Dyadic-Interaction-Modeling/code'
        sys.path.insert(0, str(repo))
        from base.config import load_cfg_from_cfg_file
        from models import get_model
        config = repo / ('config_speaker_old.yaml' if phase == 'speaker_codec' else 'config.yaml')
        cfg = load_cfg_from_cfg_file(str(config))
        self.info = dict(native_repository=str(repo), config_sha256=sha(config),
                         phase=phase, full_native_dimensions=True, motion='EMOCA-v2 pose6+exp50', audio='HuBERT last768',
                         native_source_sha256={str(p.relative_to(repo)): sha(p) for p in repo.rglob('*.py')})
        if phase == 'predictor':
            import x_transformers.autoregressive_wrapper as wrapper
            functional = wrapper.F
            class DeterministicCrossEntropy:
                def __getattr__(self, name):
                    return getattr(functional, name)

                def cross_entropy(self, logits, target, **kwargs):
                    # The 3D NLL CUDA kernel rejects deterministic mode. Flatten
                    # batch/time and preserve class axis, ignore_index and mean
                    # reduction; this is the same token CE with the 2D kernel.
                    from .ops import token_cross_entropy
                    return token_cross_entropy(logits, target, **kwargs)
            wrapper.F = DeterministicCrossEntropy()
            self.info['deterministic_ce'] = 'flatten batch/time for native token cross entropy; same targets and reduction'
            isolated = self.work / 'dim_native_workdir'
            isolated.mkdir(exist_ok=True)
            dependencies = {}
            for role, filename in [('speaker', 'config_speaker_old.yaml'), ('listener', 'config.yaml')]:
                shutil.copyfile(repo / filename, isolated / filename)
                path = self.work / 'runs/dim' / (role + '_codec_final.pt')
                target = isolated / 'runs' / (role + '_exp') / 'model/model.pth.tar'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                dependencies[role] = sha(path)
            old_cwd = Path.cwd()
            try:
                os.chdir(isolated)
                from seq2seq import ListenerGenerator
                self.model = ListenerGenerator().to(device)
            finally:
                os.chdir(old_cwd)
            self.info['codec_sha256'] = dependencies
            for role in ['speaker', 'listener']:
                native = getattr(self.model, role + '_vq')
                value = torch.load(self.work / 'runs/dim' / (role + '_codec_final.pt'), map_location='cpu', weights_only=True)['state_dict']
                assert set(value) == set(native.state_dict())
                assert all(torch.equal(v, native.state_dict()[k].cpu()) for k, v in value.items())
        else:
            self.model = get_model(cfg).to(device)
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.base_lr)
        self.required_gradient_groups = (['listener_vq', 'generator'] if phase == 'predictor' else
            ['encoder', 'quantize', 'decoder_v', 'decoder_a'] if phase == 'speaker_codec' else ['encoder', 'quantize', 'decoder'])

    def train(self):
        self.model.train()
        if self.phase == 'predictor':
            self.model.speaker_vq.eval()
            self.model.listener_vq.encoder.eval()
            self.model.listener_vq.quantize.eval()

    def loss(self, step, validation):
        split = 'validation' if validation else 'train'
        index = step % len(self.data[split])
        item = self.data[split][index]
        speaker = torch.cat([item['speaker'], item['audio']], -1)[None].to(self.device)
        listener = item['listener'][None].to(self.device)
        if self.phase == 'predictor':
            mask = torch.ones(listener.shape[:2], dtype=torch.bool, device=self.device)
            loss, pred = self.model(speaker, listener, mask)
        else:
            from metrics.loss import calc_vq_loss
            target = speaker if self.phase == 'speaker_codec' else listener
            pred, quant, _ = self.model(target)
            loss, _ = calc_vq_loss(pred, target, quant, quant_loss_weight=1.)
        assert torch.isfinite(pred).all()
        return loss, dict(**self.data_identity['origins'][split][index], native_phase=self.phase)


def run(method, work=WORK, steps=4, resume=None, device='cuda:0', validate=True, **kwargs):
    assert not kwargs.get('full_pool'), 'Two-stage full-pool preprocessing must be materialized before a formal run'
    setup(device)
    data, info = windows(work, method)
    phases = ['predictor'] if resume else (['codec', 'predictor'] if method == 'l2l' else ['speaker_codec', 'listener_codec', 'predictor'])
    reports = []
    for phase in phases:
        adapter = (L2L if method == 'l2l' else DIM)(work, phase, data, info, device)
        report = exercise(adapter, work, method, phase, steps, validate, resume if phase == 'predictor' else None)
        if phase != 'predictor':
            state = adapter.model.state_dict()
            state = {k: x.detach().cpu() for k, x in state.items()}
            if method == 'dim':
                state = dict(state_dict=state)
            torch.save(state, Path(work) / 'runs' / method / (phase + '_final.pt'))
        reports.append(report)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    save(Path(work) / 'runs' / method / 'ACCEPTED.json', dict(passed=True, method=method, stages=reports,
         real_data=True, full_native_model=True, diagnostic_only=True, full500h_convergence_claim=False))
