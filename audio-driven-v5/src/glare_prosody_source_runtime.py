"""Compatibility/instrumentation around the unmodified official offline analyzer."""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

import numpy as np
from glare_prosody_subset import IntegrityError, generation_rows, sha


def official_module(path):
    spec = importlib.util.spec_from_file_location('glare_offline_audited', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialRuntime:
    def __init__(self, contract):
        self.contract = contract
        self.native = official_module(contract['official_script'])
        self.analyzer = None
        self.eos_ids = None

    def waveform(self, track):
        audio = track['audio']
        if sha(audio['path']) != audio['sha256']:
            raise IntegrityError('Speaker source WAV changed')
        wave = self.native.load_waveform(Path(audio['path']), 16000)
        if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Empty/nonfinite source waveform')
        return self.native.normalize_waveform_loudness(wave, -20., .98)

    def load(self):
        if self.analyzer is not None:
            return
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
            raise IntegrityError('Only authorized physical GPU7 may be used')
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 28 * 1024**3:
            raise RuntimeError('GPU7 needs at least28GiB free; other jobs are preserved')
        torch.cuda.set_per_process_memory_fraction(24*1024**3 / torch.cuda.get_device_properties(0).total_memory)
        for path, digest in self.contract['assets_sha256'].items():
            if sha(path) != digest:
                raise IntegrityError('Qwen asset changed: '+path)
        self.analyzer = self.native.Qwen2AudioAnalyzer(Path(self.contract['model']),
            self.contract['prompt'], 64, 'cuda:0')
        analyzer = self.analyzer
        processor = analyzer.processor
        # Official source passes audio= regardless of its loop variable; adapt only API spelling.
        audio_kw = 'audios' if 'audios' in inspect.signature(processor.__call__).parameters else 'audio'
        class ProcessorAdapter:
            def __call__(self, **kwargs):
                waveforms = kwargs.pop('audio')
                return processor(**kwargs, **{audio_kw:waveforms})
            def __getattr__(self, name):
                return getattr(processor, name)
        analyzer.processor = ProcessorAdapter()
        eos = analyzer.model.generation_config.eos_token_id
        if eos is None:
            eos = processor.tokenizer.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        self.rows = []
        self.raw = []
        normalize = self.native.normalize_result
        def capture(raw, **kwargs):
            self.raw.append(raw)
            return normalize(raw, **kwargs)
        self.native.normalize_result = capture
        generate = analyzer.model.generate
        tower_calls = [0]
        analyzer.model.audio_tower.register_forward_hook(lambda *_: tower_calls.__setitem__(0, tower_calls[0]+1))
        merge_calls = [0]
        merge = getattr(analyzer.model, '_merge_input_ids_with_audio_features', None)
        if merge is not None:
            def audited_merge(*args, **kwargs):
                merge_calls[0] += 1
                return merge(*args, **kwargs)
            analyzer.model._merge_input_ids_with_audio_features = audited_merge
        def audited_generate(**inputs):
            features = inputs.get('input_features')
            mask = inputs.get('feature_attention_mask')
            if features is None or mask is None or not bool((mask.sum(-1)>0).all()):
                raise IntegrityError('Missing real audio features/masks')
            audio_id = analyzer.model.config.audio_token_index
            if not bool(((inputs['input_ids']==audio_id).sum(-1)>0).all()):
                raise IntegrityError('Audio tokens missing from a batch row')
            before = tower_calls[0]
            before_merge = merge_calls[0]
            result = generate(**inputs)
            if tower_calls[0] <= before:
                raise IntegrityError('Model did not invoke its audio tower')
            if merge is not None and merge_calls[0] <= before_merge:
                raise IntegrityError('Audio features were not merged into language input')
            suffix = result[:,inputs['input_ids'].shape[1]:].detach().cpu().tolist()
            self.rows = generation_rows(suffix, self.eos_ids, 64)
            for row in self.rows:
                row.update(eos_ids=self.eos_ids, max_new_tokens=64, audio_tower_verified=True,
                           processor_audio_keyword=audio_kw)
                row['audio_merge_verified'] = merge_calls[0] > before_merge if merge is not None else None
            return result
        analyzer.model.generate = audited_generate

    def infer(self, waves):
        self.load()
        self.raw, self.rows = [], []
        normalized = self.analyzer.infer_chunks(waves, 16000, 2.4)
        if len(normalized) != len(waves) or len(self.raw) != len(waves) or len(self.rows) != len(waves):
            raise IntegrityError('Official batch/raw/generation cardinality mismatch')
        return list(zip(self.raw, self.rows, normalized))


def waveform_sha(wave):
    return hashlib.sha256(np.asarray(wave, dtype=np.float32).tobytes()).hexdigest()
