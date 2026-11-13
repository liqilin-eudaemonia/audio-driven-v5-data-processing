"""Memoize immutable code/model identities during an audit, then recheck them.

Sample labels, recipes, media, cache markers and payloads are never memoized.
This helper is used by audit scripts only; it changes no cache contract values.
"""
from copy import deepcopy
from pathlib import Path


class IdentitySnapshot:
    def __init__(self,config):
        from p2_listen import features,runtime,annotations,layout
        from p2_listen.contracts import sha,ROOT
        from listen_audio_postprocess import overlay,recipes,encoder
        from listen_benchmark import official_audio_gate,chinese_asr_filter
        self.config=deepcopy(config);self.sha=sha;self.features=features
        paths=[features.__file__,runtime.__file__,annotations.__file__,layout.__file__,overlay.__file__,
               recipes.__file__,encoder.__file__,official_audio_gate.__file__,chinese_asr_filter.__file__,
               ROOT/'src/p3_listen/features.py',Path(config['output'])/'audio_postprocess/policy_manifest.json']
        self.hashes={str(p):sha(p) for p in paths}
        self.original_weights=features.weights_identity
        self.weights={k:self.original_weights(config,k) for k in ['omni','ltx','gemma']}
        self.modules=[features,overlay,recipes];self.old_sha={m:m.sha for m in self.modules}
        for m in self.modules:m.sha=self.cached_sha
        features.weights_identity=self.cached_weights

    def cached_sha(self,path):
        key=str(path)
        return self.hashes[key] if key in self.hashes else self.sha(path)

    def cached_weights(self,config,kind):
        if config!=self.config:raise ValueError('Audit configuration changed')
        return deepcopy(self.weights[kind])

    def verify_and_restore(self):
        try:
            for path,expected in self.hashes.items():
                if self.sha(path)!=expected:raise ValueError('Code/policy changed during audit: '+path)
            for k,expected in self.weights.items():
                if self.original_weights(self.config,k)!=expected:raise ValueError('Model identity changed during audit: '+k)
        finally:
            self.features.weights_identity=self.original_weights
            for m,fn in self.old_sha.items():m.sha=fn
        return dict(code_and_policy_files=len(self.hashes),model_identities=len(self.weights),unchanged_before_after=True)
