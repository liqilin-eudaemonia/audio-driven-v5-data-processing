"""Audited execution-only overlay; semantic training identity stays unchanged."""
from types import MethodType

def configure(renderer,policy):
    from p2_listen.performance import configure_renderer
    if policy!='two_thirds':return configure_renderer(renderer,policy)
    from torch.utils.checkpoint import checkpoint
    # Keep the frozen upstream helper untouched for existing checkpoints.
    configure_renderer(renderer,'all')
    def blocks(self,video,audio,perturbations):
        for index,block in enumerate(self.transformer_blocks):
            if self.training and index%3!=0:
                video,audio=checkpoint(block,video,audio,perturbations,use_reentrant=False)
            else:video,audio=block(video=video,audio=audio,perturbations=perturbations)
        return video,audio
    renderer._process_transformer_blocks=MethodType(blocks,renderer)

def select_policy(seconds,bands):
    import math
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid execution duration')
    for band in bands:
        if seconds<=band['max_duration_s']:return band['policy']
    raise ValueError('Sample exceeds benchmarked execution envelope')

def install(manifest_path):
    """Changes rematerialization only; records provenance in every new checkpoint."""
    import json,os
    from pathlib import Path
    from p2_listen.contracts import read,sha
    from p2_legacy_qwen_lora import backend
    from p2_matrix16 import checkpoint,qwen
    from p2_matrix16.model import MatrixModel
    manifest=read(manifest_path)
    if manifest['schema']!='p2-legacy-execution-v1' or manifest['base_source']!=backend.source_identity():raise ValueError('Execution base source differs')
    for path,want in manifest['overlay_source'].items():
        if sha(path)!=want:raise ValueError('Execution overlay source changed')
    for key,want in [('global_batch',24),('size',512),('world',24)]:
        if manifest[key]!=want:raise ValueError('Unsupported execution contract')
    if int(os.environ.get('WORLD_SIZE','1'))!=24:raise ValueError('Production execution overlay requires all 24 ranks')
    original_forward=MatrixModel.forward
    def forward(self,bundle,*args,**kwargs):
        if self.spec['training']['global_batch']!=24:raise ValueError('Batch changed')
        policy=select_policy(bundle['observation']['requested_duration_s'],manifest['bands'])
        if getattr(self,'_execution_policy',None)!=policy:
            configure(self.renderer,policy);self._execution_policy=policy
        return original_forward(self,bundle,*args,**kwargs)
    MatrixModel.forward=forward
    original_inject=qwen.inject_qwen_lora
    def inject(runtime,spec):
        result=original_inject(runtime,spec)
        if not manifest['qwen_checkpointing']:runtime.thinker.model.gradient_checkpointing_disable()
        return result
    qwen.inject_qwen_lora=inject
    provenance=dict(manifest_path=str(Path(manifest_path).resolve()),manifest_sha256=sha(manifest_path),spec=manifest)
    original_atomic=checkpoint.atomic_torch
    def save(path,value):
        value=dict(value,execution=provenance)
        return original_atomic(path,value)
    checkpoint.atomic_torch=save
    print(json.dumps(dict(execution_overlay=provenance['manifest_sha256'],bands=manifest['bands'],qwen_checkpointing=manifest['qwen_checkpointing'],rank=os.environ.get('RANK'))),flush=True)
