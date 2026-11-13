"""Official analyzer with API spelling, bounded memory and receipt instrumentation."""
import inspect
import os
from pathlib import Path
from glare_prosody_subset import generation_rows,IntegrityError,sha
from glare_prosody_source_runtime import OfficialRuntime


class CapacityOfficialRuntime(OfficialRuntime):
    def __init__(self,contract,gpu=7,max_memory_gib=32,reserve_memory_gib=3,asset_verifier=None):
        super().__init__(contract)
        self.gpu=int(gpu);self.max_memory_gib=max_memory_gib;self.reserve_memory_gib=reserve_memory_gib
        self.asset_verifier=asset_verifier
        self.execution_evidence=dict(mode='official_complete',loader_sha256=sha(__file__),early_stop=False,
                                     physical_gpu=self.gpu,max_memory_gib=max_memory_gib)
    def load(self):
        if self.analyzer is not None:return
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(self.gpu):raise IntegrityError('GPU visibility differs from assigned slot')
        torch.set_num_threads(2);torch.cuda.set_device(0)
        if torch.cuda.device_count()!=1:raise IntegrityError('Exactly one assigned GPU must be visible')
        free,total=torch.cuda.mem_get_info();allowance=min(self.max_memory_gib*1024**3,free-self.reserve_memory_gib*1024**3)
        if allowance<17*1024**3:raise RuntimeError('Insufficient assigned GPU memory for official Qwen')
        torch.cuda.set_per_process_memory_fraction(allowance/torch.cuda.get_device_properties(0).total_memory)
        self.memory_allowance_gib=allowance/1024**3
        if self.asset_verifier is not None:self.asset_verifier()
        else:
            for path,digest in self.contract['assets_sha256'].items():
                if sha(path)!=digest:raise IntegrityError('Qwen model asset changed: '+path)
        self.analyzer=self.native.Qwen2AudioAnalyzer(Path(self.contract['model']),self.contract['prompt'],64,'cuda:0')
        analyzer=self.analyzer;processor=analyzer.processor
        key='audios' if 'audios' in inspect.signature(processor.__call__).parameters else 'audio'
        class Adapter:
            def __call__(self,**kwargs):
                audio=kwargs.pop('audio');return processor(**kwargs,**{key:audio})
            def __getattr__(self,name):return getattr(processor,name)
        analyzer.processor=Adapter()
        eos=analyzer.model.generation_config.eos_token_id
        if eos is None:eos=processor.tokenizer.eos_token_id
        self.eos_ids=eos if isinstance(eos,list) else [eos]
        original_normalize=self.native.normalize_result
        def capture(raw,**kwargs):self.raw.append(raw);return original_normalize(raw,**kwargs)
        self.native.normalize_result=capture
        original_generate=analyzer.model.generate;calls=[0]
        analyzer.model.audio_tower.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
        def generate(**kwargs):
            if kwargs.get('input_features') is None or kwargs.get('feature_attention_mask') is None:
                raise IntegrityError('Official processor did not supply audio')
            before=calls[0];generated=original_generate(**kwargs)
            if calls[0]<=before:raise IntegrityError('Official model skipped audio tower')
            suffix=generated[:,kwargs['input_ids'].shape[1]:].detach().cpu().tolist()
            self.rows=generation_rows(suffix,self.eos_ids,64)
            for row in self.rows:row.update(eos_ids=self.eos_ids,max_new_tokens=64,audio_tower_verified=True,processor_audio_keyword=key)
            return generated
        analyzer.model.generate=generate
