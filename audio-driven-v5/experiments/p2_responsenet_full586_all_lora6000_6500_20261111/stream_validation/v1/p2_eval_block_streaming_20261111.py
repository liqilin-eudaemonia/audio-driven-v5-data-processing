"""Native LTX block buffers around already loaded, strictly checked models.

Preserve every tensor dtype/value, including separate trained LoRA/Whisper
branches. No LoRA fusion, quantization, CPU matrix multiplication or new math.
"""
import collections
import gc
import itertools


class Providers:
    def __init__(self,by_index):self.by_index=by_index
    def get(self,idx):return self.by_index[idx].get(idx)
    def release(self,idx,event):self.by_index[idx].release(idx,event)
    def cleanup(self):
        for provider in set(self.by_index.values()):provider.cleanup()


def stream(model,prefix,device='cuda:0',slots=2):
    import torch
    from ltx_core.block_streaming.utils import resolve_attr,derive_layout,allocate_layout_views,assign_tensor_to_module
    from ltx_core.block_streaming.source import PinnedWeightSource
    from ltx_core.block_streaming.pool import WeightPool
    from ltx_core.block_streaming.provider import WeightsProvider
    from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
    target=torch.device(device);blocks=resolve_attr(model,prefix)
    groups=collections.defaultdict(dict);nbytes=0
    for idx,block in enumerate(blocks):
        values=dict(itertools.chain(block.named_parameters(),block.named_buffers()))
        assert all(t.device.type=='cpu' for t in values.values()),'Load the original model on CPU first'
        layout=derive_layout(values)
        signature=tuple((key,tuple(shape),str(dtype)) for key,(shape,dtype) in layout.items())
        pinned=allocate_layout_views(layout,device=torch.device('cpu'),pin_memory=True)
        for key,value in values.items():
            pinned[key].copy_(value.detach());assert torch.equal(pinned[key],value)
            nbytes+=value.numel()*value.element_size()
        groups[signature][idx]=pinned
        del values
        block.to('meta')
    # Include duplicate names so shared embedding/output storage remains tied.
    moved={}
    for name,value in list(itertools.chain(model.named_parameters(remove_duplicate=False),model.named_buffers(remove_duplicate=False))):
        if name.startswith(prefix+'.'):continue
        if id(value) not in moved:moved[id(value)]=value.detach().to(target)
        assign_tensor_to_module(model,name,moved[id(value)])
    del moved;gc.collect()
    providers={};buffer_bytes=0;copy_stream=torch.cuda.Stream(device=target)
    for members in groups.values():
        source=PinnedWeightSource(members)
        pool=WeightPool(source.block_layout,capacity=slots,device=target,
            reuse_barrier=lambda event:copy_stream.wait_event(event))
        provider=WeightsProvider(pool,copy_stream,target,source)
        for idx in members:providers[idx]=provider
        buffer_bytes+=slots*sum(t.numel()*t.element_size() for t in next(iter(members.values())).values())
    wrapped=BlockStreamingWrapper(model,blocks,Providers(providers),target)
    wrapped._eval_stream_info=dict(blocks=len(blocks),layout_groups=len(groups),gpu_slots_per_layout=slots,
        cpu_block_bytes=nbytes,gpu_block_buffer_bytes=buffer_bytes,all_cpu_weight_copies_bitwise_equal=True,
        tensor_dtypes_preserved=True,lora_fusion=False,quantization=False,forward_unchanged=True)
    return wrapped


def text_conditioner(config,device='cuda:0'):
    from omni_av import ltx_runtime as native
    class StreamedText(native.NativeTextConditioner):
        def __init__(self):
            self.encoder,self.processor=native.load_text_modules(config,'cpu')
            self.encoder.model=stream(self.encoder.model,'model.language_model.layers',device)
            self.processor.to(device)
        def close(self):
            self.encoder.model.teardown()
            super().close()
    return StreamedText()


def render_loader(work,path,device='cuda:0',step=6000):
    from pathlib import Path
    import p2_all_lora6000_eval_20261109 as first
    if step==6000:loader=first
    else:
        import p2_all_lora_followups_eval_20261109 as loader
        loader.STEP=6500;loader.TRAIN=Path(work).parents[1]/'p2_ltx_all_lora_full28_batch28_20261109'
    model,strategy,sampler=loader.load_model(work,path,'cpu')
    model.renderer=stream(model.renderer,'transformer_blocks',device)
    model.connector.to(device);model.whisper_adapter.to(device)
    return model,strategy,sampler
