"""Opt-in execution changes; objectives, trainables and sample order stay identical."""
from types import MethodType
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def _selective_blocks(self, video, audio, perturbations):
    for index, block in enumerate(self.transformer_blocks):
        selected = index % 2 == 0 if self._p2_checkpoint_policy == "alternating" else index % 4 != 3
        if selected and self.training:
            video, audio = checkpoint(block, video, audio, perturbations, use_reentrant=False)
        else:
            video, audio = block(video=video, audio=audio, perturbations=perturbations)
    return video, audio


def configure_renderer(renderer, policy):
    """Use native non-reentrant checkpointing on all or a fixed subset of blocks.

    The pinned LTX core exposes an all-block switch only. Override this one
    instance's block traversal; block objects, state keys and LoRA paths stay intact.
    Never modify the shared LTX package or affect other renderer instances.
    """
    if policy not in ("all", "alternating", "three_quarters", "none"):
        raise ValueError("Unsupported gradient checkpoint policy")
    if not hasattr(renderer, "set_gradient_checkpointing") or not hasattr(renderer, "_process_transformer_blocks"):
        raise ValueError("Renderer lacks the pinned LTX checkpoint API")
    if not hasattr(renderer, "_p2_original_process_blocks"):
        object.__setattr__(renderer, "_p2_original_process_blocks", renderer._process_transformer_blocks)
    renderer.set_gradient_checkpointing(policy != "none")
    renderer._p2_checkpoint_policy = policy
    renderer._process_transformer_blocks = (MethodType(_selective_blocks, renderer)
        if policy in ("alternating", "three_quarters") else renderer._p2_original_process_blocks)


class BucketGradientAccumulator:
    """FP64 accumulation and bounded SUM collectives; round to each dtype once.

    Parameters must share one device. Mixed dense parameter dtypes are supported.
    Presence is reduced separately so an unused parameter keeps grad=None.
    """
    def __init__(self, parameters, bucket_mb=32):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters to accumulate")
        if type(bucket_mb) is not int or bucket_mb < 1:
            raise ValueError("gradient_bucket_mb must be a positive integer")
        device = self.parameters[0].device
        if any(p.device != device for p in self.parameters):
            raise ValueError("Gradient parameters must share a device")
        self.sizes = [p.numel() for p in self.parameters]
        self.buffer = torch.zeros(sum(self.sizes), dtype=torch.float64, device=device)
        self.views = list(self.buffer.split(self.sizes))
        self.present = [False] * len(self.parameters)
        self.bucket_elements = bucket_mb * 1024 * 1024 // self.buffer.element_size()

    def add(self):
        destinations, sources = [], []
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is not None:
                if parameter.grad.is_sparse:
                    raise ValueError("P2 gradient accumulation requires dense gradients")
                destinations.append(self.views[index])
                sources.append(parameter.grad.detach().reshape(-1))
                self.present[index] = True
        if destinations:
            torch._foreach_add_(destinations, sources)
        for parameter in self.parameters:
            parameter.grad = None

    def finish(self, world=1):
        if world < 1 or (world > 1 and (not dist.is_initialized() or dist.get_world_size() != world)):
            raise ValueError("Gradient world size differs from initialized process group")
        present = torch.tensor(self.present, dtype=torch.int32, device=self.buffer.device)
        if world > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
            for chunk in self.buffer.split(self.bucket_elements):
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
        # Each dtype has one rounded backing tensor; views must not alias FP64
        # accumulation storage, which is cleared for safe reuse after finish().
        rounded = {dtype: self.buffer.to(dtype, copy=True) for dtype in {p.dtype for p in self.parameters}}
        split = {dtype: value.split(self.sizes) for dtype, value in rounded.items()}
        for index, (parameter, used) in enumerate(zip(self.parameters, present.tolist())):
            parameter.grad = split[parameter.dtype][index].view_as(parameter) if used else None
        self.buffer.zero_()
        self.present = [False] * len(self.parameters)
