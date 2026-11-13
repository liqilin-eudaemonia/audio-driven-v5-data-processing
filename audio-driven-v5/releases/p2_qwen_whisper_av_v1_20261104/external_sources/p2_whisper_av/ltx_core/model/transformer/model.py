import dataclasses
from enum import Enum

import torch
from torch.nn.utils import parametrize

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class QwenLayerAdapter(torch.nn.Module):
    """Fuse stacked Qwen decoder-layer features into one cross-attention context."""

    def __init__(self, *, num_layers: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.layer_logits = torch.nn.Parameter(torch.zeros(self.num_layers))
        self.norm = torch.nn.LayerNorm(self.input_dim)
        self.proj = torch.nn.Linear(self.input_dim, self.output_dim)
        if self.input_dim == self.output_dim:
            torch.nn.init.eye_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4:
            raise ValueError(f"QwenLayerAdapter expects [B, layers, seq, hidden], got {context.shape}")
        if context.shape[1] != self.num_layers:
            raise ValueError(f"Qwen layer count mismatch: got {context.shape[1]}, expected {self.num_layers}")
        if context.shape[-1] != self.input_dim:
            raise ValueError(f"Qwen hidden size mismatch: got {context.shape[-1]}, expected {self.input_dim}")
        weights = torch.softmax(self.layer_logits.float(), dim=0).to(dtype=context.dtype, device=context.device)
        fused = torch.einsum("l,blsh->bsh", weights, context)
        return self.proj(self.norm(fused))


class OmniLayerAdapter(torch.nn.Module):
    """Fuse stacked Qwen3-Omni planner layers into one conditioning context."""

    def __init__(self, *, num_layers: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.layer_logits = torch.nn.Parameter(torch.zeros(self.num_layers))
        self.norm = torch.nn.LayerNorm(self.input_dim)
        self.proj = torch.nn.Linear(self.input_dim, self.output_dim)
        if self.input_dim == self.output_dim:
            torch.nn.init.eye_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4:
            raise ValueError(f"OmniLayerAdapter expects [B, layers, seq, hidden], got {context.shape}")
        if context.shape[1] != self.num_layers:
            raise ValueError(f"Omni layer count mismatch: got {context.shape[1]}, expected {self.num_layers}")
        if context.shape[-1] != self.input_dim:
            raise ValueError(f"Omni hidden size mismatch: got {context.shape[-1]}, expected {self.input_dim}")
        weights = torch.softmax(self.layer_logits.float(), dim=0).to(dtype=context.dtype, device=context.device)
        self._last_layer_weights = weights.detach()
        fused = torch.einsum("l,blsh->bsh", weights, context)
        return self.proj(self.norm(fused))


class LowRankWeightDelta(torch.nn.Module):
    """Zero-init low-rank delta over a frozen two-dimensional weight."""

    def __init__(self, weight: torch.Tensor, *, rank: int, alpha: int) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
        self.scaling = float(alpha) / float(rank)
        self.lora_A = torch.nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.lora_B = torch.nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return base + (self.lora_B @ self.lora_A) * self.scaling


class FusedQKVLowRankWeightDelta(torch.nn.Module):
    """Independent Q/K/V low-rank deltas for MultiheadAttention.in_proj_weight."""

    def __init__(self, weight: torch.Tensor, *, rank: int, alpha: int) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] != 3 * weight.shape[1]:
            raise ValueError(f"Expected fused square QKV weight, got {tuple(weight.shape)}")
        hidden = weight.shape[1]
        self.scaling = float(alpha) / float(rank)
        self.lora_A = torch.nn.Parameter(weight.new_empty(3, rank, hidden))
        self.lora_B = torch.nn.Parameter(weight.new_zeros(3, hidden, rank))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        delta = torch.bmm(self.lora_B, self.lora_A).reshape_as(base)
        return base + delta * self.scaling


class FieldAwareOmniResampler(torch.nn.Module):
    """Trainable pooling from raw multi-layer Planner tokens to 44 compact tokens."""

    FIELD_QUERY_COUNTS = (8, 16, 16)

    def __init__(
        self,
        *,
        num_layers: int,
        input_dim: int = 2048,
        output_dim: int = 1024,
        num_heads: int = 8,
        num_global_queries: int = 4,
    ) -> None:
        super().__init__()
        if output_dim % num_heads:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_layers = int(num_layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_global_queries = int(num_global_queries)
        self.field_queries = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(count, output_dim) * 0.02)
                for count in self.FIELD_QUERY_COUNTS
            ]
        )
        self.global_queries = torch.nn.Parameter(
            torch.randn(num_global_queries, output_dim) * 0.02
        )
        self.proj_in = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(output_dim)

    @property
    def output_sequence_length(self) -> int:
        return sum(self.FIELD_QUERY_COUNTS) + self.num_global_queries

    @property
    def stage2_lora_enabled(self) -> bool:
        return parametrize.is_parametrized(self.proj_in, "weight")

    def enable_stage2_lora(self, *, rank: int = 32, alpha: int = 32) -> None:
        """Freeze learned Stage-1 matrices and add zero-init Stage-2 deltas."""
        if self.stage2_lora_enabled:
            raise ValueError("Omni Resampler Stage-2 LoRA is already enabled")
        self.requires_grad_(False)
        parametrize.register_parametrization(
            self.proj_in,
            "weight",
            LowRankWeightDelta(self.proj_in.weight, rank=rank, alpha=alpha),
        )
        parametrize.register_parametrization(
            self.attn,
            "in_proj_weight",
            FusedQKVLowRankWeightDelta(self.attn.in_proj_weight, rank=rank, alpha=alpha),
        )
        parametrize.register_parametrization(
            self.attn.out_proj,
            "weight",
            LowRankWeightDelta(self.attn.out_proj.weight, rank=rank, alpha=alpha),
        )
        for parameter in self.field_queries.parameters():
            parameter.requires_grad_(True)
        self.global_queries.requires_grad_(True)
        self.norm.requires_grad_(True)

    def _pool(
        self,
        hidden: torch.Tensor,
        queries: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, layers, seq_len, hidden_dim = hidden.shape
        flat_hidden = hidden.reshape(batch * layers, seq_len, hidden_dim)
        flat_mask = (
            valid_mask[:, None, :]
            .expand(batch, layers, seq_len)
            .reshape(batch * layers, seq_len)
        )
        if not bool(flat_mask.any(dim=1).all()):
            raise ValueError("Every Omni resampler field must contain at least one valid token")
        query = (
            queries[None, None]
            .expand(batch, layers, -1, -1)
            .reshape(batch * layers, queries.shape[0], hidden_dim)
        )
        pooled, _ = self.attn(
            query,
            flat_hidden,
            flat_hidden,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        return self.norm(query + pooled).reshape(
            batch, layers, queries.shape[0], hidden_dim
        )

    def forward(
        self,
        context: torch.Tensor,
        token_mask: torch.Tensor,
        field_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 4:
            raise ValueError(
                "FieldAwareOmniResampler expects [B,layers,seq,hidden], "
                f"got {tuple(context.shape)}"
            )
        batch, layers, seq_len, hidden = context.shape
        if layers != self.num_layers or hidden != self.input_dim:
            raise ValueError(
                f"Expected [B,{self.num_layers},T,{self.input_dim}], "
                f"got {tuple(context.shape)}"
            )
        if token_mask.shape != (batch, seq_len):
            raise ValueError(
                f"Omni token mask must have shape {(batch, seq_len)}, "
                f"got {tuple(token_mask.shape)}"
            )
        if field_mask.shape != (batch, len(self.FIELD_QUERY_COUNTS), seq_len):
            raise ValueError(
                "Omni field mask must have shape "
                f"[B,{len(self.FIELD_QUERY_COUNTS)},T], got {tuple(field_mask.shape)}"
            )

        projected = self.proj_in(context)
        token_mask = token_mask.to(device=context.device, dtype=torch.bool)
        field_mask = field_mask.to(device=context.device, dtype=torch.bool)
        outputs = []
        for index, queries in enumerate(self.field_queries):
            valid_field_mask = token_mask & field_mask[:, index]
            outputs.append(self._pool(projected, queries, valid_field_mask))
        outputs.append(self._pool(projected, self.global_queries, token_mask))
        return torch.cat(outputs, dim=2)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def enable_qwen_conditioning(
        self,
        qwen_context_dim: int,
        audio_qwen_context_dim: int | None = None,
        *,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Enable optional Qwen/Qwen-VL condition branches after loading base weights."""
        parameter = next(self.parameters())
        for block in self.transformer_blocks:
            block.enable_qwen_conditioning(
                qwen_context_dim=qwen_context_dim,
                audio_qwen_context_dim=audio_qwen_context_dim,
                device=parameter.device,
                dtype=parameter.dtype,
                enable_video=enable_video,
                enable_audio=enable_audio,
            )

    def enable_qwen_layer_adapters(
        self,
        *,
        num_layers: int,
        input_dim: int,
        video_output_dim: int,
        audio_output_dim: int | None = None,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Enable trainable per-stream adapters for stacked Qwen decoder-layer features."""
        parameter = next(self.parameters())
        if enable_video and self.model_type.is_video_enabled() and not hasattr(self, "video_qwen_layer_adapter"):
            self.video_qwen_layer_adapter = QwenLayerAdapter(
                num_layers=num_layers,
                input_dim=input_dim,
                output_dim=video_output_dim,
            )
            self.video_qwen_layer_adapter.to(device=parameter.device, dtype=parameter.dtype)
        if enable_audio and self.model_type.is_audio_enabled() and not hasattr(self, "audio_qwen_layer_adapter"):
            self.audio_qwen_layer_adapter = QwenLayerAdapter(
                num_layers=num_layers,
                input_dim=input_dim,
                output_dim=audio_output_dim or video_output_dim,
            )
            self.audio_qwen_layer_adapter.to(device=parameter.device, dtype=parameter.dtype)

    def enable_omni_conditioning(
        self,
        omni_context_dim: int = 1024,
        audio_omni_context_dim: int | None = None,
        *,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Enable independent Qwen3-Omni planner branches after loading base weights."""
        parameter = next(self.parameters())
        for block in self.transformer_blocks:
            block.enable_omni_conditioning(
                omni_context_dim=omni_context_dim,
                audio_omni_context_dim=audio_omni_context_dim,
                device=parameter.device,
                dtype=parameter.dtype,
                enable_video=enable_video,
                enable_audio=enable_audio,
            )

    def enable_omni_layer_adapters(
        self,
        *,
        num_layers: int,
        input_dim: int = 1024,
        video_output_dim: int = 1024,
        audio_output_dim: int | None = None,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Enable per-stream adapters for stacked Qwen3-Omni planner layers."""
        parameter = next(self.parameters())
        if enable_video and self.model_type.is_video_enabled() and not hasattr(self, "video_omni_layer_adapter"):
            self.video_omni_layer_adapter = OmniLayerAdapter(
                num_layers=num_layers,
                input_dim=input_dim,
                output_dim=video_output_dim,
            )
            self.video_omni_layer_adapter.to(device=parameter.device, dtype=parameter.dtype)
        if enable_audio and self.model_type.is_audio_enabled() and not hasattr(self, "audio_omni_layer_adapter"):
            self.audio_omni_layer_adapter = OmniLayerAdapter(
                num_layers=num_layers,
                input_dim=input_dim,
                output_dim=audio_output_dim or video_output_dim,
            )
            self.audio_omni_layer_adapter.to(device=parameter.device, dtype=parameter.dtype)

    def enable_omni_field_resampler(
        self,
        *,
        num_layers: int,
        input_dim: int = 2048,
        output_dim: int = 1024,
        num_heads: int = 8,
    ) -> None:
        """Enable the shared trainable raw-Planner-token resampler."""
        if hasattr(self, "omni_field_resampler"):
            return
        parameter = next(self.parameters())
        self.omni_field_resampler = FieldAwareOmniResampler(
            num_layers=num_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
        )
        self.omni_field_resampler.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def enable_listener_timbre_conditioning(self, listener_timbre_context_dim: int = 1024) -> None:
        """Enable the listener-timbre condition branch on the audio stream only."""
        if not self.model_type.is_audio_enabled():
            raise ValueError("Listener timbre conditioning requires an audio-enabled model")
        parameter = next(self.parameters())
        for block in self.transformer_blocks:
            block.enable_listener_timbre_conditioning(
                listener_timbre_context_dim=listener_timbre_context_dim,
                device=parameter.device,
                dtype=parameter.dtype,
            )

    def enable_first_frame_audio_conditioning(self, first_frame_in_channels: int | None = None) -> None:
        """Enable the optional first-frame condition branch on the audio stream.

        Adds a Conv3d patch embedding (video latent channels -> audio inner dim,
        kernel/stride [1, 2, 2], mirroring the UniAV first-frame patchifier) plus a
        zero-initialized cross-attention branch in every block. Must be called after
        loading base weights.
        """
        if not self.model_type.is_audio_enabled():
            raise ValueError("First-frame audio conditioning requires an audio-enabled model")
        parameter = next(self.parameters())
        if not hasattr(self, "audio_first_frame_patch_embedding"):
            if first_frame_in_channels is None:
                if not self.model_type.is_video_enabled():
                    raise ValueError("first_frame_in_channels is required for audio-only models")
                first_frame_in_channels = self.patchify_proj.in_features
            self.audio_first_frame_patch_embedding = torch.nn.Conv3d(
                first_frame_in_channels,
                self.audio_inner_dim,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            )
            torch.nn.init.xavier_uniform_(self.audio_first_frame_patch_embedding.weight.flatten(1))
            torch.nn.init.zeros_(self.audio_first_frame_patch_embedding.bias)
            self.audio_first_frame_patch_embedding.to(device=parameter.device, dtype=parameter.dtype)
        for block in self.transformer_blocks:
            block.enable_first_frame_audio_conditioning(device=parameter.device, dtype=parameter.dtype)

    def _prepare_first_frame_context(self, audio: Modality) -> torch.Tensor | None:
        """Patchify the raw first-frame video latent into audio-dim context tokens."""
        if audio.first_frame_latent is None:
            return None
        if not hasattr(self, "audio_first_frame_patch_embedding"):
            raise ValueError(
                "Received first_frame_latent, but first-frame audio conditioning is not enabled"
            )
        ff = audio.first_frame_latent.to(dtype=self.audio_first_frame_patch_embedding.weight.dtype)
        ff = self.audio_first_frame_patch_embedding(ff)  # (B, D, 1, H/2, W/2)
        return ff.flatten(2).transpose(1, 2)  # (B, T_ff, D)

    def _prepare_qwen_context(self, modality: Modality, *, is_audio: bool) -> Modality:
        """Apply the stream-specific all-layer Qwen adapter before args preprocessing."""
        if modality.qwen_context is None or modality.qwen_context.ndim != 4:
            return modality
        adapter_name = "audio_qwen_layer_adapter" if is_audio else "video_qwen_layer_adapter"
        if not hasattr(self, adapter_name):
            raise ValueError(
                "Received stacked Qwen layer features, but Qwen layer adapters are not enabled"
            )
        adapter = getattr(self, adapter_name)
        context = modality.qwen_context.to(dtype=next(adapter.parameters()).dtype)
        return dataclasses.replace(modality, qwen_context=adapter(context))

    def _resample_omni_context(self, modality: Modality) -> Modality:
        """Compress raw Omni tokens while preserving the four-layer axis."""
        if modality.omni_context is None or modality.omni_context.ndim != 4:
            return modality
        if hasattr(self, "omni_field_resampler"):
            resampler = self.omni_field_resampler
            if modality.omni_context.shape[-1] == resampler.input_dim:
                if modality.omni_context_mask is None or modality.omni_field_mask is None:
                    raise ValueError(
                        "Raw Omni context requires token and field masks"
                    )
                raw_context = modality.omni_context.to(
                    dtype=next(resampler.parameters()).dtype
                )
                context = resampler(
                    raw_context,
                    modality.omni_context_mask,
                    modality.omni_field_mask,
                )
                mask = torch.ones(
                    context.shape[0],
                    context.shape[2],
                    dtype=torch.bool,
                    device=context.device,
                )
                modality = dataclasses.replace(
                    modality,
                    omni_context=context,
                    omni_context_mask=mask,
                    omni_field_mask=None,
                )
        return modality

    def _prepare_omni_context(self, modality: Modality, *, is_audio: bool) -> Modality:
        """Resample raw tokens, then apply the stream-specific layer adapter."""
        modality = self._resample_omni_context(modality)
        if modality.omni_context is None or modality.omni_context.ndim != 4:
            return modality
        adapter_name = "audio_omni_layer_adapter" if is_audio else "video_omni_layer_adapter"
        if not hasattr(self, adapter_name):
            raise ValueError(
                "Received stacked Omni layer features, but Omni layer adapters are not enabled"
            )
        adapter = getattr(self, adapter_name)
        context = modality.omni_context.to(dtype=next(adapter.parameters()).dtype)
        return dataclasses.replace(modality, omni_context=adapter(context))

    def set_omni_residual_monitoring(
        self, enabled: bool, block_indices: set[int] | None = None
    ) -> None:
        """Toggle detached Omni residual measurements on selected transformer blocks."""
        selected = block_indices or set()
        for block in self.transformer_blocks:
            block._monitor_omni_residuals = enabled and block.idx in selected
            block._omni_residual_metrics.clear()

    def pop_omni_residual_metrics(self) -> dict[str, torch.Tensor]:
        """Return and clear the latest representative-block Omni residual ratios."""
        metrics: dict[str, torch.Tensor] = {}
        for block in self.transformer_blocks:
            for stream, value in block._omni_residual_metrics.items():
                metrics[f"{stream}_omni_residual_ratio/block_{block.idx}"] = value
            block._omni_residual_metrics.clear()
            block._monitor_omni_residuals = False
        return metrics

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video = self._prepare_qwen_context(video, is_audio=False) if video is not None else None
        audio = self._prepare_qwen_context(audio, is_audio=True) if audio is not None else None
        if (
            video is not None
            and audio is not None
            and video.omni_context is not None
            and video.omni_context is audio.omni_context
        ):
            video = self._resample_omni_context(video)
            audio = dataclasses.replace(
                audio,
                omni_context=video.omni_context,
                omni_context_mask=video.omni_context_mask,
                omni_field_mask=video.omni_field_mask,
            )
        video = self._prepare_omni_context(video, is_audio=False) if video is not None else None
        audio = self._prepare_omni_context(audio, is_audio=True) if audio is not None else None
        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        if audio is not None and audio.first_frame_latent is not None:
            first_frame_context = self._prepare_first_frame_context(audio)
            audio_args = dataclasses.replace(audio_args, first_frame_context=first_frame_context)
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
