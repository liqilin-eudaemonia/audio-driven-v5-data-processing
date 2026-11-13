from dataclasses import dataclass, replace
import os

import torch

from ltx_core.distributed.comm import all_gather_dim
from ltx_core.distributed.parallel_state import get_sp_group, is_sp_enabled
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        self._video_config = video
        self._audio_config = audio
        self._rope_type = rope_type
        self._attention_function = attention_function
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
                sequence_parallel=True,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps
        self._monitor_omni_residuals = False
        self._omni_residual_metrics: dict[str, torch.Tensor] = {}

    def _record_omni_residual(
        self, stream: str, hidden: torch.Tensor, residual: torch.Tensor
    ) -> None:
        if not self._monitor_omni_residuals:
            return
        self._omni_residual_metrics[stream] = torch.stack(
            (
                residual.detach().float().square().sum(),
                hidden.detach().float().square().sum(),
            )
        )

    def enable_qwen_conditioning(
        self,
        qwen_context_dim: int,
        audio_qwen_context_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Add optional Qwen/Qwen-VL cross-attention branches after checkpoint loading."""
        if enable_video and self._video_config is not None and not hasattr(self, "qwen_attn2"):
            self.qwen_attn2 = self._create_qwen_attention(self._video_config, qwen_context_dim)
            self._zero_qwen_output(self.qwen_attn2)
            self.qwen_attn2.to(device=device, dtype=dtype)

        if enable_audio and self._audio_config is not None and not hasattr(self, "audio_qwen_attn2"):
            self.audio_qwen_attn2 = self._create_qwen_attention(
                self._audio_config,
                audio_qwen_context_dim or qwen_context_dim,
            )
            self._zero_qwen_output(self.audio_qwen_attn2)
            self.audio_qwen_attn2.to(device=device, dtype=dtype)

    def enable_omni_conditioning(
        self,
        omni_context_dim: int,
        audio_omni_context_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        enable_video: bool = True,
        enable_audio: bool = True,
    ) -> None:
        """Add independent Qwen3-Omni planner cross-attention branches."""
        if enable_video and self._video_config is not None and not hasattr(self, "omni_attn2"):
            self.omni_attn2 = self._create_condition_attention(self._video_config, omni_context_dim)
            self._zero_attention_output(self.omni_attn2)
            self.omni_attn2.to(device=device, dtype=dtype)

        if enable_audio and self._audio_config is not None and not hasattr(self, "audio_omni_attn2"):
            self.audio_omni_attn2 = self._create_condition_attention(
                self._audio_config,
                audio_omni_context_dim or omni_context_dim,
            )
            self._zero_attention_output(self.audio_omni_attn2)
            self.audio_omni_attn2.to(device=device, dtype=dtype)

    def enable_listener_timbre_conditioning(
        self,
        listener_timbre_context_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Add a listener-timbre cross-attention branch to the audio stream."""
        if self._audio_config is None:
            raise ValueError("Listener timbre conditioning requires an audio-enabled block")
        if not hasattr(self, "audio_timbre_attn2"):
            self.audio_timbre_attn2 = self._create_condition_attention(
                self._audio_config,
                listener_timbre_context_dim,
            )
            self._zero_attention_output(self.audio_timbre_attn2)
            self.audio_timbre_attn2.to(device=device, dtype=dtype)

    def enable_first_frame_audio_conditioning(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Add an optional first-frame cross-attention branch on the audio stream.

        The context tokens are patchified first-frame video latents projected to the
        audio inner dim by the model-level embedding (see LTXModel). The output
        projection starts at zero so the branch is a no-op until trained.
        """
        if self._audio_config is None:
            raise ValueError("First-frame audio conditioning requires an audio-enabled block")
        if not hasattr(self, "audio_ff_attn2"):
            self.audio_ff_attn2 = self._create_condition_attention(self._audio_config, self._audio_config.dim)
            self._zero_attention_output(self.audio_ff_attn2)
            self.audio_ff_attn2.to(device=device, dtype=dtype)

    def _create_condition_attention(self, config: TransformerConfig, context_dim: int) -> Attention:
        return Attention(
            query_dim=config.dim,
            context_dim=context_dim,
            heads=config.heads,
            dim_head=config.d_head,
            rope_type=self._rope_type,
            norm_eps=self.norm_eps,
            attention_function=self._attention_function,
            apply_gated_attention=config.apply_gated_attention,
        )

    @staticmethod
    def _zero_attention_output(attn: Attention) -> None:
        out_proj = attn.to_out[0]
        torch.nn.init.zeros_(out_proj.weight)
        if out_proj.bias is not None:
            torch.nn.init.zeros_(out_proj.bias)

    def _create_qwen_attention(self, config: TransformerConfig, context_dim: int) -> Attention:
        """Backward-compatible alias for the original Qwen helper."""
        return self._create_condition_attention(config, context_dim)

    @staticmethod
    def _zero_qwen_output(attn: Attention) -> None:
        """Backward-compatible alias for the original Qwen helper."""
        BasicAVTransformerBlock._zero_attention_output(attn)

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        finite_debug = os.environ.get("LTX_CP_FINITE_DEBUG") == "1"

        def check_finite(stage: str) -> None:
            if not finite_debug:
                return
            for modality_name, value in (("video", vx), ("audio", ax)):
                if value is not None and not torch.isfinite(value).all():
                    nonfinite = int((~torch.isfinite(value)).sum().item())
                    raise FloatingPointError(
                        f"Non-finite {modality_name} activation after {stage} "
                        f"in transformer block {self.idx}: {nonfinite}/{value.numel()}"
                    )

        check_finite("input")

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa

            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            v_mask = (
                perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                if not all_perturbed and not none_perturbed
                else None
            )
            vx = (
                vx
                + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                    perturbation_mask=v_mask,
                    all_perturbed=all_perturbed,
                )
                * vgate_msa
            )
            del vgate_msa, norm_vx, v_mask
            vx = vx + self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if video.qwen_context is not None:
                if not hasattr(self, "qwen_attn2"):
                    raise ValueError("Received video qwen_context, but Qwen conditioning is not enabled")
                vx = vx + self.qwen_attn2(
                    rms_norm(vx, eps=self.norm_eps),
                    context=video.qwen_context,
                    mask=video.qwen_context_mask,
                    pe=video.qwen_query_pe,
                    k_pe=video.qwen_key_pe,
                )
            if video.omni_context is not None:
                if not hasattr(self, "omni_attn2"):
                    raise ValueError("Received video omni_context, but Omni conditioning is not enabled")
                omni_residual = self.omni_attn2(
                    rms_norm(vx, eps=self.norm_eps),
                    context=video.omni_context,
                    mask=video.omni_context_mask,
                )
                self._record_omni_residual("video", vx, omni_residual)
                vx = vx + omni_residual
            check_finite("video attention branches")

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            a_mask = (
                perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if not all_perturbed and not none_perturbed
                else None
            )
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                    perturbation_mask=a_mask,
                    all_perturbed=all_perturbed,
                )
                * agate_msa
            )
            del agate_msa, norm_ax, a_mask
            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if audio.qwen_context is not None:
                if not hasattr(self, "audio_qwen_attn2"):
                    raise ValueError("Received audio qwen_context, but Qwen conditioning is not enabled")
                ax = ax + self.audio_qwen_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.qwen_context,
                    mask=audio.qwen_context_mask,
                    pe=audio.qwen_query_pe,
                    k_pe=audio.qwen_key_pe,
                )
            if audio.omni_context is not None:
                if not hasattr(self, "audio_omni_attn2"):
                    raise ValueError("Received audio omni_context, but Omni conditioning is not enabled")
                omni_residual = self.audio_omni_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.omni_context,
                    mask=audio.omni_context_mask,
                )
                self._record_omni_residual("audio", ax, omni_residual)
                ax = ax + omni_residual
            if audio.listener_timbre_context is not None:
                if not hasattr(self, "audio_timbre_attn2"):
                    raise ValueError(
                        "Received listener_timbre_context, but listener timbre conditioning is not enabled"
                    )
                ax = ax + self.audio_timbre_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.listener_timbre_context,
                )
            if audio.first_frame_context is not None:
                if not hasattr(self, "audio_ff_attn2"):
                    raise ValueError(
                        "Received audio first_frame_context, but first-frame audio conditioning is not enabled"
                    )
                ax = ax + self.audio_ff_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.first_frame_context,
                )
            check_finite("audio attention branches")

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )
                del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)

                if is_sp_enabled():
                    vx_for_ca = all_gather_dim(vx_scaled, dim=1, group=get_sp_group())
                    if video.cross_positional_embeddings is not None:
                        k_pe = (
                            all_gather_dim(video.cross_positional_embeddings[0], dim=2, group=get_sp_group()),
                            all_gather_dim(video.cross_positional_embeddings[1], dim=2, group=get_sp_group()),
                        )
                    else:
                        k_pe = None
                else:
                    vx_for_ca = vx_scaled
                    k_pe = video.cross_positional_embeddings

                # Compact CP padding is excluded from video-to-audio keys, just as
                # it is from video self-attention in Ulysses.
                if (
                    video.self_attention_mask is not None
                    and video.self_attention_mask.ndim == 4
                    and video.self_attention_mask.shape[-2] == 1
                ):
                    valid_counts = (
                        video.self_attention_mask[:, 0, 0, :] == 0
                    ).sum(dim=-1)
                    if not (valid_counts == valid_counts[0]).all():
                        raise ValueError(
                            "All samples in a CP batch must have the same padded sequence length"
                        )
                    valid_len = int(valid_counts[0].item())
                    vx_for_ca = vx_for_ca[:, :valid_len]
                    if k_pe is not None:
                        k_pe = (
                            k_pe[0][:, :, :valid_len, :],
                            k_pe[1][:, :, :valid_len, :],
                        )

                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_for_ca,
                        pe=audio.cross_positional_embeddings,
                        k_pe=k_pe,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )
                del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled

            del vx_norm3, ax_norm3
            check_finite("audio-video cross attention")

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        check_finite("feed-forward")

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
