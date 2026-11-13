"""Online condition encoding for LTX training: Gemma text + frozen Qwen3-VL first-frame features.

The Qwen processing mirrors UniAV's `qwen_utils/encode_first_frame_dialogue.py` contract:
  - first frame + instruction + full ``sd_text`` through the chat template (right padding),
  - hidden states captured at the decoder block output BEFORE the final RMSNorm,
  - right truncation / right padding to a fixed sequence length,
  - hidden states at padded positions cleared to zero.

Deviations from UniAV (by design):
  - The image canvas equals the VAE input resolution (e.g. 1088x640) and the crop uses the
    same aspect-preserving bicubic resize + center crop as ``scripts/process_videos.py``, so
    Qwen sees the exact framing the video VAE encoded (UniAV instead crops first, then
    resizes to a 1280x720-derived canvas).
  - By default only the LAST decoder layer is kept. Experimental all-layer mode keeps
    every decoder layer for trainable LTX-side adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from ltx_trainer import logger

if TYPE_CHECKING:
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor

# Same instruction text as UniAV qwen_utils/encode_first_frame_dialogue.py::DEFAULT_INSTRUCTION.
QWEN_INSTRUCTION = (
    "Analyze the first frame of the video together with its full structured text prompt. "
    "Pay attention to the people, scene, dialogue, actions, expressions, interactions, "
    "environment, and audio description."
)

QWEN_IMAGE_FACTOR = 32  # patch_size (16) * merge_size (2) for Qwen3-VL image tokenization


# ---------------------------------------------------------------------------
# First-frame reading + VAE-aligned resize
# ---------------------------------------------------------------------------


def read_first_frame(video_path: str | Path) -> Image.Image:
    """Read the first frame of a video (or an image file), decord first, cv2 fallback.

    Mirrors UniAV qwen_utils/describe_first_frames.py::read_first_frame.
    """
    path = Path(video_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        with Image.open(path) as image:
            return image.convert("RGB")
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        if len(reader) == 0:
            raise ValueError("video has no frames")
        return Image.fromarray(reader[0].asnumpy()).convert("RGB")
    except Exception as decord_error:  # noqa: BLE001
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError("cv2 could not read the first frame")
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            finally:
                cap.release()
        except Exception as cv2_error:
            raise RuntimeError(
                f"failed to read first frame of {path} with decord ({decord_error}) and cv2 ({cv2_error})"
            ) from cv2_error


def resize_and_center_crop_to_vae_frame(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Aspect-preserving bicubic resize + center crop, matching process_videos._resize_and_crop.

    This reproduces the exact framing that the video VAE encoder saw during dataset
    preprocessing (reshape_mode="center"), so the Qwen condition is pixel-aligned
    with the VAE input.
    """
    if target_width % QWEN_IMAGE_FACTOR or target_height % QWEN_IMAGE_FACTOR:
        raise ValueError(
            f"Qwen canvas {target_width}x{target_height} must be divisible by {QWEN_IMAGE_FACTOR} "
            "to avoid processor-side re-resizing"
        )

    tensor = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).unsqueeze(0).float()
    current_height, current_width = tensor.shape[2], tensor.shape[3]

    current_aspect = current_width / current_height
    target_aspect = target_width / target_height
    if current_aspect > target_aspect:
        new_width = int(current_width * target_height / current_height)
        tensor = resize(tensor, size=[target_height, new_width], interpolation=InterpolationMode.BICUBIC)
    else:
        new_height = int(current_height * target_width / current_width)
        tensor = resize(tensor, size=[new_height, target_width], interpolation=InterpolationMode.BICUBIC)

    current_height, current_width = tensor.shape[2], tensor.shape[3]
    top = (current_height - target_height) // 2
    left = (current_width - target_width) // 2
    tensor = tensor[:, :, top : top + target_height, left : left + target_width]

    array = tensor.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(array)


def load_first_frame_for_qwen(video_path: str | Path, target_width: int, target_height: int) -> Tensor:
    """Read + VAE-align the first frame, returned as a uint8 HWC tensor (collate-friendly)."""
    image = resize_and_center_crop_to_vae_frame(read_first_frame(video_path), target_width, target_height)
    return torch.from_numpy(np.array(image, copy=True))


# ---------------------------------------------------------------------------
# Online Qwen3-VL encoder
# ---------------------------------------------------------------------------


@dataclass
class QwenBatchEncoding:
    context: Tensor  # [B, max_length, hidden] or [B, layers, max_length, hidden], pads zeroed
    attention_mask: Tensor  # [B, max_length] long, 1 = valid


class OnlineQwenEncoder:
    """Qwen3-VL feature extractor, optionally trainable through Stage-2 language LoRA."""

    def __init__(
        self,
        model_path: str,
        device: torch.device | str,
        *,
        max_length: int = 2560,
        image_width: int = 1088,
        image_height: int = 640,
        dtype: torch.dtype = torch.bfloat16,
        instruction: str = QWEN_INSTRUCTION,
        feature_layer_mode: str = "last",
        train_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 36,
        lora_target_modules: list[str] | None = None,
        gradient_checkpointing: bool = True,
        dynamic_length: bool = False,
        lora_checkpoint: str | Path | None = None,
    ) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as _QwenModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as _QwenModelClass

        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.instruction = instruction
        self.train_lora = bool(train_lora)
        self.dynamic_length = bool(dynamic_length)
        if feature_layer_mode not in {"last", "all"}:
            raise ValueError(f"Unsupported Qwen feature_layer_mode: {feature_layer_mode}")
        self.feature_layer_mode = feature_layer_mode

        logger.info(f"Loading online Qwen3-VL encoder from {model_path}...")
        model = _QwenModelClass.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.qwen_lora_targets: list[str] = []
        use_lora = self.train_lora or lora_checkpoint is not None
        if use_lora:
            from ltx_trainer.stage2_lora import apply_qwen_lora, load_qwen_lora

            model, self.qwen_lora_targets = apply_qwen_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                last_n_layers=lora_last_n_layers,
                target_leaf_names=lora_target_modules
                or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            if lora_checkpoint:
                load_qwen_lora(model, str(lora_checkpoint))

        model = model.to(self.device)
        if self.train_lora:
            model.train()
            base = model.get_base_model()
            base.config.use_cache = False
            if gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
        else:
            model.eval()
            model.requires_grad_(False)
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "right"

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            size = dict(getattr(image_processor, "size") or {})
            size["longest_edge"] = self.image_width * self.image_height
            image_processor.size = size
        self._bind_backbone_refs()

    def _bind_backbone_refs(self) -> None:
        from ltx_trainer.stage2_lora import get_qwen_base_model

        base_model = getattr(get_qwen_base_model(self.model), "model", None)
        if base_model is None or not hasattr(base_model, "language_model"):
            raise TypeError("Expected a Qwen3-VL model exposing model.language_model.layers")
        self.base_model = base_model
        self.base_model.visual.eval()
        layers = list(base_model.language_model.layers)
        if not layers:
            raise ValueError("Qwen3-VL language model has no decoder layers")
        self.layers = layers
        self.last_layer = layers[-1]
        self.hidden_size = int(get_qwen_base_model(self.model).config.text_config.hidden_size)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _build_messages(self, image: Image.Image, prompt_text: str) -> list[dict]:
        text = f"{self.instruction.strip()}\n\n{prompt_text.strip()}".strip()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def encode_batch(
        self,
        images: list[Image.Image | Tensor],
        texts: list[str],
        *,
        require_grad: bool | None = None,
    ) -> QwenBatchEncoding:
        """Encode first frames and structured text, preserving autograd for Stage-2 LoRA."""
        use_grad = self.train_lora if require_grad is None else bool(require_grad)
        if use_grad and not self.train_lora:
            raise ValueError("require_grad=True needs a trainable Qwen LoRA encoder")

        if len(images) != len(texts):
            raise ValueError(f"images/texts length mismatch: {len(images)} vs {len(texts)}")

        pil_images = [
            Image.fromarray(img.cpu().numpy()) if isinstance(img, Tensor) else img
            for img in images
        ]
        messages = [
            self._build_messages(img, text)
            for img, text in zip(pil_images, texts, strict=True)
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)
        input_length = int(inputs["input_ids"].shape[1])
        if use_grad and input_length > self.max_length:
            raise ValueError(
                f"Stage-2 Qwen input length {input_length} exceeds hard limit {self.max_length}; "
                "the sample must be filtered before training"
            )

        layers_to_capture = self.layers if self.feature_layer_mode == "all" else [self.last_layer]
        captured: list[Tensor] = []
        handles = []

        def _capture(_module, _args, output) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden)

        grad_context = torch.enable_grad() if use_grad else torch.no_grad()
        try:
            handles = [layer.register_forward_hook(_capture) for layer in layers_to_capture]
            with grad_context:
                self.base_model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        expected_layers = len(layers_to_capture)
        if len(captured) != expected_layers:
            raise RuntimeError(
                f"Failed to capture Qwen hidden states: got {len(captured)}, expected {expected_layers}"
            )

        if self.feature_layer_mode == "all":
            hidden_states = torch.stack(captured, dim=1)
        else:
            hidden_states = captured[-1]
        if not use_grad:
            hidden_states = hidden_states.detach()
        attention_mask = inputs["attention_mask"].detach()

        current_length = hidden_states.shape[-2]
        if current_length > self.max_length:
            logger.warning(
                f"Qwen sequence length {current_length} exceeds max_length {self.max_length}; right-truncating"
            )
            hidden_states = hidden_states[..., : self.max_length, :]
            attention_mask = attention_mask[:, : self.max_length]
        elif current_length < self.max_length and not self.dynamic_length:
            pad = self.max_length - current_length
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
            attention_mask = F.pad(attention_mask, (0, pad), value=0)

        attention_mask = attention_mask.to(dtype=torch.long)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states * attention_mask[:, None, :, None].to(hidden_states.dtype)
        else:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return QwenBatchEncoding(context=hidden_states, attention_mask=attention_mask)


# ---------------------------------------------------------------------------
# Online Gemma text encoding
# ---------------------------------------------------------------------------


# no_grad, NOT inference_mode: the embeddings feed cross-attention whose adaln
# params train in full-SFT mode; inference tensors cannot be saved for backward.
@torch.no_grad()
def encode_text_batch_online(
    text_encoder: torch.nn.Module,
    embeddings_processor: "EmbeddingsProcessor",
    texts: list[str],
) -> dict[str, Tensor]:
    """Run the full Gemma pipeline (LLM -> feature extractor -> connectors) for a batch.

    Returns final per-modality embeddings, i.e. the same tensors the offline path
    produces AFTER EmbeddingsProcessor.create_embeddings — callers must not re-apply
    the connectors.
    """
    video_encodings: list[Tensor] = []
    audio_encodings: list[Tensor] = []
    masks: list[Tensor] = []
    for text in texts:
        hidden_states, attention_mask = text_encoder.encode(text)
        out = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        video_encodings.append(out.video_encoding)
        if out.audio_encoding is not None:
            audio_encodings.append(out.audio_encoding)
        masks.append(out.attention_mask)

    return {
        "video_prompt_embeds": torch.cat(video_encodings, dim=0),
        "audio_prompt_embeds": torch.cat(audio_encodings, dim=0) if audio_encodings else None,
        "prompt_attention_mask": torch.cat(masks, dim=0),
    }


# Frozen Gemma/connector outputs feed trainable attention: no_grad, not inference_mode.
@torch.no_grad()
def encode_modality_text_batch_online(
    text_encoder: torch.nn.Module,
    embeddings_processor: "EmbeddingsProcessor",
    video_texts: list[str],
    audio_texts: list[str],
) -> dict[str, Tensor]:
    """Encode distinct planner prompts through the native modality-specific paths.

    The same Gemma backbone is used for both prompts, but video representations are
    taken only from the video feature extractor/connector and audio representations
    only from the audio feature extractor/connector.  Keeping independent masks is
    required because the compiled prompts can have different sequence lengths.
    """
    if len(video_texts) != len(audio_texts):
        raise ValueError(
            "video_texts and audio_texts must have the same batch size, got "
            f"{len(video_texts)} and {len(audio_texts)}"
        )

    video_encodings: list[Tensor] = []
    audio_encodings: list[Tensor] = []
    video_masks: list[Tensor] = []
    audio_masks: list[Tensor] = []
    for video_text, audio_text in zip(video_texts, audio_texts, strict=True):
        video_hidden, video_attention = text_encoder.encode(video_text)
        video_out = embeddings_processor.process_hidden_states(
            video_hidden, video_attention
        )
        video_encodings.append(video_out.video_encoding)
        video_masks.append(video_out.attention_mask)

        audio_hidden, audio_attention = text_encoder.encode(audio_text)
        audio_out = embeddings_processor.process_hidden_states(
            audio_hidden, audio_attention
        )
        if audio_out.audio_encoding is None:
            raise ValueError(
                "Planner audio_prompt requires an audio Gemma connector"
            )
        audio_encodings.append(audio_out.audio_encoding)
        audio_masks.append(audio_out.attention_mask)

    return {
        "video_prompt_embeds": torch.cat(video_encodings, dim=0),
        "audio_prompt_embeds": torch.cat(audio_encodings, dim=0),
        "video_prompt_attention_mask": torch.cat(video_masks, dim=0),
        "audio_prompt_attention_mask": torch.cat(audio_masks, dim=0),
        # Backward-compatible alias used by existing video paths.
        "prompt_attention_mask": torch.cat(video_masks, dim=0),
    }
