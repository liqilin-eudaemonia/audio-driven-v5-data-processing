"""Frozen Whisper encoder features with an explicit common audio/video timeline."""
from dataclasses import dataclass
import math

import numpy as np
import torch


def encoder_numerics_contract():
    return dict(policy='whisper-fp32-no-tf32-v1',encoder_dtype='float32',
                autocast=False,matmul_allow_tf32=False,cudnn_allow_tf32=False,
                float32_matmul_precision='highest')


def configure_encoder_numerics():
    """Explicit old-cache numerical policy; does not initialize a CUDA context."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return encoder_numerics_contract()


@dataclass(frozen=True)
class SpeakerFeatures:
    hidden: torch.Tensor             # B,S,D; frozen perception, never Listener GT
    valid: torch.Tensor              # B,S bool
    bounds: torch.Tensor             # B,1,S,2 seconds relative to the shared clip start

    def validate(self):
        if self.hidden.ndim != 3 or self.hidden.requires_grad:
            raise ValueError("Expected frozen B,S,D Speaker features")
        b, s, d = self.hidden.shape
        if min(b, s, d) < 1 or not self.hidden.is_floating_point():
            raise ValueError('Whisper features must be nonempty floating-point tensors')
        if self.valid.shape != (b, s) or self.valid.dtype != torch.bool:
            raise ValueError("Invalid Speaker validity mask")
        if self.bounds.shape != (b, 1, s, 2) or self.bounds.requires_grad:
            raise ValueError("Invalid Speaker time bounds")
        if not (self.hidden.device == self.bounds.device == self.valid.device):
            raise ValueError("Speaker devices differ")
        if not torch.isfinite(self.hidden).all() or not torch.isfinite(self.bounds).all():
            raise ValueError("Nonfinite Speaker features")
        if (self.bounds[..., 1] < self.bounds[..., 0]).any():
            raise ValueError("Reversed Speaker intervals")
        for i in range(b):
            times = self.bounds[i, 0, self.valid[i]].mean(-1)
            if len(times) > 1 and (times[1:] <= times[:-1]).any():
                raise ValueError("Speaker timestamps must be strictly increasing")
        return self

    def to(self, device):
        return SpeakerFeatures(self.hidden.to(device), self.valid.to(device), self.bounds.to(device))


def windows(num_samples, rate=16000, window_s=30, context_s=2):
    """Disjoint retained cores, overlapping encoder windows; no input truncation.

    26-second cores with up to 2 seconds of context on each side. Every retained
    20ms token occurs exactly once. Window starts stay on the encoder time grid.
    """
    window, context = round(window_s * rate), round(context_s * rate)
    core = window - 2 * context
    if num_samples <= 0 or rate != 16000 or core <= 0 or core % 320 or context % 320:
        raise ValueError("Invalid Whisper window geometry")
    for start in range(0, num_samples, core):
        end = min(start + core, num_samples)
        yield max(0, start - context), min(num_samples, end + context), start, end


class FrozenWhisper:
    """Encoder only. Loading/extraction is separate from the training graph."""
    def __init__(self, path, device="cpu", dtype=torch.float32):
        from transformers import AutoFeatureExtractor, WhisperModel
        if dtype != torch.float32:
            raise ValueError('Whisper cache requires FP32 encoder weights and computation')
        configure_encoder_numerics()
        self.processor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
        full = WhisperModel.from_pretrained(path, local_files_only=True,
                                            torch_dtype=dtype, attn_implementation="sdpa")
        self.encoder = full.encoder.to(device).requires_grad_(False).eval()
        del full
        self.device, self.dtype = device, dtype
        if (self.processor.sampling_rate != 16000 or self.processor.hop_length != 160
                or self.encoder.conv2.stride != (2,) or self.processor.chunk_length != 30):
            raise ValueError("Unsupported Whisper time grid; do not silently retime features")

    @torch.no_grad()
    def encode(self, wave, origin_s=0.):
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or not math.isfinite(origin_s):
            raise ValueError("Invalid 16kHz mono waveform")
        rows, bounds = [], []
        for begin, end, core_begin, core_end in windows(len(wave)):
            inp = self.processor(wave[begin:end], sampling_rate=16000, return_tensors="pt",
                                 padding="max_length", truncation=False)
            with torch.autocast(torch.device(self.device).type,enabled=False):
                out = self.encoder(inp.input_features.to(device=self.device, dtype=self.dtype)).last_hidden_state[0]
            valid_count = math.ceil((end - begin) / 320)
            if valid_count > out.shape[0]:
                raise ValueError("Whisper returned too few frames")
            starts = torch.arange(valid_count, dtype=torch.int64) * 320 + begin
            keep = (starts >= core_begin) & (starts < core_end)
            starts = starts[keep]
            ends = (starts + 320).clamp(max=len(wave))
            rows.append(out[:valid_count][keep.to(out.device)].float().cpu())
            bounds.append(torch.stack([starts, ends], -1).double().div(16000).add(origin_s).float())
        hidden = torch.cat(rows)[None]
        times = torch.cat(bounds)[None, None]
        if hidden.shape[1] != math.ceil(len(wave) / 320):
            raise ValueError("Whisper core coverage mismatch")
        return SpeakerFeatures(hidden, torch.ones(hidden.shape[:2], dtype=torch.bool), times).validate()

    @torch.no_grad()
    def encode_batch(self, waves, origins=None, batch_size=16):
        """Batch encoder windows without changing each clip's disjoint time grid."""
        if type(batch_size) is not int or batch_size < 1 or not waves:
            raise ValueError('Expected a positive batch and nonempty waveform list')
        origins = [0.] * len(waves) if origins is None else origins
        if len(origins) != len(waves):
            raise ValueError('Waveform/origin count mismatch')
        jobs = []
        lengths = []
        for index, (wave, origin) in enumerate(zip(waves, origins)):
            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or not math.isfinite(origin):
                raise ValueError('Invalid 16kHz mono waveform')
            lengths.append(len(wave))
            jobs.extend((index, wave[b:e], b, e, cb, ce) for b,e,cb,ce in windows(len(wave)))
        rows = [[] for _ in waves]; bounds = [[] for _ in waves]
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset:offset+batch_size]
            inp = self.processor([j[1] for j in batch], sampling_rate=16000,
                return_tensors='pt', padding='max_length', truncation=False)
            with torch.autocast(torch.device(self.device).type,enabled=False):
                output = self.encoder(inp.input_features.to(self.device, self.dtype)).last_hidden_state
            for value, (i, _, begin, end, core_begin, core_end) in zip(output, batch):
                valid_count = math.ceil((end-begin)/320)
                if valid_count > value.shape[0]:
                    raise ValueError('Whisper returned too few frames')
                starts = torch.arange(valid_count, dtype=torch.int64)*320 + begin
                keep = (starts >= core_begin) & (starts < core_end)
                starts = starts[keep]; ends = (starts+320).clamp(max=lengths[i])
                rows[i].append(value[:valid_count][keep.to(value.device)].float().cpu())
                bounds[i].append(torch.stack((starts, ends), -1).double().div(16000).add(origins[i]).float())
            del output
        result = []
        for n, features, times in zip(lengths, rows, bounds):
            hidden = torch.cat(features)[None]
            if hidden.shape[1] != math.ceil(n/320):
                raise ValueError('Whisper core coverage mismatch')
            result.append(SpeakerFeatures(hidden, torch.ones(hidden.shape[:2], dtype=torch.bool),
                torch.cat(times)[None, None]).validate())
        return result


def validate_encoder_batching(encoder, batch_size=16, atol=1e-4, rtol=2e-4):
    """Actual encoder: short + >26s waveform, including a full configured batch."""
    configure_encoder_numerics()
    if type(batch_size) is not int or batch_size < 1 or encoder.dtype != torch.float32:
        raise ValueError('Invalid FP32 Whisper validation configuration')
    lengths=(23056,27*16000+123)
    short,long=(.1*np.sin(np.arange(n,dtype=np.float32)*.03) for n in lengths)
    # The long clip occupies two encoder windows, the short clip one. Fill the
    # rest with the same short signal so production-sized GPU kernels are checked.
    waves=[short,long]+[short]*max(0,batch_size-3)
    actual=encoder.encode_batch(waves,batch_size=batch_size)
    reference=[encoder.encode(short),encoder.encode(long)]
    errors=[]
    for index,value in enumerate(actual):
        ref=reference[1 if index==1 else 0]
        torch.testing.assert_close(value.hidden,ref.hidden,atol=atol,rtol=rtol)
        if not torch.equal(value.bounds,ref.bounds) or not torch.equal(value.valid,ref.valid):
            raise ValueError('Whisper batch/serial timing or validity differs')
        if value.hidden.dtype != torch.float32 or value.hidden.requires_grad:
            raise ValueError('Whisper batch validation must return frozen FP32 features')
        errors.append(float((value.hidden-ref.hidden).abs().max()))
    return dict(passed=True,device=str(encoder.device),batch_windows=batch_size,
        full_batch_exercised=True,reference_lengths_samples=list(lengths),
        max_abs_errors=errors,atol=atol,rtol=rtol,numerics=encoder_numerics_contract())
