"""Multi-length online-conditioning dataset with longest-stem dedup.

Opt-in companion to ``LengthBucketSampler``. Does **not** modify
``OnlineConditionDataset`` / ``PrecomputedDataset`` behavior.

Indexing rules:
  - One or more ``preprocessed_data_roots`` (e.g. ``..._ltx73_av``, ``..._ltx121_av``).
  - Bucket key = pixel frame count, parsed from the path segment ``ltx(\\d+)``.
  - Join latents to metadata JSONL rows by video stem (same as online conditioning).
  - If the same stem appears in multiple length roots, **keep only the longest**.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ltx_trainer import logger
from ltx_trainer.datasets import PrecomputedDataset, _latent_metadata_keys, _metadata_video_key

_LTX_FRAMES_RE = re.compile(r"ltx(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class _BucketSample:
    root_index: int
    latent_index: int
    stem: str
    pixel_frames: int


def parse_pixel_frames_from_path(path: str | Path) -> int | None:
    """Extract pixel frame count from a precomputed root path containing ``ltxNNN``."""
    matches = _LTX_FRAMES_RE.findall(str(path))
    if not matches:
        return None
    # Prefer the last match so nested unexpected ``ltx`` tokens don't win over the root name.
    return int(matches[-1])


def latent_frames_to_pixel_frames(latent_frames: int) -> int:
    """Convert VAE latent temporal size to pixel frames: ``(F_lat - 1) * 8 + 1``."""
    return (int(latent_frames) - 1) * 8 + 1


class LengthBucketOnlineDataset(Dataset):
    """Online-conditioning dataset indexed as length buckets.

    ``__getitem__`` accepts ``(bucket_key, sample_idx)`` tuples from
    ``LengthBucketSampler``. Return schema matches ``OnlineConditionDataset``:
    ``latents`` / ``audio_latents`` dicts, ``sd_text``, ``first_frame``.
    """

    def __init__(  # noqa: PLR0913
        self,
        data_roots: list[str],
        metadata_jsonls: list[str],
        *,
        sample_index_files: list[str] | None = None,
        with_audio: bool = True,
        audio_latents_dir: str = "audio_latents",
        include_conditions: bool = False,
        caption_field: str = "sd_text",
        video_field: str = "video",
        path_remap: dict[str, str] | None = None,
        qwen_image_width: int = 1088,
        qwen_image_height: int = 640,
        allowed_frames: list[int] | None = None,
    ) -> None:
        super().__init__()
        from ltx_trainer.online_conditioning import load_first_frame_for_qwen

        if not data_roots:
            raise ValueError("LengthBucketOnlineDataset requires a non-empty data_roots list")

        self._load_first_frame = load_first_frame_for_qwen
        self.caption_field = caption_field
        self.video_field = video_field
        self.path_remap = dict(path_remap or {})
        self.qwen_image_width = int(qwen_image_width)
        self.qwen_image_height = int(qwen_image_height)
        self.allowed_frames = set(int(x) for x in allowed_frames) if allowed_frames else None

        sources = {"latents": "latents"}
        if include_conditions:
            sources["conditions"] = "conditions"
        if with_audio:
            sources[audio_latents_dir] = "audio_latents"

        index_files = list(sample_index_files or [])
        if index_files and len(index_files) != len(data_roots):
            raise ValueError(
                "sample_index_files must be empty or align one-to-one with data_roots: "
                f"{len(index_files)} indexes for {len(data_roots)} roots"
            )
        if not index_files:
            index_files = [None] * len(data_roots)

        self._latent_datasets: list[PrecomputedDataset] = []
        self._root_pixel_frames: list[int] = []
        for root, sample_index_path in zip(data_roots, index_files, strict=True):
            pixel_frames = parse_pixel_frames_from_path(root)
            if pixel_frames is None:
                raise ValueError(
                    f"Cannot parse pixel frame count from data root (expected 'ltxNNN' in path): {root}"
                )
            if self.allowed_frames is not None and pixel_frames not in self.allowed_frames:
                logger.info(f"LengthBucketOnlineDataset: skipping root (not in allowed_frames): {root}")
                continue
            self._latent_datasets.append(
                PrecomputedDataset(
                    root,
                    sources,
                    sample_index_path=sample_index_path,
                    sample_index_video_field=video_field,
                )
            )
            self._root_pixel_frames.append(pixel_frames)

        if not self._latent_datasets:
            raise ValueError("LengthBucketOnlineDataset: no data roots left after allowed_frames filtering")

        self._metadata, self._media_meta = self._build_metadata_index(metadata_jsonls)
        self.buckets = self._build_buckets_longest_stem()
        self.num_samples = sum(len(v) for v in self.buckets.values())
        if self.num_samples == 0:
            raise ValueError(
                "LengthBucketOnlineDataset: no samples after metadata join + longest-stem dedup. "
                "Check metadata_jsonls / path stems / data roots."
            )
        stats = [(k, len(v)) for k, v in sorted(self.buckets.items())]
        logger.info(
            f"LengthBucketOnlineDataset: {self.num_samples:,} samples after longest-stem dedup, "
            f"{len(self.buckets)} buckets {stats}"
        )
        self._log_training_data_hours()

    def _remap_path(self, path: str) -> str:
        for src, dst in self.path_remap.items():
            if path.startswith(src):
                return dst + path[len(src) :]
        return path

    def _read_caption(self, row: dict[str, Any]) -> str:
        value = row.get(self.caption_field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            return "\n".join(str(part) for part in value)
        if self.caption_field == "sd_text":
            lines = row.get("sd_text_lines")
            if isinstance(lines, list):
                return "\n".join(str(part) for part in lines)
        return ""

    def _build_metadata_index(
        self, metadata_jsonls: list[str]
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[int | None, float]]]:
        """Return metadata keyed by relative precompute-cache path."""
        index: dict[str, tuple[str, str]] = {}
        media_meta: dict[str, tuple[int | None, float]] = {}
        duplicates = 0
        for jsonl_path in metadata_jsonls:
            path = Path(jsonl_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Metadata JSONL not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                    video_path = row.get(self.video_field)
                    if not isinstance(video_path, str) or not video_path.strip():
                        continue
                    caption = self._read_caption(row)
                    if not caption.strip():
                        continue
                    key = _metadata_video_key(video_path)
                    if key in index:
                        duplicates += 1
                    index[key] = (self._remap_path(video_path), caption)
                    n_frames = row.get("n_frames")
                    fps_raw = row.get("fps", 24.0)
                    try:
                        fps = float(fps_raw) if fps_raw is not None else 24.0
                    except (TypeError, ValueError):
                        fps = 24.0
                    if fps <= 0:
                        fps = 24.0
                    try:
                        n_frames_i = int(n_frames) if n_frames is not None else None
                    except (TypeError, ValueError):
                        n_frames_i = None
                    media_meta[key] = (n_frames_i, fps)
        if duplicates:
            logger.warning(f"LengthBucket metadata index: {duplicates} duplicate cache keys (last row wins)")
        logger.info(f"LengthBucket metadata index: {len(index):,} rows from {len(metadata_jsonls)} JSONL file(s)")
        return index, media_meta

    def _log_training_data_hours(self) -> None:
        """Log total / per-bucket media hours used after longest-stem dedup.

        Training hours use each sample's **bucket pixel frame count** / fps (what the
        model actually trains on). Source ``n_frames`` hours are logged separately when
        available in metadata.
        """
        default_fps = 24.0
        train_seconds = 0.0
        source_seconds = 0.0
        source_known = 0
        per_bucket_train_hours: dict[int, float] = {}

        for frames, samples in sorted(self.buckets.items()):
            bucket_seconds = 0.0
            for sample in samples:
                _n_frames, fps = self._media_meta.get(sample.stem, (None, default_fps))
                if fps <= 0:
                    fps = default_fps
                bucket_seconds += float(frames) / fps
                if _n_frames is not None and _n_frames > 0:
                    source_seconds += float(_n_frames) / fps
                    source_known += 1
            per_bucket_train_hours[frames] = bucket_seconds / 3600.0
            train_seconds += bucket_seconds

        train_hours = train_seconds / 3600.0
        source_hours = source_seconds / 3600.0
        per_bucket_str = ", ".join(f"{k}:{v:.2f}h({len(self.buckets[k]):,})" for k, v in per_bucket_train_hours.items())
        logger.info(
            f"LengthBucketOnlineDataset: training data duration = {train_hours:.2f} hours "
            f"({self.num_samples:,} samples after longest-stem dedup; "
            f"hours = sum(bucket_frames / fps) / 3600)"
        )
        logger.info(f"LengthBucketOnlineDataset: per-bucket training hours: {per_bucket_str}")
        if source_known:
            logger.info(
                f"LengthBucketOnlineDataset: source-clip duration (from metadata n_frames) = "
                f"{source_hours:.2f} hours on {source_known:,}/{self.num_samples:,} samples"
            )

    def _build_buckets_longest_stem(self) -> dict[int, list[_BucketSample]]:
        """Join each root's latents to metadata; keep only the longest length per stem."""
        best_by_stem: dict[str, _BucketSample] = {}
        unmatched = 0
        total_latents = 0

        for root_index, (latent_ds, pixel_frames) in enumerate(
            zip(self._latent_datasets, self._root_pixel_frames, strict=True)
        ):
            latent_rel_paths = latent_ds.sample_files["latents"]
            total_latents += len(latent_rel_paths)
            for latent_index, rel_path in enumerate(latent_rel_paths):
                stem = next(
                    (candidate for candidate in _latent_metadata_keys(rel_path) if candidate in self._metadata),
                    None,
                )
                if stem is None:
                    unmatched += 1
                    continue
                candidate = _BucketSample(
                    root_index=root_index,
                    latent_index=latent_index,
                    stem=stem,
                    pixel_frames=pixel_frames,
                )
                previous = best_by_stem.get(stem)
                if previous is None or candidate.pixel_frames > previous.pixel_frames:
                    best_by_stem[stem] = candidate

        if unmatched:
            logger.warning(
                f"LengthBucketOnlineDataset: {unmatched:,}/{total_latents:,} latent files had no "
                "metadata row and were skipped before longest-stem dedup"
            )

        dropped_shorter = total_latents - unmatched - len(best_by_stem)
        if dropped_shorter > 0:
            logger.info(
                f"LengthBucketOnlineDataset: longest-stem dedup dropped {dropped_shorter:,} "
                "shorter duplicate stem entries"
            )

        buckets: dict[int, list[_BucketSample]] = {}
        for sample in best_by_stem.values():
            buckets.setdefault(sample.pixel_frames, []).append(sample)

        # Stable order inside each bucket for reproducible sample_idx.
        for frame_key in buckets:
            buckets[frame_key].sort(key=lambda s: (s.root_index, s.latent_index, s.stem))
        return buckets

    def get_bucket_keys(self) -> list[int]:
        return list(self.buckets.keys())

    def get_bucket_size(self, bucket_key: int) -> int:
        return len(self.buckets.get(int(bucket_key), []))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: tuple[int, int] | int) -> dict[str, Any]:
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError(
                "LengthBucketOnlineDataset expects index=(bucket_key, sample_idx) from "
                f"LengthBucketSampler, got {index!r}"
            )
        bucket_key, sample_idx = int(index[0]), int(index[1])
        bucket = self.buckets.get(bucket_key)
        if bucket is None:
            raise KeyError(f"Unknown bucket_key={bucket_key}; available={sorted(self.buckets)}")
        if not 0 <= sample_idx < len(bucket):
            raise IndexError(
                f"sample_idx={sample_idx} out of range for bucket {bucket_key} (size={len(bucket)})"
            )

        ref = bucket[sample_idx]
        sample = self._latent_datasets[ref.root_index][ref.latent_index]
        video_path, caption = self._metadata[ref.stem]
        sample["sd_text"] = caption
        sample["first_frame"] = self._load_first_frame(
            video_path,
            target_width=self.qwen_image_width,
            target_height=self.qwen_image_height,
        )
        sample["idx"] = (bucket_key, sample_idx)
        sample["pixel_frames"] = ref.pixel_frames
        return sample
