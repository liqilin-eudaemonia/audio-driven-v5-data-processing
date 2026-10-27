import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ltx_trainer import logger
from ltx_trainer.omni_condition_utils import (
    OmniConditionVersion,
    load_listener_timbre_embedding,
    load_omni_condition_file,
)

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"

_QWEN_SEQUENCE_KEYS = {
    "qwen_video_context",
    "qwen_context",
    "qwen_hidden_states",
    "qwen_audio_context",
    "qwen_attention_mask",
    "qwen_context_mask",
    "qwen_audio_attention_mask",
    "qwen_audio_context_mask",
}


def collate_precomputed_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate precomputed samples with strict handling of offline condition bundles."""
    if not batch:
        return {}

    special_keys = ("qwen_conditions", "omni_conditions", "listener_timbre_embedding")
    for key in special_keys:
        presence = [key in sample for sample in batch]
        if any(presence) and not all(presence):
            raise ValueError(f"Batch silently mixes samples with and without {key}")
    if not any(key in batch[0] for key in special_keys):
        return default_collate(batch)

    remainder = []
    for sample in batch:
        sample_copy = dict(sample)
        for key in special_keys:
            sample_copy.pop(key, None)
        remainder.append(sample_copy)

    collated = default_collate(remainder)
    if "qwen_conditions" in batch[0]:
        collated["qwen_conditions"] = _collate_qwen_condition_dicts(
            [sample["qwen_conditions"] for sample in batch]
        )
    if "omni_conditions" in batch[0]:
        collated["omni_conditions"] = _collate_omni_condition_dicts(
            [sample["omni_conditions"] for sample in batch]
        )
    if "listener_timbre_embedding" in batch[0]:
        embeddings = [sample["listener_timbre_embedding"] for sample in batch]
        dimensions = {tuple(embedding.shape) for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError(f"Listener timbre embedding dimensions differ within batch: {sorted(dimensions)}")
        collated["listener_timbre_embedding"] = torch.stack(embeddings)
    return collated


def _collate_omni_condition_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    versions = {item["version"] for item in items}
    if len(versions) != 1:
        raise ValueError(f"Cannot mix Omni condition versions in one batch: {sorted(versions)}")

    version = next(iter(versions))
    if version == "teacher_forcing_raw":
        layer_counts = {item["plan_latent"].shape[0] for item in items}
        hidden_sizes = {item["plan_latent"].shape[-1] for item in items}
        if len(layer_counts) != 1 or len(hidden_sizes) != 1:
            raise ValueError("Raw Omni layer counts and hidden sizes must match")
        max_tokens = max(item["plan_latent"].shape[1] for item in items)
        batch_size = len(items)
        layers = next(iter(layer_counts))
        hidden = next(iter(hidden_sizes))
        plan_latent = items[0]["plan_latent"].new_zeros(
            (batch_size, layers, max_tokens, hidden)
        )
        attention_mask = torch.zeros(
            (batch_size, layers, max_tokens), dtype=torch.bool
        )
        field_mask = torch.zeros((batch_size, 3, max_tokens), dtype=torch.bool)
        for index, item in enumerate(items):
            tokens = item["plan_latent"].shape[1]
            plan_latent[index, :, :tokens] = item["plan_latent"]
            attention_mask[index, :, :tokens] = item["omni_attention_mask"]
            item_field_mask = item.get("plan_field_mask")
            if not isinstance(item_field_mask, Tensor):
                raise ValueError("Raw Omni conditions require plan_field_mask")
            field_mask[index, :, :tokens] = item_field_mask
        collated: dict[str, Any] = {
            "plan_latent": plan_latent,
            "omni_attention_mask": attention_mask,
            "plan_field_mask": field_mask,
            "version": version,
        }
    else:
        layer_shapes = {tuple(item["plan_latent"].shape) for item in items}
        if len(layer_shapes) != 1:
            raise ValueError(
                f"Omni plan latent shapes differ within batch: {sorted(layer_shapes)}"
            )
        collated = {
            "plan_latent": torch.stack([item["plan_latent"] for item in items]),
            "omni_attention_mask": torch.stack(
                [item["omni_attention_mask"] for item in items]
            ),
            "version": version,
        }
    metadata_keys = (
        "audio_prompt",
        "video_prompt",
        "raw_plan_json",
        "planner_route",
        "causal_validated",
        "source",
        "sample_id",
    )
    collated.update({key: [item[key] for item in items] for key in metadata_keys})
    return collated


def _collate_qwen_condition_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for item in items:
        keys.update(item.keys())

    collated: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if len(values) != len(items):
            # Optional per-sample metadata; keep as a python list.
            collated[key] = [item.get(key) for item in items]
            continue

        first = values[0]
        if isinstance(first, Tensor) and key in _QWEN_SEQUENCE_KEYS:
            collated[key] = _pad_qwen_tensors(values)
        elif isinstance(first, Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                collated[key] = _pad_qwen_tensors(values)
        else:
            collated[key] = values
    return collated


def _pad_qwen_tensors(tensors: list[Tensor]) -> Tensor:
    """Pad Qwen tensors along the sequence axis to the batch maximum length."""
    if not tensors:
        raise ValueError("Cannot pad an empty Qwen tensor list")

    first = tensors[0]
    if first.ndim == 1:
        return pad_sequence(tensors, batch_first=True)
    if first.ndim == 2:
        return pad_sequence(tensors, batch_first=True)
    if first.ndim == 3:
        # [layers, seq, hidden] -> pad seq
        num_layers = first.shape[0]
        hidden = first.shape[-1]
        max_seq = max(tensor.shape[1] for tensor in tensors)
        out = first.new_zeros(len(tensors), num_layers, max_seq, hidden)
        for index, tensor in enumerate(tensors):
            if tensor.shape[0] != num_layers or tensor.shape[-1] != hidden:
                raise ValueError(
                    f"Qwen layered tensor shape mismatch: expected [{num_layers}, seq, {hidden}], got {tuple(tensor.shape)}"
                )
            out[index, :, : tensor.shape[1]] = tensor
        return out

    raise ValueError(f"Unsupported Qwen tensor rank for padding: {first.ndim}, shape={tuple(first.shape)}")


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
        *,
        sample_index_path: str | None = None,
        sample_index_video_field: str = "video",
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_index_path = sample_index_path
        self.sample_index_video_field = sample_index_video_field
        self.sample_files = (
            self._load_sample_index(sample_index_path)
            if sample_index_path
            else self._discover_samples()
        )
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources.
        Uses a fast two-pass approach: first globs all sources in parallel to build
        full-path sets in memory, then checks expected paths via set membership.
        This avoids O(N * num_sources) stat calls on networked filesystems while
        correctly handling path remapping (e.g. latent_X.pt -> condition_X.pt).
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]

        # Pass 1: Glob all sources in parallel, build full-path sets
        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            source_path = self.source_paths[dir_name]
            paths = list(source_path.glob("**/*.pt"))
            path_set = {str(p) for p in paths}
            return paths, path_set

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        # Get primary source files (cached from glob, no second scan)
        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        # Log source sizes
        for dir_name, (paths, _) in glob_results.items():
            logger.debug(f"Source {dir_name}: {len(paths)} files")

        # Build path sets for non-primary sources
        other_path_sets = {
            dir_name: path_set for dir_name, (_, path_set) in glob_results.items() if dir_name != data_key
        }

        # Pass 2: For each primary file, check if expected paths exist in other sources' sets
        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        valid_count = 0

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check all other sources via set lookup (O(1) per source, no stat calls)
            all_exist = True
            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(f"Fast index: {valid_count} valid samples from {len(data_files)} total ({skipped} skipped)")
        else:
            logger.debug(f"Fast index: {valid_count} valid samples from {len(data_files)} total")

        return sample_files

    def _load_sample_index(self, index_path: str) -> dict[str, list[Path]]:
        """Load prevalidated relative cache paths without scanning source directories."""
        path = Path(index_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Precomputed sample index not found: {path}")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources))
        sample_files: dict[str, list[Path]] = {
            output_key: [] for output_key in self.data_sources.values()
        }
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value:
                    continue
                if value.startswith("{"):
                    try:
                        row = json.loads(value)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON: {exc.msg}"
                        ) from exc
                    video_path = row.get(self.sample_index_video_field)
                    if not isinstance(video_path, str) or not video_path.strip():
                        raise ValueError(
                            f"{path}:{line_number}: missing {self.sample_index_video_field!r}"
                        )
                    rel_path = Path(_metadata_video_key(video_path) + ".pt")
                else:
                    rel_path = Path(value)

                if rel_path.is_absolute() or ".." in rel_path.parts or rel_path.suffix != ".pt":
                    raise ValueError(
                        f"{path}:{line_number}: unsafe or non-.pt relative path: {rel_path}"
                    )
                data_file = self.source_paths[data_key] / rel_path
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                count += 1

        if count == 0:
            raise ValueError(f"Precomputed sample index is empty: {path}")
        logger.info(
            f"Loaded offline sample index: {count:,} samples from {path}; "
            "source directory scan skipped"
        )
        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )

        # Verify all output keys have the same number of samples
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data


_FPS_SUFFIX_RE = re.compile(r"_\d+(?:\.\d+)?fps$")


def _latent_stem_to_video_stem(stem: str) -> str:
    """Strip the _24fps-style suffix appended by process_videos.py resampling."""
    return _FPS_SUFFIX_RE.sub("", stem)


def _metadata_video_key(video_path: str) -> str:
    """Map an absolute source video path to the relative precompute-cache key."""
    stem_path = Path(video_path).with_suffix("").as_posix()
    match = re.search(r"(?:^|/)0_DATA/(?:1_AUDIO|2_Avatar)/(.*)\Z", stem_path)
    return match.group(1) if match else Path(stem_path).name


def _metadata_sample_key(row: dict[str, Any], video_field: str) -> str:
    """Use an explicitly validated sample ID when source videos carry a _target suffix."""
    video_path = row[video_field]
    video_key = _metadata_video_key(video_path)
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        return video_key
    video_name = Path(video_key).name
    if video_name not in {sample_id, f"{sample_id}_target"}:
        raise ValueError(
            f"Metadata sample_id/video mismatch: {sample_id!r} versus {video_path!r}"
        )
    return sample_id


def _latent_metadata_keys(rel_path: Path) -> tuple[str, ...]:
    """Candidate metadata keys, preferring path-preserving exact matches."""
    relative = rel_path.with_suffix("").as_posix().lstrip("./")
    path = Path(relative)
    legacy_name = _latent_stem_to_video_stem(path.name)
    candidates = [relative]
    if legacy_name != path.name:
        candidates.append(path.with_name(legacy_name).as_posix())
    candidates.extend((path.name, legacy_name))
    return tuple(dict.fromkeys(candidates))


class OfflineOmniConditionDataset(Dataset):
    """Attach Route-A Omni conditions and optional listener timbre to another dataset.

    Conditions can be addressed either by ``sample_ids`` under a common root or by
    explicit per-sample paths. The two addressing modes are intentionally exclusive.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        omni_condition_root: str | Path | None = None,
        sample_ids: list[str] | None = None,
        omni_condition_paths: list[str | Path] | None = None,
        listener_timbre_root: str | Path | None = None,
        listener_timbre_paths: list[str | Path] | None = None,
        expected_version: OmniConditionVersion | None = None,
        expected_route: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.expected_version = expected_version
        self.expected_route = expected_route
        self.sample_ids = sample_ids
        self.omni_paths = self._resolve_paths(
            len(dataset),
            root=omni_condition_root,
            sample_ids=sample_ids,
            explicit_paths=omni_condition_paths,
            label="Omni condition",
            required=True,
        )
        self.listener_paths = self._resolve_paths(
            len(dataset),
            root=listener_timbre_root,
            sample_ids=sample_ids,
            explicit_paths=listener_timbre_paths,
            label="listener timbre",
            required=False,
        )

    @staticmethod
    def _resolve_paths(
        count: int,
        *,
        root: str | Path | None,
        sample_ids: list[str] | None,
        explicit_paths: list[str | Path] | None,
        label: str,
        required: bool,
    ) -> list[Path] | None:
        if root is not None and explicit_paths is not None:
            raise ValueError(f"{label}: use sample_id/root or explicit paths, not both")
        if explicit_paths is not None:
            if len(explicit_paths) != count:
                raise ValueError(f"{label}: expected {count} explicit paths, got {len(explicit_paths)}")
            return [Path(path).expanduser() for path in explicit_paths]
        if root is not None:
            if sample_ids is None or len(sample_ids) != count:
                actual = None if sample_ids is None else len(sample_ids)
                raise ValueError(f"{label}: expected {count} sample_ids for root lookup, got {actual}")
            root_path = Path(root).expanduser()
            paths = []
            for sample_id in sample_ids:
                sample_path = Path(sample_id)
                if not sample_id or sample_path.is_absolute() or ".." in sample_path.parts:
                    raise ValueError(f"{label}: unsafe sample_id {sample_id!r}")
                paths.append(root_path / sample_path.with_suffix(".pt"))
            return paths
        if required:
            raise ValueError(f"{label}: provide a root with sample_ids or explicit paths")
        return None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_sample = self.dataset[index]
        if not isinstance(base_sample, dict):
            raise TypeError(f"Wrapped dataset must return dict samples, got {type(base_sample).__name__}")

        sample = dict(base_sample)
        expected_sample_id = self.sample_ids[index] if self.sample_ids is not None else None
        sample["omni_conditions"] = load_omni_condition_file(
            self.omni_paths[index],
            expected_version=self.expected_version,
            expected_sample_id=expected_sample_id,
            expected_route=self.expected_route,
        ).as_dict()
        if self.listener_paths is not None:
            sample["listener_timbre_embedding"] = load_listener_timbre_embedding(self.listener_paths[index])
        return sample


def collate_online_condition_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate online-conditioning samples: tensors via default_collate, raw strings as lists."""
    if not batch:
        return {}

    samples = [dict(sample) for sample in batch]
    texts = [sample.pop("sd_text") for sample in samples]
    first_frames = [sample.pop("first_frame") for sample in samples]
    collated = collate_precomputed_batch(samples)
    collated["sd_text"] = texts
    collated["first_frame"] = torch.stack(first_frames)  # [B, H, W, 3] uint8, fixed canvas
    return collated


class OnlineConditionDataset(Dataset):
    """Precomputed latents + raw text/first-frame for training-time online condition encoding.

    Reuses PrecomputedDataset for the latent sources (video / audio latents stay offline),
    joins each latent file to its metadata JSONL row by video stem, and returns:
      - ``latent_conditions`` / ``audio_latent_conditions``: as in the offline path
      - ``sd_text``: the full raw caption string (no truncation; includes the trailing
        [AUDIO_CAPTION] music/audio analysis)
      - ``first_frame``: uint8 HWC tensor, VAE-aligned resize+center-crop of the video's
        first frame at the configured Qwen canvas (see online_conditioning)
    """

    def __init__(  # noqa: PLR0913
        self,
        data_root: str,
        metadata_jsonls: list[str],
        *,
        with_audio: bool = True,
        audio_latents_dir: str = "audio_latents",
        include_conditions: bool = False,
        caption_field: str = "sd_text",
        video_field: str = "video",
        path_remap: dict[str, str] | None = None,
        qwen_image_width: int = 1088,
        qwen_image_height: int = 640,
        max_training_frames: int | None = None,
    ) -> None:
        super().__init__()
        from ltx_trainer.online_conditioning import load_first_frame_for_qwen

        self._load_first_frame = load_first_frame_for_qwen
        self.caption_field = caption_field
        self.video_field = video_field
        self.path_remap = dict(path_remap or {})
        self.qwen_image_width = int(qwen_image_width)
        self.qwen_image_height = int(qwen_image_height)
        self.max_training_frames = max_training_frames

        sources = {"latents": "latents"}
        if include_conditions:
            sources["conditions"] = "conditions"
        if with_audio:
            sources[audio_latents_dir] = "audio_latents"
        self._latents = PrecomputedDataset(data_root, sources)

        self._metadata = self._build_metadata_index(metadata_jsonls)
        self._indices = self._match_samples()

    def _remap_path(self, path: str) -> str:
        for src, dst in self.path_remap.items():
            if path.startswith(src):
                return dst + path[len(src) :]
        return path

    def _read_caption(self, row: dict[str, Any]) -> str:
        """Full raw caption with sd_text_lines compatibility (mirrors UniAV row_text)."""
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

    def _build_metadata_index(self, metadata_jsonls: list[str]) -> dict[str, tuple[str, str]]:
        """Index JSONL rows by relative precompute key -> (video path, caption)."""
        index: dict[str, tuple[str, str]] = {}
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
                    key = _metadata_sample_key(row, self.video_field)
                    if key in index:
                        duplicates += 1
                    index[key] = (self._remap_path(video_path), caption)
        if duplicates:
            logger.warning(f"Metadata index: {duplicates} duplicate video cache keys (last row wins)")
        logger.info(f"Metadata index: {len(index)} rows from {len(metadata_jsonls)} JSONL file(s)")
        return index

    def _match_samples(self) -> list[tuple[int, str]]:
        """Match latent files to metadata rows by stem; keep only joined samples."""
        latent_rel_paths = self._latents.sample_files["latents"]
        matched: list[tuple[int, str]] = []
        for latent_index, rel_path in enumerate(latent_rel_paths):
            key = next(
                (candidate for candidate in _latent_metadata_keys(rel_path) if candidate in self._metadata),
                None,
            )
            if key is not None:
                matched.append((latent_index, key))
        skipped = len(latent_rel_paths) - len(matched)
        if not matched:
            raise ValueError(
                "OnlineConditionDataset: no latent files matched the metadata JSONLs. "
                "Check metadata_jsonls and path stems."
            )
        if skipped:
            logger.warning(
                f"OnlineConditionDataset: {skipped}/{len(latent_rel_paths)} latent files have no "
                "metadata row and were skipped"
            )
        logger.info(f"OnlineConditionDataset: {len(matched)} joined samples")
        return matched

    def __len__(self) -> int:
        return len(self._indices)

    def _crop_paired_latents(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.max_training_frames is None:
            return sample
        video_data = sample["latents"]
        video_latents = video_data["latents"]
        max_latent_frames = (self.max_training_frames - 1) // 8 + 1
        if video_latents.shape[1] <= max_latent_frames:
            return sample

        sample = dict(sample)
        video_data = dict(video_data)
        original_latent_frames = int(video_latents.shape[1])
        fps = float(video_data.get("fps", 30.0))
        target_duration = self.max_training_frames / fps
        video_data["latents"] = video_latents[:, :max_latent_frames].contiguous()
        video_data["num_frames"] = max_latent_frames
        sample["latents"] = video_data

        audio_data = sample.get("audio_latents")
        if audio_data is not None:
            audio_data = dict(audio_data)
            audio_latents = audio_data["latents"]
            original_duration = float(audio_data["duration"])
            audio_steps = min(
                int(audio_latents.shape[1]),
                max(
                    1,
                    math.ceil(
                        int(audio_latents.shape[1])
                        * target_duration
                        / original_duration
                    ),
                ),
            )
            audio_data["latents"] = audio_latents[:, :audio_steps].contiguous()
            audio_data["num_time_steps"] = audio_steps
            audio_data["duration"] = min(original_duration, target_duration)
            sample["audio_latents"] = audio_data

        logger.debug(
            "Cropped training sample from "
            f"{original_latent_frames} to {max_latent_frames} video latent frames"
        )
        return sample

    @property
    def sample_ids(self) -> list[str]:
        """Stable cache keys used to join external Omni/timbre condition files."""
        return [key for _, key in self._indices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        latent_index, key = self._indices[index]
        sample = self._crop_paired_latents(self._latents[latent_index])
        video_path, caption = self._metadata[key]
        sample["sd_text"] = caption
        sample["first_frame"] = self._load_first_frame(
            video_path,
            target_width=self.qwen_image_width,
            target_height=self.qwen_image_height,
        )
        sample["idx"] = index
        return sample
