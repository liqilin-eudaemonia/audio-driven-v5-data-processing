from pathlib import Path


DEFAULT_OUTPUT_STRIP_PREFIXES = (
    Path("/mnt/data04/480634/public/0_DATA/1_AUDIO"),
    Path("/public/0_DATA/1_AUDIO"),
)


def media_path_to_output_key(media_path: str | Path, dataset_root: str | Path) -> Path:
    """Map a media path to a safe relative key under a precomputed output dir.

    Relative media paths keep their structure. Absolute paths under the dataset
    root are made relative to that root. Known absolute media roots are stripped
    so outputs stay compact, e.g. MOVIE/... instead of mnt/data04/.../MOVIE/....
    """
    path = Path(str(media_path).strip())
    root = Path(dataset_root).resolve(strict=False)

    if path.is_absolute():
        # Preserve the dataset's lexical namespace before resolving symlinks.
        # Preprocessing commonly stages external media as links under the
        # manifest root; following such a link would leak its absolute source
        # path into the cache key and break metadata/condition joins.
        try:
            return _sanitize_relative_path(path.relative_to(root))
        except ValueError:
            pass
        resolved_path = path.resolve(strict=False)
        try:
            return _sanitize_relative_path(resolved_path.relative_to(root))
        except ValueError:
            pass
        for prefix in DEFAULT_OUTPUT_STRIP_PREFIXES:
            try:
                return _sanitize_relative_path(resolved_path.relative_to(prefix))
            except ValueError:
                continue

    return _sanitize_relative_path(path)


def _sanitize_relative_path(path: Path) -> Path:
    parts = []
    for part in path.parts:
        if part in {"", ".", path.anchor}:
            continue
        if part == "..":
            parts.append("__parent__")
        else:
            parts.append(part)

    if not parts:
        raise ValueError(f"Cannot create output key from empty path: {path}")

    return Path(*parts)
