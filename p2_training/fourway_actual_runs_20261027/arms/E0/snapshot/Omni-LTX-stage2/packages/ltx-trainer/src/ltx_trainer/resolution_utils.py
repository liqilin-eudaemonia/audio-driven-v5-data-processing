"""Aspect-ratio preserving spatial sizing for VAE/DiT token budgets."""

from __future__ import annotations

import math


def match_area_resolution(
    source_height: int,
    source_width: int,
    *,
    target_height: int = 720,
    target_width: int = 1280,
    multiple: int = 32,
) -> tuple[int, int]:
    """Return an aligned H/W near a fixed pixel budget without cropping.

    The output keeps the source aspect ratio closely while matching the target
    area; one alignment cell may put it slightly above the nominal budget. This mirrors
    area/token-budget sizing used by portrait A2V pipelines: landscape,
    portrait, and square inputs receive different canvases with comparable
    spatial token counts.
    """
    values = {
        "source_height": source_height,
        "source_width": source_width,
        "target_height": target_height,
        "target_width": target_width,
        "multiple": multiple,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Resolution values must be positive, got {values}")

    max_area = target_height * target_width
    source_aspect = source_height / source_width
    ideal_height = math.sqrt(max_area * source_aspect)
    ideal_width = math.sqrt(max_area / source_aspect)

    def aligned_neighbours(value: float) -> set[int]:
        center = value / multiple
        return {
            max(multiple, index * multiple)
            for index in range(max(1, math.floor(center) - 3), math.ceil(center) + 4)
        }

    candidates = [
        (height, width)
        for height in aligned_neighbours(ideal_height)
        for width in aligned_neighbours(ideal_width)
    ]
    if not candidates:
        raise ValueError(
            f"Pixel budget {max_area} is too small for alignment multiple {multiple}"
        )

    def score(size: tuple[int, int]) -> tuple[float, float, float]:
        height, width = size
        aspect_error = abs(math.log((height / width) / source_aspect))
        area_error = abs(height * width - max_area) / max_area
        # Geometry is weighted more heavily than token-count error so a 16:9
        # frame is not stretched merely to fill the last few percent of area.
        return 2.0 * aspect_error + area_error, aspect_error, area_error

    return min(candidates, key=score)
