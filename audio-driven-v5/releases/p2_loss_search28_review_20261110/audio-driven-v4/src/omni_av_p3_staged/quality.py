"""Paired identity checks; missing faces must not improve the passing mean."""
import math


QUALITY_VERSION = 2
IDENTITY_DEFAULTS = dict(id_min_samples=6,id_max_sample_drop=1,id_min_frame_coverage=.5,
                         id_frame_coverage_tolerance=.125)


def identity_comparison(reference, candidate, expected_ids, thresholds):
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise ValueError('Duplicate quality sample IDs')

    def valid_rows(payload):
        rows = payload.get('per_sample', [])
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError('Identity metric sample ID mismatch')
        return {r['id']: r for r in rows
                if r.get('status') == 'success'
                and isinstance(r.get('id_similarity'), (int, float))
                and math.isfinite(r['id_similarity'])
                and r.get('id_valid_frames', 0) > 0
                and thresholds.get('id_min_frame_coverage', .5) <= r.get('id_frame_coverage', 0) <= 1}

    baseline, found = valid_rows(reference), valid_rows(candidate)
    common = sorted(set(baseline) & set(found))
    minimum = thresholds.get('id_min_samples', 6)
    required = max(minimum, len(baseline) - thresholds.get('id_max_sample_drop', 1))
    coverage_ok = len(baseline) >= minimum and len(found) >= required and len(common) >= required
    before = sum(baseline[k]['id_similarity'] for k in common) / len(common) if common else None
    after = sum(found[k]['id_similarity'] for k in common) / len(common) if common else None
    frame_ok = bool(common) and all(
        found[k]['id_frame_coverage'] >= baseline[k]['id_frame_coverage']
        - thresholds.get('id_frame_coverage_tolerance', .125) for k in common)
    return dict(baseline_valid=len(baseline), candidate_valid=len(found), paired_count=len(common),
                paired_ids=common, baseline_paired_mean=before, candidate_paired_mean=after,
                coverage_passed=coverage_ok, frame_coverage_passed=frame_ok,
                nonregression_passed=coverage_ok and frame_ok and after >= before - thresholds['id_tolerance'])
