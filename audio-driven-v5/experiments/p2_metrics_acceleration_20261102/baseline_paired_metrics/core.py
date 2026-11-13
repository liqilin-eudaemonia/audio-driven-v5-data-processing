"""Metric arithmetic. No pretrained networks or GPU dependencies."""
import math
import numpy as np

EYE_BROW_INDICES = [43, 48, 49, 50, 51, 101, 102, 103, 104, 105,
                   35, 36, 37, 39, 41, 42, 89, 90, 91, 93, 95, 96]
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def pixel_scores(gt, generated):
    from skimage.metrics import structural_similarity
    if gt.shape != generated.shape or gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError('Pixel metrics require matching H,W,3 RGB images')
    if min(gt.shape[:2]) < 11:
        raise ValueError('SSIM needs image sides >=11 for the fixed 11-pixel window')
    mse = float(np.mean((gt.astype(np.float64) - generated.astype(np.float64)) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    ssim = structural_similarity(gt, generated, data_range=255, channel_axis=-1,
                                gaussian_weights=True, sigma=1.5,
                                win_size=11, use_sample_covariance=False)
    return dict(psnr=psnr, ssim=float(ssim), mse=mse)


def frechet(x, y, population=False):
    """Exact low-rank formula for small sets; PSD covariance formula otherwise.

    FID uses sample covariance; legacy OmniResponse I3D uses population covariance.
    A single feature is rejected even for the population estimator.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Features must be N,D and M,D with matching D')
    if min(len(x), len(y)) < 2:
        raise ValueError('Fréchet distance requires at least two samples per set')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite features')
    mean_term = np.sum((x.mean(0) - y.mean(0)) ** 2)
    a = (x - x.mean(0)) / np.sqrt(len(x) - (not population))
    b = (y - y.mean(0)) / np.sqrt(len(y) - (not population))
    trace = np.sum(a*a) + np.sum(b*b)
    if max(len(x), len(y)) <= x.shape[1]:
        cross = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    else:
        c1, c2 = a.T @ a, b.T @ b
        vals, vecs = np.linalg.eigh((c1 + c1.T) / 2)
        root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
        product = root @ c2 @ root
        cross = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T)/2), 0)).sum()
    result = float(mean_term + trace - 2*cross)
    if result < -1e-6 * max(1., float(mean_term + trace)):
        raise ArithmeticError('Materially negative Fréchet distance')
    return max(0., result)


def emotion_scores(gt, generated):
    """-1 means undetected. Diversity uses the same common valid support."""
    gt, generated = np.asarray(gt), np.asarray(generated)
    if gt.shape != generated.shape or gt.ndim != 1:
        raise ValueError('Emotion labels must be aligned 1D arrays')
    if not (np.isin(gt, range(-1, 7)).all() and np.isin(generated, range(-1, 7)).all()):
        raise ValueError('Emotion labels must be -1 or one of seven classes')
    valid = (gt >= 0) & (generated >= 0)
    n = int(valid.sum())
    return dict(valid_frames=n, total_frames=len(gt),
                matches=int((gt[valid] == generated[valid]).sum()),
                accuracy=float((gt[valid] == generated[valid]).mean()) if n else None,
                diversity_gt=int(len(np.unique(gt[valid]))) if n else None,
                diversity_generated=int(len(np.unique(generated[valid]))) if n else None,
                histogram_gt=np.bincount(gt[valid], minlength=7).tolist(),
                histogram_generated=np.bincount(generated[valid], minlength=7).tolist())


def validate_intervals(intervals, duration):
    result = []
    previous_end = 0.
    for item in intervals:
        if len(item) != 2:
            raise ValueError('Listening intervals must contain [start,end] pairs')
        start, end = map(float, item)
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start < end <= duration + 1e-6):
            raise ValueError(f'Invalid listening interval {item} for {duration:.6f}s')
        if start < previous_end:
            raise ValueError('Listening intervals must be sorted and non-overlapping')
        previous_end = end
        result.append((start, end))
    return result


def response_scores(gt, generated, fps, intervals):
    """AnyTalker Eq.4: phase-wise motion gap, then duration weighting.

    Input landmarks are already in the shared 224-coordinate system. Never
    connect landmarks across undetected frames or across listening phases.
    """
    gt, generated = np.asarray(gt), np.asarray(generated)
    if gt.shape != generated.shape or gt.ndim != 3 or gt.shape[1:] != (106, 2):
        raise ValueError('Landmarks must have matching T,106,2 shape')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive')
    intervals = validate_intervals(intervals, len(gt)/fps)
    times = np.arange(len(gt))/fps
    g, p = gt[:, EYE_BROW_INDICES], generated[:, EYE_BROW_INDICES]
    valid = np.isfinite(g).all((1, 2)) & np.isfinite(p).all((1, 2))
    gm = np.linalg.norm(np.diff(g, axis=0), axis=2).mean(1)
    pm = np.linalg.norm(np.diff(p, axis=0), axis=2).mean(1)
    phases = []
    for start, end in intervals:
        inside = (times >= start) & (times < end)
        possible = inside[:-1] & inside[1:]
        use = possible & valid[:-1] & valid[1:]
        n = int(use.sum())
        a, b = (float(gm[use].mean()), float(pm[use].mean())) if n else (None, None)
        phases.append(dict(start=start, end=end, duration=end-start,
                           possible_steps=int(possible.sum()), valid_steps=n,
                           motion_gt=a, motion_generated=b,
                           delta=abs(a-b) if n else None))
    eligible = [p for p in phases if p['delta'] is not None]
    duration = sum(p['duration'] for p in eligible)
    return dict(phases=phases, valid_phase_duration=duration,
                requested_phase_duration=sum(b-a for a, b in intervals),
                delta=sum(p['delta']*p['duration'] for p in eligible)/duration if duration else None)


def json_safe(value):
    """Keep mathematical infinity explicit, without invalid JSON NaN tokens."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('Unexpected NaN in evaluation report')
        return 'inf' if value > 0 else '-inf'
    return value
