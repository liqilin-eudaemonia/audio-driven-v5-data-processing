import hashlib
import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def decode(path, roi=None):
    """Decode CFR RGB frames. ROI is a fixed, explicit [x,y,width,height]."""
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Cannot decode video/FPS: {path}')
        frames, stamps = [], []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000)
            if roi is not None:
                if len(roi) != 4 or any(int(v) != v for v in roi):
                    raise ValueError('ROI must be integer [x,y,width,height]')
                x, y, w, h = map(int, roi)
                if x < 0 or y < 0 or min(w,h) <= 0 or x+w > bgr.shape[1] or y+h > bgr.shape[0]:
                    raise ValueError(f'ROI outside frame: {roi}')
                bgr = bgr[y:y+h, x:x+w]
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not frames:
            raise ValueError(f'Empty video: {path}')
        # Some OpenCV codecs expose no timestamps at all: make this explicit.
        pts_checked = len(stamps) > 1 and max(stamps) > 0
        if pts_checked and not np.allclose(np.diff(stamps), 1/fps, atol=.002, rtol=.02):
            raise ValueError('Variable/non-monotonic frame timestamps; convert to aligned CFR first')
        return np.stack(frames), float(fps), pts_checked
    finally:
        capture.release()


def load_pair(row):
    lazy_audit=None
    if 'gt_reference' in row:
        if row.get('dataset')!='seamless' or 'gt' in row or row.get('gt_roi') is not None:
            raise ValueError('Lazy GT requires dataset=seamless and its final crop; do not combine with gt/gt_roi')
        from ...seamless.reader import decode_reference
        g,gf,lazy_audit=decode_reference(row['gt_reference'])
        gc=True
        gt_digest=lazy_audit['rgb_sha256']
    else:
        g, gf, gc = decode(row['gt'], row.get('gt_roi'))
        gt_digest=sha256(row['gt'])
    original_gt_shape=list(g.shape[1:])
    if row.get('gt_resize') is not None:
        if row.get('gt_resize_method')!='pil_bicubic':raise ValueError('Explicit supported GT resize method required')
        size=row['gt_resize']
        if len(size)!=2 or any(type(x) is not int or x<=0 for x in size):raise ValueError('Invalid GT resize size')
        from PIL import Image
        g=np.stack([np.asarray(Image.fromarray(x).resize(tuple(size),Image.Resampling.BICUBIC)) for x in g])
    p, pf, pc = decode(row['generated'], row.get('generated_roi'))
    if abs(gf-pf) > 1e-3:
        raise ValueError(f'FPS mismatch: GT={gf}, generated={pf}; no automatic retiming')
    if g.shape[1:] != p.shape[1:]:
        raise ValueError('Spatial sizes differ; supply identically sized listener crops')
    n = min(len(g), len(p))
    metadata = dict(fps=gf, aligned_frames=n, gt_frames=len(g), generated_frames=len(p),
                    trimmed_gt=len(g)-n, trimmed_generated=len(p)-n,
                    timestamp_checks=dict(gt=gc, generated=pc),
                    gt_sha256=gt_digest, generated_sha256=sha256(row['generated']))
    if row.get('gt_resize') is not None:
        metadata['gt_resize']=dict(original_shape=original_gt_shape,size=row['gt_resize'],method=row['gt_resize_method'])
    if lazy_audit is not None:
        metadata['gt_sha256_kind']='decoded_full_gt_rgb_sequence'
        metadata['gt_decoder']=lazy_audit
    return g[:n], p[:n], metadata
