"""The four upstream formulas plus auditable counts and seed-wise aggregation."""
import os
from pathlib import Path
import sys
from .common import ROOT, sha, read
from .upstream import metrics, REACTIONS

FID_HOME = ROOT / '.eval_models/torch'
FID_WEIGHT = FID_HOME / 'hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'


def fid_identity():
    import importlib.metadata
    package_root = ROOT / '.eval_deps/pytorch-fid-0.3.0'
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    if not FID_WEIGHT.is_file() or sha(FID_WEIGHT) != read(ROOT / 'configs/p2_glare_assets.lock.json')['fid_inception_sha256']:
        raise ValueError('Missing or changed pinned FID weights; run evaluation setup')
    import pytorch_fid
    package = Path(pytorch_fid.__file__).parent
    return dict(weights_sha256=sha(FID_WEIGHT) if FID_WEIGHT.exists() else None,
        source_sha256={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))},
        packages={k: importlib.metadata.version(k) for k in ('torch','torchvision','pytorch-fid','numpy','scipy')},
        dimensions=2048, preprocessing='pytorch-fid RGB [0,1], internal 299 resize and [-1,1]',
        frame_pool='each side own reaction intervals, all frames, unmatched included')


def score_seed(records, device='cpu', batch_size=16):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive FID batch size required')
    modules = metrics(); shared = modules['common']
    pairs = []; videos = []; details = []
    counts = {k: dict(tp=0, fp=0, fn=0, matched_tiou_sum=0., matched_atd_sum=0.) for k in REACTIONS}
    for row in records:
        gt = shared.read_annotation(Path(row['gt_csv']))
        gen = shared.read_annotation(Path(row['generated_csv']))
        if gt.frame_count != gen.frame_count or gt.frame_count != row['geometry']['frame_count']:
            raise ValueError(f'CSV/video frame counts differ: {row["sample_id"]}')
        key = row['sample_id']
        pairs.append((key, gt, gen))
        videos.append((key, (gt, Path(row['gt_video'])), (gen, Path(row['generated_video']))))
        matches = shared.match_events(gt.events, gen.events)
        serialized = []
        for name in REACTIONS:
            selected = [m for m in matches if m.ground_truth.reaction == name]
            counts[name]['tp'] += len(selected)
            counts[name]['fp'] += sum(e.reaction == name for e in gen.events) - len(selected)
            counts[name]['fn'] += sum(e.reaction == name for e in gt.events) - len(selected)
            counts[name]['matched_tiou_sum'] += sum(m.tiou for m in selected)
            counts[name]['matched_atd_sum'] += sum(modules['r_atd'].pair_penalty(m.ground_truth, m.prediction) for m in selected)
        for m in matches:
            serialized.append(dict(reaction=m.ground_truth.reaction, gt_frames=[m.ground_truth.start,m.ground_truth.end],
                generated_frames=[m.prediction.start,m.prediction.end], tiou=m.tiou,
                atd_raw=modules['r_atd'].pair_penalty(m.ground_truth,m.prediction)))
        def events(annotation):
            return [dict(reaction=e.reaction, frames=[e.start,e.end], seconds=[e.start/25,e.end/25]) for e in annotation.events]
        details.append(dict(sample_id=key, gt_events=events(gt), generated_events=events(gen), matches=serialized))
    values = {'R-F1':modules['r_f1'].compute_r_f1(pairs), 'R-tIoU':modules['r_tiou'].compute_r_tiou(pairs),
              'R-ATD':modules['r_atd'].compute_r_atd(pairs)}
    frame_counts = [sum(e.end-e.start for row in videos for e in row[side][0].events) for side in (1,2)]
    model = None; fid_runtime = None
    if min(frame_counts) >= 2:
        fid_runtime = fid_identity()
        if fid_runtime['weights_sha256'] is None:
            raise FileNotFoundError(f'Pre-fetch the pinned FID weights before scoring: {FID_WEIGHT}')
        import torch
        import pytorch_fid.inception
        torch.set_num_threads(4)
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested FID CUDA device is unavailable')
        # No first-use network download during a scientific evaluation run.
        os.environ['TORCH_HOME'] = str(FID_HOME)
        torch.hub.set_dir(str(FID_HOME / 'hub'))
        inception = pytorch_fid.inception.InceptionV3
        model = inception([inception.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
            normalize_input=True, requires_grad=False, use_fid_inception=True).to(device).eval()
    real, nreal = modules['r_fid'].collect_statistics(videos,1,model,device,batch_size,25.)
    generated, ngen = modules['r_fid'].collect_statistics(videos,2,model,device,batch_size,25.)
    if [nreal,ngen] != frame_counts:
        raise ValueError('Reaction frame pool differs from annotation')
    values['R-FID'] = modules['r_fid'].frechet_distance(real,generated) if min(frame_counts)>=2 else None
    for name, row in counts.items():
        tp, fp, fn = row['tp'], row['fp'], row['fn']
        row.update(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                   precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                   tiou=row['matched_tiou_sum']/tp if tp else 0.,
                   atd_raw=row['matched_atd_sum']/tp if tp else None)
    return dict(metrics=values, per_class=counts, per_clip=details,
        matched_events=sum(c['tp'] for c in counts.values()),
        reaction_frames=dict(ground_truth=nreal, generated=ngen), fid_runtime=fid_runtime,
        r_atd_raw=values['R-ATD']/100 if values['R-ATD'] is not None else None, r_atd_report_scale=100,
        null_reasons={k:v for k,v in (('R-ATD','no matched events' if values['R-ATD'] is None else None),
                        ('R-FID','fewer than two reaction frames in at least one pool' if values['R-FID'] is None else None)) if v})


def aggregate_seeds(results):
    names = ('R-F1','R-tIoU','R-ATD','R-FID')
    if not results:
        raise ValueError('No seeds to aggregate')
    values = {k:[v['metrics'][k] for v in results.values()] for k in names}
    return dict(mean={k:sum(v)/len(v) if all(x is not None for x in v) else None for k,v in values.items()},
                valid_seed_counts={k:sum(x is not None for x in v) for k,v in values.items()},
                planned_seed_count=len(results), policy='equal mean over all planned seeds; any null propagates')
