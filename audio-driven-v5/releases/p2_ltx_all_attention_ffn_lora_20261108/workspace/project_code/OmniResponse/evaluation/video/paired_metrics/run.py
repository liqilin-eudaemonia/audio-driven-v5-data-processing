"""Run from repository root: python -m evaluation.video.paired_metrics.run --help."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np

from .core import (EMOTIONS, emotion_scores, frechet, json_safe, pixel_scores,
                   response_scores, validate_intervals)
from .media import load_pair, sha256

METRICS = ('psnr', 'ssim', 'fid', 'fvd', 'lpips', 'emo_acc', 'emo_div', 'delta_resp')


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_manifest(path):
    text = path.read_text()
    rows = json.loads(text) if path.suffix == '.json' else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON array or JSONL')
    identifiers = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in identifiers:
            raise ValueError('Each row needs a unique nonempty string id (one identity per row)')
        identifiers.add(row['id'])
        if 'gt_reference' in row:
            if row.get('dataset')!='seamless' or 'gt' in row:
                raise ValueError('Use an unambiguous dataset=seamless lazy GT reference')
            for key in ('manifest','crop_config'):
                row['gt_reference'][key]=str((path.parent/Path(row['gt_reference'][key])).resolve())
            keys=('generated',)
        else:
            keys=('gt','generated')
        for key in keys:
            source = Path(row[key])
            row[key] = str((path.parent/source).resolve())
    return rows


def summarize(results, distribution, selected, schema, datasets, requested_pairs, initialization_errors):
    summary = dict(schema=schema, requested_pairs=requested_pairs,
                   datasets=datasets,
                   pairs_with_errors=sum(bool(r['errors']) for r in results),
                   initialization_errors=initialization_errors, metrics={}, coverage={}, errors={})
    for metric in ('psnr','ssim','lpips'):
        if metric not in selected:
            continue
        valid = [r for r in results if metric in r['metrics']]
        frames = sum(r['media']['aligned_frames'] for r in valid)
        summary['metrics'][metric] = sum(r['metrics'][metric]*r['media']['aligned_frames'] for r in valid)/frames if frames else None
        summary['coverage'][metric] = dict(pairs=len(valid), frames=frames)
    for metric in ('fid','fvd'):
        if metric not in selected:
            continue
        a, b = distribution[metric]
        summary['coverage'][metric] = dict(pairs=len(a), samples=sum(len(x) for x in a))
        try:
            summary['metrics'][metric] = frechet(np.concatenate(a),np.concatenate(b),population=metric=='fvd')
        except Exception as exc:
            summary['metrics'][metric] = None
            summary['errors'][metric] = f'{type(exc).__name__}: {exc}'
    em = [r['emotion'] for r in results if r.get('emotion',{}).get('valid_frames',0)]
    if selected & {'emo_acc','emo_div'}:
        n = sum(e['valid_frames'] for e in em)
        summary['coverage']['emotion'] = dict(video_identities=len(em), common_valid_frames=n,
                                             aligned_frames=sum(r.get('media',{}).get('aligned_frames',0) for r in results))
        if 'emo_acc' in selected:
            summary['metrics']['emo_acc'] = sum(e['matches'] for e in em)/n if n else None
        if 'emo_div' in selected:
            for key in ('generated','gt'):
                summary['metrics']['emo_div_'+key] = float(np.mean([e['diversity_'+key] for e in em])) if em else None
    if 'delta_resp' in selected:
        responses = [r['response'] for r in results if 'response' in r]
        phases = [p for r in responses for p in r['phases'] if p['delta'] is not None]
        duration = sum(p['duration'] for p in phases)
        summary['metrics']['delta_resp'] = sum(p['delta']*p['duration'] for p in phases)/duration if duration else None
        summary['coverage']['delta_resp'] = dict(valid_phases=len(phases), valid_phase_duration=duration,
            requested_phase_duration=sum(r['requested_phase_duration'] for r in responses),
            valid_steps=sum(p['valid_steps'] for p in phases),
            possible_steps=sum(p['possible_steps'] for r in responses for p in r['phases']))
    summary['status'] = 'partial' if (summary['pairs_with_errors'] or summary['errors'] or
                           any(v is None for v in summary['metrics'].values())) else 'complete'
    return summary


def evaluate(args):
    import cv2
    cv2.setNumThreads(args.threads)
    selected = set(args.metrics.split(','))
    if not selected or not selected <= set(METRICS):
        raise ValueError(f'Metrics must be a comma-separated subset of {METRICS}')
    if min(args.threads, args.batch_size, args.fid_frames, args.fvd_frames, args.fvd_stride) < 1:
        raise ValueError('Thread, batch and frame counts must be positive')
    if args.face_padding < 0:
        raise ValueError('Face padding must be nonnegative')
    if args.fvd_frames < 16 and 'fvd' in selected:
        raise ValueError('I3D requires at least 16 input frames')
    face_device = getattr(args, 'face_device', 'cpu')
    emotion_device = getattr(args, 'emotion_device', 'cpu')
    pixel_workers = getattr(args, 'pixel_workers', 1)
    if pixel_workers < 1:
        raise ValueError('pixel_workers must be positive')
    rows = read_manifest(args.manifest)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('summary.json','pairs.jsonl','contract.json')):
        raise FileExistsError('Use a fresh output directory to preserve prior evaluations')
    assets = json.loads(args.assets.read_text()) if args.assets else {}
    contract = dict(schema='omniresponse-paired-video-v1', implementation_revision='2026-11-02-seamless-lazy-gt',
                    datasets=sorted({r.get('dataset','paired_video') for r in rows}),
                    arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    manifest_sha256=sha256(args.manifest), emotion_labels=EMOTIONS,
                    alignment='CFR frame-index, trim to shorter; GT resize only if explicitly declared in each row',
                    fid='pytorch-fid TensorFlow-compatible Inception pool3 2048, sample covariance',
                    fvd='StyleGAN-V/TFHub I3D, legacy bilinear+round uint8 RGB 224, population covariance',
                    delta_resp='AnyTalker Eq4, shared fixed square max>=224, EMA center .7; common adjacent valid support',
                    assets={}, versions={}, code={})
    for path in sorted(Path(__file__).parent.glob('*.py')):
        contract['code'][path.name] = sha256(path)
    legacy_fvd = Path(__file__).parents[1]/'fvd'/'frechet_video_distance.py'
    contract['code']['../fvd/frechet_video_distance.py'] = sha256(legacy_fvd)
    if any(r.get('dataset')=='seamless' for r in rows):
        adapter=Path(__file__).parents[2]/'seamless'/'reader.py'
        contract['code']['../../seamless/reader.py']=sha256(adapter)
    for name in ('numpy','scipy','torch','torchvision','opencv-python','scikit-image','lpips',
                 'pytorch-fid','deepface','tensorflow','insightface','onnxruntime','onnxruntime-gpu','av'):
        try:
            contract['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            contract['versions'][name] = None
    for name, path in assets.items():
        p = Path(path)
        if p.is_file():
            contract['assets'][name] = dict(path=str(p.resolve()), sha256=sha256(p))
        elif p.is_dir() and name == 'insightface_root':
            contract['assets'][name] = {f.name:sha256(f) for f in sorted((p/'models'/'buffalo_l').glob('*.onnx'))}
    contract['execution'] = dict(face_device=face_device, emotion_device=emotion_device,
                                  pixel_workers=pixel_workers, gt_cache=str(getattr(args, 'gt_cache', None)))
    write_json(out/'contract.json', contract)
    network = faces = None
    initialization_errors = {}
    if selected & {'lpips','fid','fvd'}:
        try:
            import torch
            torch.set_num_threads(args.threads)
            from .backends import Networks
            network = Networks(assets,args.device,args.batch_size)
        except Exception as exc:
            initialization_errors['networks'] = f'{type(exc).__name__}: {exc}'
    if selected & {'emo_acc','emo_div','delta_resp'}:
        try:
            from .backends import Faces
            faces = Faces(assets, bool(selected & {'emo_acc','emo_div'}), args.threads,padding=args.face_padding, device=face_device, emotion_device=emotion_device)
        except Exception as exc:
            initialization_errors['faces'] = f'{type(exc).__name__}: {exc}'
    if faces is not None and hasattr(faces, 'execution'):
        contract['execution']['face_backend'] = faces.execution
        write_json(out/'contract.json', contract)
    cache = None
    if getattr(args, 'gt_cache', None):
        from .cache import GTFeatureCache
        cache = GTFeatureCache(args.gt_cache, dict(code=contract['code'], assets=contract['assets'],
            versions=contract['versions'], face_device=face_device, emotion_device=emotion_device,
            network_device=args.device))
    results, distribution = [], {name:[[],[]] for name in ('fid','fvd')}
    for row in rows:
        started = time.monotonic()
        record = dict(id=row['id'], identity=row.get('identity','listener'), input=row, metrics={}, errors={}, timings={}, gt_cache_hits={})
        try:
            stage_start = time.monotonic()
            gt, generated, metadata = load_pair(row)
            record['timings']['decode'] = time.monotonic()-stage_start
            record['media'] = metadata
            n, fps = len(gt), metadata['fps']
            if selected & {'psnr','ssim'}:
                stage_start = time.monotonic()
                if pixel_workers == 1:
                    pixel = [pixel_scores(g,p) for g,p in zip(gt,generated)]
                else:
                    with ThreadPoolExecutor(max_workers=pixel_workers) as pool:
                        pixel = list(pool.map(pixel_scores, gt, generated))
                record['timings']['pixels'] = time.monotonic()-stage_start
                record['pixels'] = dict(frames=n, exact_frames=sum(p['mse']==0 for p in pixel),
                                        mse=float(np.mean([p['mse'] for p in pixel])))
                for key in ('psnr','ssim'):
                    if key in selected:
                        record['metrics'][key] = float(np.mean([p[key] for p in pixel]))
            for metric in ('lpips','fid','fvd'):
                if metric not in selected:
                    continue
                try:
                    stage_start = time.monotonic()
                    if network is None:
                        raise RuntimeError(initialization_errors.get('networks'))
                    if metric == 'lpips':
                        record['metrics']['lpips'] = float(np.mean(network.lpips(gt, generated)))
                    else:
                        if metric == 'fid':
                            indices = np.unique(np.linspace(0,n-1,min(n,args.fid_frames),dtype=int))
                        else:
                            span = (args.fvd_frames-1)*args.fvd_stride+1
                            if n < span:
                                raise ValueError(f'Short clip: {n} aligned frames, need {span}; no padding')
                            indices = (n-span)//2 + np.arange(args.fvd_frames)*args.fvd_stride
                        extract = getattr(network,metric)
                        if cache is None:
                            a = extract(gt[indices])
                        else:
                            cached, hit = cache.get(gt[indices], metric, dict(indices=indices.tolist()),
                                                    lambda: (extract(gt[indices]),))
                            a = cached[0]
                            record['gt_cache_hits'][metric] = hit
                        b = extract(generated[indices])
                        distribution[metric][0].append(a)
                        distribution[metric][1].append(b)
                        record[metric] = dict(status='features_ready', indices=indices.tolist(),
                                             feature_count=len(a), feature_dimension=a.shape[1])
                    record['timings'][metric] = time.monotonic()-stage_start
                except Exception as exc:
                    record['errors'][metric] = f'{type(exc).__name__}: {exc}'
            if selected & {'emo_acc','emo_div','delta_resp'}:
                try:
                    if faces is None:
                        raise RuntimeError(initialization_errors.get('faces'))
                    from .backends import normalize_landmarks
                    stage_start = time.monotonic()
                    if cache is None:
                        gl, gb, ge, gc = faces.extract(gt)
                    else:
                        (gl, gb, ge, gc), hit = cache.get(gt, 'faces',
                            dict(padding=args.face_padding, emotion=bool(selected & {'emo_acc','emo_div'})),
                            lambda: faces.extract(gt))
                        record['gt_cache_hits']['faces'] = hit
                    record['timings']['faces_gt'] = time.monotonic()-stage_start
                    stage_start = time.monotonic()
                    pl, pb, pe, pc = faces.extract(generated)
                    record['timings']['faces_generated'] = time.monotonic()-stage_start
                    record['face_detection'] = dict(gt_valid=int(np.isfinite(gb).all(1).sum()),
                                                    generated_valid=int(np.isfinite(pb).all(1).sum()),
                                                    gt_multiple=sum(c>1 for c in gc),
                                                    generated_multiple=sum(c>1 for c in pc), total=n)
                    if selected & {'emo_acc','emo_div'}:
                        em = emotion_scores(ge,pe)
                        record['emotion'] = em
                        if em['valid_frames']:
                            if 'emo_acc' in selected:
                                record['metrics']['emo_acc'] = em['accuracy']
                            if 'emo_div' in selected:
                                record['metrics']['emo_div_generated'] = em['diversity_generated']
                                record['metrics']['emo_div_gt'] = em['diversity_gt']
                        else:
                            record['errors']['emotion'] = 'No common valid emotion frames'
                    if 'delta_resp' in selected:
                        try:
                            intervals = row.get('listening_intervals')
                            if intervals is None and args.whole_clip_listening:
                                intervals = [[0,n/fps]]
                            if intervals is None:
                                raise ValueError('Supply listening_intervals or --whole-clip-listening')
                            # Accept full-GT intervals but record any tail lost to aligned support.
                            validate_intervals(intervals, metadata['gt_frames']/fps)
                            clipped = [[a,min(b,n/fps)] for a,b in intervals if a < n/fps]
                            gl, pl, side = normalize_landmarks(gl,pl,gb,pb)
                            response = response_scores(gl,pl,fps,clipped)
                            response['original_intervals'] = intervals
                            response['original_phase_duration'] = sum(b-a for a,b in intervals)
                            response['normalization_side_px'] = side
                            record['response'] = response
                            if response['delta'] is None:
                                raise ValueError('No measurable listening phases on common valid adjacent frames')
                            record['metrics']['delta_resp'] = response['delta']
                        except Exception as exc:
                            record['errors']['delta_resp'] = f'{type(exc).__name__}: {exc}'
                except Exception as exc:
                    record['errors']['faces'] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            record['errors']['pair'] = f'{type(exc).__name__}: {exc}'
        record['seconds'] = time.monotonic()-started
        results.append(record)
        with (out/'pairs.jsonl').open('a') as stream:
            stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False)+'\n')
        print(f"{len(results)}/{len(rows)} {row['id']} {record['seconds']:.1f}s errors={list(record['errors'])}", flush=True)
    if getattr(args,'feature_output',None):
        arrays={}
        for metric,(a,b) in distribution.items():
            dim=2048 if metric=='fid' else 400
            arrays[metric+'_gt']=np.concatenate(a) if a else np.empty((0,dim))
            arrays[metric+'_generated']=np.concatenate(b) if b else np.empty((0,dim))
        np.savez(args.feature_output,**arrays)
    summary=summarize(results,distribution,selected,contract['schema'],contract['datasets'],len(rows),initialization_errors)
    write_json(out/'summary.json', summary)
    print(json.dumps(json_safe(summary),ensure_ascii=False,indent=2))
    return 0 if summary['status']=='complete' or args.allow_partial else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,help='JSON with explicit local weight paths')
    parser.add_argument('--metrics',default=','.join(METRICS))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--face-device',default='cpu',help='ONNX face/landmarks device; CUDA fallback is rejected')
    parser.add_argument('--emotion-device',default='cpu',help='TensorFlow emotion device')
    parser.add_argument('--gt-cache',type=Path,help='Shared GT feature cache keyed by pixels, code, assets and backend')
    parser.add_argument('--pixel-workers',type=int,default=1,help='Parallel CPU PSNR/SSIM frame calculations')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--face-padding',type=int,default=0,help='Fixed black border for face detection/landmarks only; original coordinates restored')
    parser.add_argument('--fid-frames',type=int,default=32)
    parser.add_argument('--fvd-frames',type=int,default=32)
    parser.add_argument('--fvd-stride',type=int,default=1)
    parser.add_argument('--whole-clip-listening',action='store_true',help='Explicitly declare every aligned frame to be listener-only')
    parser.add_argument('--feature-output',type=Path,help='Save distribution features for exact distributed aggregation')
    parser.add_argument('--allow-partial',action='store_true',help='Exit 0 despite unavailable metrics; report remains partial')
    raise SystemExit(evaluate(parser.parse_args()))


if __name__ == '__main__':
    main()
