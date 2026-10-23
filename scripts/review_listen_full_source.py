"""Reprocess complete source conversations and publish one continuous review each.

Runs independently of the frozen full queue. Exact baseline intervals may reuse
QC after waveform verification; every new cut gets fresh QC3/QC4. Rejected local
events trigger child extraction and new QC, with explicit termination evidence.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import listen_local_recovery as local
import run_listen_v8_audio_full as baseline
from p3_listen.common import read, write, sha

ROOT = baseline.ROOT
OUT = ROOT/'experiments/p3_listen_full_source_recovery_20261023'
PREVIEW = ROOT/'data_previews/seamless_listen_full_source_recovery'
DEFAULT_PAIRS = ['naturalistic_V02_S3610_I00000088', 'improvised_V01_S0122_I00000125']


def compact(row):
    keys = ('sample_id','start','end','duration_s','speaker_index','listener_index','text',
            'recovery_complete_sentence','recovery_depth','recovery_parent_id','boundary_review_flags')
    return dict({k:row[k] for k in keys if k in row}, pool=row['audio_final']['pool'],
                reasons=row['audio_final']['reasons'], quality=row['audio_final']['quality'])


def classify_timeline(context, original, attempts, final, diagnostic):
    """A total partition: every second is inspectable, including all unused time."""
    edges = {0., context.limit, *context.shots}
    for rows in (original, attempts, final, context.regions, context.source_gaps,
                 diagnostic['blocked'], diagnostic['unresolved']):
        for r in rows:
            edges.update([max(0., min(context.limit, r['start'])), max(0., min(context.limit, r['end']))])
    edges = sorted(edges)
    spans = []
    for a, b in zip(edges, edges[1:]):
        if b-a < 1e-6:
            continue
        t = (a+b)/2
        kept = [r for r in final if r['start'] <= t < r['end']]
        rejected = [r for r in attempts if r['start'] <= t < r['end'] and r['audio_final']['pool'] != 'keep_candidate']
        owner = next((r['speaker_index'] for r in context.regions if r['start'] <= t < r['end']),None)
        blocks = [r for r in diagnostic['blocked'] if r['start'] <= t < r['end']
                  and r.get('speaker_index',owner) == owner]
        gaps = [r for r in context.source_gaps if r['start'] <= t < r['end']]
        failures = [r for r in diagnostic['unresolved'] if r['start'] <= t < r['end']]
        if kept:
            status, reasons, ids = 'keep', ['passed_audio_quality'], [r['sample_id'] for r in kept]
        elif rejected:
            deepest = max(rejected, key=lambda r:r.get('recovery_depth', 0))
            status, reasons, ids = 'withheld', deepest['audio_final']['reasons'], [deepest['sample_id']]
        elif blocks:
            status, reasons, ids = 'blocked', sorted({r['reason'] for r in blocks}), []
        else:
            status, reasons, ids = 'uncovered', sorted({r['reason'] for r in gaps+failures}) or ['insufficient_safe_context_or_padding'], []
        value = dict(start=a, end=b, status=status, reasons=reasons, clip_ids=ids)
        if spans and all(spans[-1][k] == value[k] for k in ('status','reasons','clip_ids')):
            spans[-1]['end'] = b
        else:
            spans.append(value)
    assert abs(sum(r['end']-r['start'] for r in spans)-context.limit) < 1e-5
    return spans


def render_source(context, norm, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'source.mp4'
    panels = []
    for i, source in enumerate(context.sources):
        scale = min(720/source['width'], 480/source['height'])
        width, height = int(source['width']*scale)//2*2, int(source['height']*scale)//2*2
        panels.append(dict(x=i*720+(720-width)/2, y=(480-height)/2, width=width, height=height,
                           source_width=source['width'], source_height=source['height'],
                           rotation_degrees=context.geometry[i]['rotation_degrees']))
    if not target.exists():
        cmd = ['ffmpeg','-v','error','-nostdin','-y']
        for source in context.sources:
            cmd += ['-threads','1','-noautorotate','-i',local.old.base.media.uri(source['source'])]
        for track in norm['tracks']:
            cmd += ['-i',track['path']]
        filters = []
        for i, p in enumerate(panels):
            filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale={p["width"]}:{p["height"]},setsar=1,'
                           f'pad=720:480:(ow-iw)/2:(oh-ih)/2:black,fps=25[v{i}]')
        filters += ['[v0][v1]hstack=inputs=2[v]',
                    '[2:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a0]',
                    '[3:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a1]',
                    '[a0][a1]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]']
        cmd += ['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
                '-t',str(context.limit),'-c:v','libx264','-preset','fast','-crf','23','-threads','2',
                '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)]
        subprocess.run(cmd, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]))
    duration = float(probe['format']['duration'])
    assert abs(duration-context.limit) <= .1
    assert next(s for s in probe['streams'] if s['codec_type'] == 'audio')['channels'] == 2
    return panels, dict(duration=duration, sha256=sha(target), full_source=True, original_timeline=True,
                        audio='A in left channel, B in right channel; existing whole-track gain, no denoising')


def process(pair, item, models, prepare_only=False):
    old_path = baseline.OUT/'results'/(pair+'.json')
    old_result = read(old_path)
    original = old_result['clips']
    scan_path = OUT/'source_scans'/(pair+'.json')
    scan = read(scan_path) if scan_path.exists() else None
    shots = scan['shots'] if scan and scan['source_items'] == [item['a'],item['b']] else None
    print(json.dumps(dict(pair=pair,stage='loading_and_scanning_source',cached_shots=shots is not None)),flush=True)
    context = local.SourceContext(item,shot_times=shots)
    write(scan_path,dict(source_items=[item['a'],item['b']],shots=context.shots_by_track,
                        detector='native-frame ffmpeg scene > 0.35; analysis width 320'))
    norm_path = baseline.OUT/'normalized_pairs'/(pair+'.json')
    norm = read(norm_path)
    for track in norm['tracks']:
        assert sha(Path(track['path'])) == track['sha256']
    candidates, diagnostic = context.propose(original)
    write(OUT/'prepared'/(pair+'.json'), dict(pair=pair, clips=candidates, diagnostic=diagnostic,
          shots=context.shots_by_track, regions=context.regions, source_gaps=context.source_gaps,
          source_seconds=context.limit))
    print(json.dumps(dict(pair=pair, stage='prepared', candidates=len(candidates), shots=context.shots_by_track)), flush=True)
    if prepare_only:
        return None
    feedback = dict(rejects=read(baseline.OUT/'human_rejections.json')['rows'],
                    positives=read(baseline.OUT/'human_retained.json')['rows'])
    cache = {baseline.interval_key(r):r for r in original}
    pending = [(r, 0, None) for r in candidates]
    final, attempts, seen = [], [], set()
    while pending:
        plan, depth, parent = pending.pop(0)
        key = baseline.interval_key(plan)
        if key in seen:
            continue
        seen.add(key)
        plan.update(recovery_depth=depth, recovery_parent_id=parent)
        previous = cache.get(key)
        if previous and baseline.cache_matches(plan, previous, norm):
            row = copy.deepcopy(previous)
            provenance = dict(reused=True, file=str(old_path), sha256=sha(old_path), exact_interval=True)
        else:
            if not models:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                models.extend([baseline.full.legacy.EventModel(baseline.full.legacy.MODEL, batch_size=16),
                               baseline.full.legacy.PannsModel(batch_size=16)])
            row = baseline.RAW_SCREEN(plan, norm, *models, feedback['rejects'], feedback['positives'])
            row = baseline.original.assess(row, baseline.OLD_FEEDBACK)
            provenance = dict(reused=False, fresh_qc=True, algorithm='frozen QC3 + QC4')
        row = baseline.finish_row(plan, row, feedback)
        row['quality_provenance'] = provenance
        attempts.append(row)
        if row['audio_final']['pool'] == 'keep_candidate':
            final.append(row)
        else:
            blocks, kind = local.quality_blocks(row)
            if kind == 'localized_quality' and depth < 6:
                children, detail = context.propose([row], domain=row)
                diagnostic['blocked'].extend(detail['blocked'])
                diagnostic['unresolved'].extend(detail['unresolved'])
                for child in children:
                    assert child['duration_s'] < row['duration_s']-1e-5
                    pending.append((child, depth+1, row['sample_id']))
            elif kind == 'localized_quality':
                diagnostic['unresolved'].append(dict(start=row['start'], end=row['end'], reason='recovery_depth_limit'))
        write(OUT/'progress.json', dict(pair=pair, processed=len(attempts), pending=len(pending), kept=len(final)))
        print(json.dumps(dict(pair=pair, processed=len(attempts), pending=len(pending), pool=row['audio_final']['pool'])), flush=True)
    final.sort(key=lambda r:(r['start'], r['end']))
    assert all(a['end'] <= b['start']+1e-6 for a,b in zip(final, final[1:]))
    assert not any(r['start'] < t < r['end'] for r in final for t in context.shots)
    old_keep = [(r['start'],r['end']) for r in original if r['audio_final']['pool']=='keep_candidate']
    new_keep = [(r['start'],r['end']) for r in final]
    summary = dict(pair=pair, source_seconds=context.limit, original_clips=len(original),
        old_kept_clips=sum(r['audio_final']['pool']=='keep_candidate' for r in original),
        initial_candidates=len(candidates), qc_attempts=len(attempts), final_kept_clips=len(final),
        old_kept_seconds=local.seconds(old_keep), new_kept_seconds=local.seconds(new_keep),
        added_seconds=local.seconds(local.subtract(new_keep,old_keep)),
        removed_seconds=local.seconds(local.subtract(old_keep,new_keep)),
        partial_sentence_kept=sum(not r['recovery_complete_sentence'] for r in final),
        detected_cuts=context.shots_by_track, training_admitted=False, visual_filter_applied=False,
        note='Diagnostic full-source resegmentation; no extrapolation from these two selected conversations.')
    write(OUT/'results'/(pair+'.json'),dict(summary=summary, clips=final, attempts=attempts, diagnostic=diagnostic,
          original_result_sha256=sha(old_path), normalization_sha256=sha(norm_path)))
    rows = [compact(r) for r in attempts]
    for row in rows:
        row['crops'] = [local.preview_crop(g,row['start'],row['end']) for g in context.geometry]
    overview_crops = [dict(start=a,end=min(context.limit,a+10),
                          crops=[local.preview_crop(g,a,min(context.limit,a+10)) for g in context.geometry])
                      for a in range(0,math_ceil(context.limit),10)]
    panels, media = render_source(context,norm,PREVIEW/pair)
    value = dict(pair=pair, summary=summary, media=pair+'/source.mp4', media_verification=media, panels=panels,
        old=[compact(r) for r in original], attempts=rows, kept_ids=[r['sample_id'] for r in final],
        timeline=classify_timeline(context,original,attempts,final,diagnostic),
        shots=context.shots_by_track, regions=context.regions, source_gaps=context.source_gaps,
        old_unresolved=read(baseline.OUT/'audit'/(pair+'.json'))['unresolved'],
        diagnostic=diagnostic, overview_crops=overview_crops,
        transcripts=[[dict(start=s['start'],end=s['end'],text=s['text']) for s in p['segments']] for p in context.people])
    write(PREVIEW/pair/'review.json',value)
    print(json.dumps(summary),flush=True)
    return value


def math_ceil(value):
    import math
    return math.ceil(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', action='append')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    pairs = args.pair or DEFAULT_PAIRS
    baseline.verify()
    baseline.bind()
    catalog = {r['pair']:r for r in read(baseline.OUT/'catalog.json')['rows']}
    models, values = [], []
    for pair in pairs:
        values.append(process(pair,catalog[pair],models,args.prepare_only))
    if args.prepare_only:
        return
    PREVIEW.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'scripts/listen_full_source_review.html',PREVIEW/'index.html')
    write(PREVIEW/'index.json',dict(revision=local.REVISION,pairs=[dict(pair=v['pair'],summary=v['summary']) for v in values],
        training_admitted=False, visual_filter_applied=False))
    source = Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if source.exists():
        from fontTools import subset
        from fontTools.ttLib import TTFont
        font = TTFont(source,fontNumber=2)
        sub = subset.Subsetter()
        sub.populate(text=(PREVIEW/'index.html').read_text()+json.dumps(values,ensure_ascii=False))
        sub.subset(font)
        font.flavor='woff'
        font.save(PREVIEW/'review.woff')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_full_source'
    if not link.exists():
        link.symlink_to(PREVIEW,target_is_directory=True)
    write(OUT/'contract.json',dict(revision=local.REVISION,pairs=pairs,
        code_sha256={name:sha(ROOT/name) for name in ['scripts/listen_local_recovery.py',
            'scripts/review_listen_full_source.py','scripts/listen_full_source_review.html']},
        frozen_baseline_contract_sha256=sha(baseline.OUT/'contract.json'),
        complete_source_review=True, production_queue_replaced=False, training_admitted=False))
    write(OUT/'summary.json',[v['summary'] for v in values])
    print('http://127.0.0.1:8875/listen_full_source/',flush=True)


if __name__=='__main__':
    main()
