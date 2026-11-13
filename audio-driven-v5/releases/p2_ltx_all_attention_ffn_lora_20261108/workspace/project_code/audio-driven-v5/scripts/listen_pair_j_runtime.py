"""Checkpointed CPU adapter; the frozen J decision code is not modified."""
import collections
import copy
from pathlib import Path
import sys
import time

from listen_pair_j_queue import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Paused(Exception):
    pass


class Engine:
    def __init__(self, work, check, progress):
        import numpy as np
        import torch
        import listen_local_recovery_semantic_trial as repair
        import run_listen_v8_audio_full as screening
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        self.np = np; self.repair = repair; self.screening = screening
        self.work = Path(work); self.manifest = read(self.work / 'manifest.json')
        self.check = check; self.progress = progress; self.models = []
        screening.FEEDBACK = read(self.work / 'inputs/feedback.json')['rows']
        screening.OLD_FEEDBACK = read(self.work / 'inputs/qc4_review_feedback.json')['rows']
        self.feedback = dict(rejects=read(self.work / 'inputs/human_rejections.json')['rows'],
                             positives=read(self.work / 'inputs/human_retained.json')['rows'])
        self.reviews = read(self.work / 'inputs/listener_noise_rejections.json')['rows']

    def context(self, item):
        pair = item['pair']; path = self.work / 'snapshots' / (pair + '.json')
        if path.exists():
            snapshot = read(path)
            if snapshot.get('cluster_contract') != self.manifest['contract']:
                raise ValueError('Snapshot contract mismatch')
        else:
            self.progress(stage='preparing', pair=pair)
            entry = self.manifest['source_inputs'][pair]
            if entry.get('snapshot'):
                if sha(entry['snapshot']) != entry['snapshot_sha256']:
                    raise ValueError('Source snapshot changed')
                snapshot = read(entry['snapshot'])
            else:
                base = self.repair.base.SourceContext(item, shot_times=[[], []])
                snapshot = dict(pair=pair, item=item, people=base.people,
                                rms=[x.tolist() for x in base.rms], limit=base.limit,
                                sources=base.sources)
            snapshot['cluster_contract'] = self.manifest['contract']
            write(path, snapshot)
        ctx = self.repair.SourceContext.__new__(self.repair.SourceContext)
        ctx.item = snapshot['item']; ctx.people = self.repair.prepare_people(snapshot['people'])
        ctx.rms = [self.np.array(x) for x in snapshot['rms']]
        ctx.limit = snapshot['limit']; ctx.sources = snapshot['sources']
        ctx.records = [dict(id=s['id']) for s in ctx.sources]
        ctx.geometry = [dict(s, rotation_degrees=0, arrays={}) for s in ctx.sources]
        ctx.shots = []; ctx.shots_by_track = [[], []]
        ctx.regions, ctx.source_gaps = self.repair.role_regions(ctx.people, ctx.limit, [])
        ctx.reviews = self.reviews
        return ctx, sha(path)

    def process(self, item):
        pair = item['pair']; self.check(); began = time.monotonic()
        folder = self.work / 'results' / pair
        commit = folder / 'result.json'
        # Recover a crash after result publication but before completion receipt.
        if commit.exists():
            existing = read(commit)
            if existing.get('cluster_contract') != self.manifest['contract']:
                raise ValueError('Result contract mismatch')
            if existing.get('complete'):
                write(folder / 'complete.json', dict(contract=self.manifest['contract'],
                      result_sha256=sha(commit)))
                return existing
        ctx, snapshot_hash = self.context(item); self.check()
        entry = self.manifest['source_inputs'][pair]
        for p, digest in entry['files'].items():
            if sha(p) != digest:
                raise ValueError('Pair input changed: ' + p)
        norm = read(entry['norm'])
        for track in norm['tracks']:
            self.check()
            if sha(track['path']) != track['sha256']:
                raise ValueError('Normalized audio changed: ' + track['path'])
        cache = {}
        for name in entry['cache_results']:
            value = read(name); digest = sha(name)
            for row in value['clips'] + value.get('superseded_attempts', []):
                cache[self.screening.interval_key(row)] = (row, name, digest)
        for directory in self.manifest.get('extra_cache_dirs', []) + [str(self.work / 'results')]:
            for path in (Path(directory) / pair / 'qc').glob('*.json'):
                row = read(path)
                cache[self.screening.interval_key(row)] = (row, str(path), sha(path))
        checkpoint = folder / 'checkpoint.json'
        if checkpoint.exists():
            state = read(checkpoint)
            if (state['contract'] != self.manifest['contract'] or
                    state['snapshot_sha256'] != snapshot_hash):
                raise ValueError('Checkpoint input/contract mismatch')
            state['resume_count'] += 1
        else:
            initial, diagnostic = ctx.propose([])
            write(folder / 'initial.json', dict(pair=pair, clips=initial, diagnostic=diagnostic,
                  regions=ctx.regions, cluster_contract=self.manifest['contract'],
                  local_feedback_audit=[p.get('local_feedback_audit', []) for p in ctx.people]))
            state = dict(contract=self.manifest['contract'], snapshot_sha256=snapshot_hash,
                         pending=[[r, 0, None] for r in initial], seen=[], leaves=[], parents=[],
                         counts={}, diagnostic=diagnostic, initial_ids=[r['sample_id'] for r in initial],
                         resume_count=0, elapsed=0.)
            write(checkpoint, state)
        seen = {tuple(key) for key in state['seen']}
        counts = collections.Counter(state['counts']); diagnostic = state['diagnostic']
        elapsed_before = state['elapsed']
        while state['pending']:
            self.check()
            plan, depth, parent = state['pending'][0]
            key = self.screening.interval_key(plan)
            if key in seen:
                state['pending'].pop(0); write(checkpoint, state); continue
            plan.update(recovery_depth=depth, recovery_parent_id=parent)
            self.progress(stage='screening', pair=pair, attempted=sum(counts.values()),
                          remaining=len(state['pending']), resume_count=state['resume_count'])
            old = cache.get(key)
            if old and self.screening.cache_matches(plan, old[0], norm):
                row = copy.deepcopy(old[0])
                provenance = dict(reused=True, file=old[1], sha256=old[2], exact_interval=True)
                counts['reused'] += 1
            else:
                if not self.models:
                    self.progress(stage='loading_models', pair=pair)
                    self.models = [self.screening.full.legacy.EventModel(
                        self.screening.full.legacy.MODEL, device='cpu', batch_size=4),
                        self.screening.full.legacy.PannsModel(device='cpu', batch_size=4)]
                    self.check()
                self.progress(stage='screening', pair=pair)
                row = self.screening.RAW_SCREEN(plan, norm, *self.models,
                                               self.feedback['rejects'], self.feedback['positives'])
                row = self.screening.original.assess(row, self.screening.OLD_FEEDBACK)
                provenance = dict(reused=False, fresh_qc=True, device='cpu',
                                  algorithm='unchanged QC3 + QC4')
                counts['fresh'] += 1
            row = self.screening.finish_row(plan, row, self.feedback)
            row['quality_provenance'] = provenance
            write(folder / 'qc' / (row['sample_id'] + '.json'), row)
            seen.add(key); children = []
            fallback = self.repair.quality_fallback(plan, row)
            if fallback:
                children = fallback
                for child in children:
                    assert plan['start'] <= child['start'] and child['end'] <= plan['end']
                diagnostic.setdefault('head_quality_fallback', []).append(dict(
                    attempted_id=plan['sample_id'], fallback_ids=[c['sample_id'] for c in children],
                    reason='optional_head_extension_failed_quality'))
            elif row['audio_final']['pool'] != 'keep_candidate':
                _, kind = self.repair.base.quality_blocks(row)
                if kind == 'localized_quality' and depth < 6:
                    children, detail = ctx.propose([row], domain=row)
                    diagnostic['blocked'].extend(detail['blocked'])
                    diagnostic['unresolved'].extend(detail['unresolved'])
                    children = [c for c in children if self.screening.interval_key(c) not in seen]
                    for child in children:
                        assert child['duration_s'] < row['duration_s'] - 1e-5
                elif kind == 'localized_quality':
                    diagnostic['unresolved'].append(dict(start=row['start'], end=row['end'],
                                                         reason='recovery_depth_limit'))
            state['pending'].pop(0)
            state['pending'].extend([[c, depth + 1, row['sample_id']] for c in children])
            (state['parents'] if children else state['leaves']).append(row)
            state.update(seen=list(seen), counts=dict(counts),
                         elapsed=elapsed_before + time.monotonic() - began)
            write(checkpoint, state)
            self.progress(stage='screening', pair=pair, attempted=sum(counts.values()),
                          remaining=len(state['pending']), resume_count=state['resume_count'])
        self.check()
        leaves = sorted(state['leaves'], key=lambda r: (r['start'], r['end']))
        assert all(a['end'] <= b['start'] + 1e-6 for a, b in zip(leaves, leaves[1:]))
        assert all(2 - 1e-6 <= r['duration_s'] <= 11.95 + 1e-6 for r in leaves)
        kept = [(r['start'], r['end']) for r in leaves if r['audio_final']['pool'] == 'keep_candidate']
        result = dict(pair=pair, revision=self.repair.REVISION, source_seconds=ctx.limit,
                      clips=leaves, superseded_attempts=state['parents'], initial_ids=state['initial_ids'],
                      diagnostic=diagnostic, regions=ctx.regions, source_gaps=ctx.source_gaps,
                      local_feedback_audit=[p.get('local_feedback_audit', []) for p in ctx.people],
                      counts=dict(counts), pool_counts=dict(collections.Counter(r['audio_final']['pool'] for r in leaves)),
                      kept_seconds=self.repair.base.seconds(kept), elapsed=state['elapsed'],
                      complete=True, training_admitted=False, cluster_contract=self.manifest['contract'],
                      checkpoint_resumes=state['resume_count'], device='cpu', render_media=False)
        write(commit, result)
        write(folder / 'complete.json', dict(contract=self.manifest['contract'], result_sha256=sha(commit)))
        return result
