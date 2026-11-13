"""Full-inventory input adapter for the unchanged checkpointed J engine."""
import copy
from pathlib import Path

from listen_pair_j_runtime import Engine
from listen_pair_j_queue import read, write, sha


class FullEngine(Engine):
    def __init__(self, work, check, progress):
        super().__init__(work, check, progress)
        # Newly required normalization stays in this run, never in production.
        self.screening.full.OUT = self.work
        self.screening.full.configure_normalization()

    def empty_result(self, item, reason=None, ctx=None, diagnostic=None):
        folder = self.work / 'results' / item['pair']
        diagnostic = diagnostic or dict(blocked=[], unresolved=[])
        if reason:
            diagnostic['unresolved'].append(dict(reason=reason))
        initial = dict(pair=item['pair'], clips=[], diagnostic=diagnostic,
                       cluster_contract=self.manifest['contract'])
        write(folder / 'initial.json', initial)
        result = dict(initial, revision=self.repair.REVISION, source_seconds=ctx.limit if ctx else None,
                      superseded_attempts=[], initial_ids=[], regions=ctx.regions if ctx else [],
                      source_gaps=ctx.source_gaps if ctx else [], counts={}, pool_counts={},
                      kept_seconds=0., elapsed=0., complete=True, training_admitted=False,
                      checkpoint_resumes=0, device='cpu', render_media=False,
                      source_status='unusable_source' if reason else 'no_eligible_candidates',
                      source_catalog_status=item.get('source_catalog_status'))
        write(folder / 'result.json', result)
        write(folder / 'complete.json', dict(contract=self.manifest['contract'],
                                            result_sha256=sha(folder / 'result.json')))
        return result

    def process(self, item):
        pair = item['pair']; self.check()
        # Only explicitly verified, completed J CPU pilot results are reusable.
        cached = self.manifest.get('completed_j_cache', {}).get(pair)
        folder = self.work / 'results' / pair
        if cached and not (folder / 'checkpoint.json').exists() and not (folder / 'result.json').exists():
            if sha(cached['result']) != cached['result_sha256']:
                raise ValueError('Completed pilot result changed')
            old = read(cached['result'])
            if not old.get('complete') or old['cluster_contract'] != cached['contract']:
                raise ValueError('Invalid pilot result')
            result = copy.deepcopy(old)
            result.update(cluster_contract=self.manifest['contract'], reused_completed_j_result=cached,
                          counts={'reused_completed_result_attempts': sum(old.get('counts', {}).values())},
                          source_catalog_status=item.get('source_catalog_status'))
            initial = read(cached['initial']); initial['cluster_contract'] = self.manifest['contract']
            write(folder / 'initial.json', initial); write(folder / 'result.json', result)
            write(folder / 'complete.json', dict(contract=self.manifest['contract'],
                                                result_sha256=sha(folder / 'result.json')))
            return result
        entry_path = self.work / 'source_inputs' / (pair + '.json')
        if entry_path.exists():
            record = read(entry_path)
            if record['contract'] != self.manifest['contract']:
                raise ValueError('Source input contract mismatch')
            self.manifest['source_inputs'][pair] = record['entry']
        else:
            entry = dict(self.manifest.get('known_snapshots', {}).get(pair, {}))
            self.manifest['source_inputs'][pair] = entry
            try:
                ctx, _ = self.context(item)
            except ValueError as e:
                if not str(e).startswith(self.screening.KNOWN_FAILURES):
                    raise
                return self.empty_result(item, reason=str(e))
            self.check(); candidates, diagnostic = ctx.propose([])
            if not candidates:
                return self.empty_result(item, ctx=ctx, diagnostic=diagnostic)
            roots = [Path(p) for p in self.manifest['baseline_roots']]
            norm_paths = [p / 'normalized_pairs' / (pair + '.json') for p in roots]
            norm_path = next((p for p in norm_paths if p.exists()), None)
            if norm_path is None:
                self.progress(stage='normalizing', pair=pair)
                self.screening.full.legacy.normalize(item, candidates)
                norm_path = self.work / 'normalized_pairs' / (pair + '.json')
            caches = [p / 'results' / (pair + '.json') for p in roots if (p / 'results' / (pair + '.json')).exists()]
            entry.update(norm=str(norm_path), cache_results=[str(p) for p in reversed(caches)],
                         files={str(p): sha(p) for p in caches + [norm_path]})
            write(entry_path, dict(contract=self.manifest['contract'], entry=entry))
        result = super().process(item)
        return result
