"""Count one decision per clip/class; never match GT event times or repeat counts."""
from collections import defaultdict
import math
import statistics

from .contracts import CLASSES, KNOWN, SCHEMA, digest, labels, pair_key, validate_record


def metric_counts(tp, fp, fn):
    return {'precision': tp / (tp + fp) if tp + fp else (0.0 if fn else None),
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def modality(records, mod, classes_supported, masks=None):
    capability = {r['capabilities'][mod] for r in records}
    if len(capability) != 1: raise ValueError('A method must declare consistent modality support')
    if not next(iter(capability)):
        return {'status': 'not_applicable', 'reason': 'method_does_not_generate_' + mod,
                'macro_classes': classes_supported, 'full_dataset_metrics': None}
    per_class = {}; complete = True
    for c in CLASSES[mod]:
        allowed = None if masks is None else set(masks[c])
        valid = []; gt_unknown = []; pred_unknown = []; gt_error = []; pred_error = []
        for r in records:
            g = r['reference'][mod][c]['state']; p = r['generated'][mod][c]['state']; sid = r['id']
            if g == 'unknown': gt_unknown.append(sid)
            if p == 'unknown': pred_unknown.append(sid)
            if g == 'error': gt_error.append(sid)
            if p == 'error': pred_error.append(sid)
            if g in KNOWN and p in KNOWN and (allowed is None or sid in allowed): valid.append(r)
        if len(valid) != len(records): complete = False
        tp = fp = fn = tn = 0
        for r in valid:
            g = r['reference'][mod][c]['state'] == 'present'; p = r['generated'][mod][c]['state'] == 'present'
            tp += int(g and p); fp += int(not g and p); fn += int(g and not p); tn += int(not g and not p)
        ids = sorted(r['id'] for r in valid)
        per_class[c] = dict(tp=tp, fp=fp, fn=fn, tn=tn, **metric_counts(tp, fp, fn),
            requested_pairs=len(records), evaluated_pairs=len(valid), evaluated_ids=ids, evaluated_ids_sha256=digest(ids),
            reference_positive_all=sum(r['reference'][mod][c]['state'] == 'present' for r in records),
            reference_negative_all=sum(r['reference'][mod][c]['state'] == 'absent' for r in records),
            reference_positive_evaluated=tp+fn, reference_negative_evaluated=fp+tn,
            predicted_positive_evaluated=tp+fp, false_positive_rate=fp/(fp+tn) if fp+tn else None,
            reference_unknown_ids=gt_unknown, prediction_unknown_ids=pred_unknown,
            reference_error_ids=gt_error, prediction_error_ids=pred_error,
            macro_eligible=c in classes_supported)
    # Classes stay fixed BEFORE seeing generation/abstention. No dropping a lost class.
    missing = [c for c in classes_supported if per_class[c]['reference_positive_evaluated'] == 0]
    macro = {name: None for name in ('macro_precision', 'macro_recall', 'macro_f1')}
    if classes_supported and not missing:
        for field in ('precision', 'recall', 'f1'):
            macro['macro_' + field] = sum(per_class[c][field] for c in classes_supported) / len(classes_supported)
    sums = {key: sum(per_class[c][key] for c in classes_supported) for key in ('tp', 'fp', 'fn')}
    subset = dict(**macro, micro=metric_counts(**sums), macro_classes=classes_supported,
                  macro_class_count=len(classes_supported), missing_supported_classes=missing)
    return dict(status='complete' if complete else 'partial', requested_pairs=len(records),
        macro_classes=classes_supported, full_dataset_metrics=subset if complete else None,
        available_subset_metrics=subset, per_class=per_class,
        no_reference_positive_classes=[c for c in CLASSES[mod] if c not in classes_supported],
        reference_all_empty_pairs=sum(all(r['reference'][mod][c]['state'] == 'absent' for c in CLASSES[mod]) for r in records),
        note='Subset metrics are conditional. TN and double-empty clips do not increase F1. Unknown/error are never negatives.')


def aggregate_draws(draws, field='full_dataset_metrics'):
    result = {}
    for mod in CLASSES:
        result[mod] = {}
        for metric in ('macro_precision', 'macro_recall', 'macro_f1'):
            values = [d[mod].get(field, {}).get(metric) if d[mod].get(field) is not None else None for d in draws]
            known = [v for v in values if v is not None]
            result[mod][metric] = dict(values=values, draws=len(values), valid_draws=len(known),
                mean=statistics.mean(known) if known and len(known) == len(values) else None,
                sample_std=statistics.stdev(known) if len(known) > 1 and len(known) == len(values)
                    else (0.0 if len(known) == len(values) == 1 else None))
    return result


def score(records, expected=None):
    records = [validate_record(r) for r in records]
    if not records: raise ValueError('No event records')
    index = {}
    for r in records:
        k = pair_key(r)
        if k in index: raise ValueError('Duplicate dataset/id/method/draw: ' + str(k))
        index[k] = r
    if len({r['protocol_identity'] for r in records}) != 1: raise ValueError('Cannot mix event protocols')
    refs = {}; ref_ident = {}
    for r in records:
        k = (r['dataset'], r['id']); identity = digest([r['reference_identity'], r['reference']])
        if k in ref_ident and ref_ident[k] != identity: raise ValueError('Inconsistent reference labels or media across methods/draws')
        refs[k] = r; ref_ident[k] = identity
    missing_records = []
    if expected is not None:
        seen = set()
        for item in expected:
            k = pair_key(item)
            if k in seen: raise ValueError('Duplicate frozen manifest key')
            seen.add(k)
            if k not in index:
                reference = refs.get(k[:2])
                if reference is None: raise ValueError('Reference labels missing entirely: ' + str(k[:2]))
                r = dict(reference, dataset=k[0], id=k[1], method=k[2], draw_id=k[3],
                         capabilities=item['capabilities'], generated=labels('error', 'missing_detection_record'))
                index[k] = validate_record(r); missing_records.append(k)
        if set(index) - seen: raise ValueError('Event records outside frozen manifest')
    by_dataset = defaultdict(list)
    for r in index.values(): by_dataset[r['dataset']].append(r)
    datasets = {}
    for dataset, items in sorted(by_dataset.items()):
        ids = sorted({r['id'] for r in items}); methods = sorted({r['method'] for r in items})
        supported = {mod: [c for c in cs if any(refs[(dataset, sid)]['reference'][mod][c]['state'] == 'present' for sid in ids)]
                     for mod, cs in CLASSES.items()}
        groups = defaultdict(list)
        for r in items: groups[(r['method'], r['draw_id'])].append(r)
        # A missing sample cannot silently redefine any system's requested population.
        for key, group in groups.items():
            if sorted(r['id'] for r in group) != ids: raise ValueError('Incomplete method/draw population; supply frozen expected manifest')
        draw_sets = {m: sorted(d for mm, d in groups if mm == m) for m in methods}
        if len({tuple(v) for v in draw_sets.values()}) != 1: raise ValueError('Methods have different draw sets')
        common = {}; results = {}
        for draw in next(iter(draw_sets.values())):
            common[draw] = {}
            for mod, cs in CLASSES.items():
                capable = [m for m in methods if groups[(m, draw)][0]['capabilities'][mod]]
                common[draw][mod] = {}
                for c in cs:
                    common[draw][mod][c] = [sid for sid in ids if refs[(dataset, sid)]['reference'][mod][c]['state'] in KNOWN
                        and all(index[(dataset, sid, m, draw)]['generated'][mod][c]['state'] in KNOWN for m in capable)]
        for method in methods:
            outputs = []; comparison = []
            for draw in draw_sets[method]:
                group = sorted(groups[(method, draw)], key=lambda x: x['id'])
                outputs.append(dict(draw_id=draw, **{mod: modality(group, mod, supported[mod]) for mod in CLASSES}))
                comparison.append(dict(draw_id=draw, **{mod: modality(group, mod, supported[mod], common[draw][mod]) for mod in CLASSES}))
            results[method] = dict(draws=outputs, full_dataset_draw_aggregate=aggregate_draws(outputs),
                common_subset_draws=comparison, common_subset_draw_aggregate=aggregate_draws(comparison, 'available_subset_metrics'))
        datasets[dataset] = dict(requested_ids=ids, requested_pairs=len(ids), reference_supported_classes=supported,
            methods=results, common_evaluable_ids_by_draw_class=common,
            reference_identities={sid: ref_ident[(dataset, sid)] for sid in ids})
    return dict(schema=SCHEMA, datasets=datasets, missing_detection_records=missing_records,
        protocol_identity=records[0]['protocol_identity'], timing_match=False, repeated_instances_counted_once=True,
        interpretation='Reference-class consistency only; not context appropriateness or within-generation AV coordination.',
        aggregation='Per-class corpus counts, reference-positive fixed macro classes; equal-weight per-draw scores. No GT-based selection.')
