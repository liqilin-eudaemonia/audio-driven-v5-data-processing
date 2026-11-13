"""Traceable reports and blind annotation import; no fabricated human scores."""
import csv
import io
from pathlib import Path

from .contracts import CLASSES, NAMES, SCHEMA, digest, labels, read, rows, save, save_rows, sha, validate_labels
from .media import verify
from .scoring import score


def collect(work):
    work=Path(work);contract,expected=verify(work);det=read(work/'detector_contract.json');records=[]
    prepared=read(work/'PREPARED.json')
    if sha(work/'prepared_pairs.jsonl')!=prepared['prepared_pairs_sha256']:raise ValueError('Prepared manifest changed')
    complete=read(work/'DETECTION_COMPLETE.json') if (work/'DETECTION_COMPLETE.json').exists() else None
    for row in rows(work/'prepared_pairs.jsonl'):
        sides={}
        for side,media in row['prepared_media'].items():
            path=Path(media['folder'])/'detection.json'
            if path.exists():
                d=read(path)
                if d['detector_identity']!=det['identity_hash'] or d['media_identity']!=media['identity']:
                    raise ValueError('Detection does not belong to this media/protocol')
                if complete and sha(path)!=complete['detections'][media['media_id']]:raise ValueError('Detection artifact changed')
                sides[side]=d['labels']
            else:sides[side]=labels('error','missing_detection_record')
            validate_labels(sides[side])
        records.append(dict(schema=SCHEMA,**{k:row[k] for k in ['dataset','id','method','draw_id','capabilities']},
            reference_identity=digest(row['prepared_media']['reference']['identity']),
            protocol_identity=digest([contract['policy_identity'],det['identity_hash']]),
            reference=sides['reference'],generated=sides['generated'],self_check_only=row['self_check_only'],
            media_ids={side:m['media_id'] for side,m in row['prepared_media'].items()}))
    return records,expected


def human_labels(path):
    """Disagreement stays unknown. No unapproved majority vote or missing=negative."""
    votes={};raters={}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('state'):continue
            mod=row['modality'];c=row['class'];state=row['state'];rater=row.get('rater_id','').strip()
            if mod not in CLASSES or c not in CLASSES[mod] or state not in {'present','absent','unknown'}:
                raise ValueError('Invalid manual class or state')
            if not rater or not row.get('evidence','').strip():raise ValueError('Manual rater and evidence required')
            key=(row['media_id'],mod,c)
            if rater in raters.setdefault(key,set()):raise ValueError('Duplicate rater/unit')
            raters[key].add(rater);votes.setdefault(key,[]).append((state,row['evidence']))
    output={};disagreements=0
    for (mid,mod,c),items in votes.items():
        target=output.setdefault(mid,labels())
        agree=len({s for s,_ in items})==1;disagreements+=not agree
        target[mod][c]=dict(state=items[0][0] if agree else 'unknown',
            evidence='manual consensus: '+' | '.join(e for _,e in items) if agree else 'manual rater disagreement',
            source='manual',raters=len(items))
    return output,dict(annotated_units=len(votes),disagreement_units=disagreements,
        units_with_multiple_raters=sum(len(v)>1 for v in votes.values()),adjudication='unanimous labels only; disagreement=unknown')


def review(work):
    work=Path(work);_,_=verify(work);media={m['media_id']:m for r in rows(work/'prepared_pairs.jsonl') for m in r['prepared_media'].values()}
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['media_id','modality','class','state','evidence','rater_id'])
    writer.writeheader();index=[]
    for mid,m in sorted(media.items()):
        index.append(dict(media_id=mid,video=m.get('visual25'),audio=m.get('audio'),
            duration_s=m['duration_s'],instructions='Label observable events only; no method/side shown. Empty state is unannotated, not absent.'))
        for mod,cs in CLASSES.items():
            for c in cs:writer.writerow(dict(media_id=mid,modality=mod,**{'class':c}))
    template=work/'human_labels.template.csv'
    if not template.exists():template.write_text(buffer.getvalue())
    save_rows(work/'blind_review_media.jsonl',index)
    save(work/'HUMAN_REVIEW_STATUS.json',dict(status='pending',blind_media=len(media),
        label_template=str(template.resolve()),automatic_labels_hidden=True,
        context_appropriateness='separate human protocol not defined here',within_generation_av_coordination='separate human protocol not defined here'))


def report(work, annotations=None):
    work=Path(work);records,expected=collect(work);summary=score(records,expected)
    partial=any(v[mod]['status']=='partial' for data in summary['datasets'].values() for result in data['methods'].values() for v in result['draws'] for mod in CLASSES)
    summary.update(status='partial' if partial else 'complete',detector_based=True,pilot_only=True,formal_benchmark_ready=False,
        self_check_only=all(r['self_check_only'] for r in records),human_reliability_validation='pending')
    for dataset,data in summary['datasets'].items():
        for method,result in data['methods'].items():
            result['self_check_only']=all(r['self_check_only'] for r in records if r['dataset']==dataset and r['method']==method)
    save_rows(work/'event_pairs.jsonl',records)
    if annotations:
        gold,audit=human_labels(annotations);tests=[]
        unique={mid:labels_ for r in records for side,mid in r['media_ids'].items() for labels_ in [r[side]]}
        if set(gold)-set(unique):raise ValueError('Human labels include unknown media IDs')
        for mid,values in gold.items():
            tests.append(dict(schema=SCHEMA,dataset='detector_reliability',id=mid,method='annotation_pipeline',draw_id='0',
                reference_identity=digest(values),protocol_identity=records[0]['protocol_identity'],capabilities=dict(visual=True,audio=True),
                reference=values,generated=unique[mid]))
        if tests:
            save(work/'human_detector_validation.json',dict(human_annotation_audit=audit,metrics=score(tests),
                calibration_performed=False,formal_benchmark_ready=False))
            summary['human_reliability_validation']='available_separate_report_not_threshold_calibration'
    save(work/'summary.json',summary)
    csvfile=io.StringIO();fieldnames=['dataset','method','draw','modality','class','macro_eligible','tp','fp','fn','tn',
        'precision','recall','f1','reference_positive_all','reference_positive_evaluated','evaluated_pairs','requested_pairs','false_positive_rate']
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames);writer.writeheader()
    lines=['# 整段事件类别一致性报告','',
        '**Pilot／外部检测结果，尚非正式benchmark。** 不评价语境适当性或生成内部视听协调性；不匹配GT时间，不因重复或双空加分。',
        '主宏平均类别由冻结参考正例支持确定；无参考正例类另报FP／误报率。未知和技术失败不作负例。',
        '人工核验：'+summary['human_reliability_validation']+'。GT自对照只验证执行一致性，不能证明检测语义准确。','']
    def value(x):return 'N/A' if x is None else f'{x:.4f}'
    for dataset,data in summary['datasets'].items():
        lines+=['## '+dataset,'','| 方法 | 模态 | 完整集Macro-P | 完整集Macro-R | 完整集Macro-F1 | 有效draw/请求draw |',
                '|---|---|---:|---:|---:|---:|']
        for method,result in data['methods'].items():
            label=method+(' [GT自对照，非模型结果]' if result['self_check_only'] else '')
            for mod in CLASSES:
                agg=result['full_dataset_draw_aggregate'][mod]
                lines.append('| '+label+' | '+mod+' | '+' | '.join(value(agg[k]['mean']) for k in ['macro_precision','macro_recall','macro_f1'])+
                    f' | {agg["macro_f1"]["valid_draws"]}/{agg["macro_f1"]["draws"]} |')
            for draw in result['draws']:
                for mod in CLASSES:
                    for c,item in draw[mod].get('per_class',{}).items():
                        writer.writerow(dict(dataset=dataset,method=method,draw=draw['draw_id'],modality=mod,
                            **{'class':c},**{k:item[k] for k in fieldnames if k in item}))
        lines+=['','逐类TP/FP/FN、参考支持、失败／未知ID及每类共同可评集合见summary.json；逐类数值见per_class.csv。',
                '跨方法共同子集另列在common_subset_draws，不能把不同子集的条件均值直接排名。','']
    (work/'per_class.csv').write_text(csvfile.getvalue());(work/'REPORT.md').write_text('\n'.join(lines)+'\n')
    review(work)
    if annotations:
        status=read(work/'HUMAN_REVIEW_STATUS.json');status.update(status='labels_imported',annotation_sha256=sha(annotations),calibration_status='pending')
        save(work/'HUMAN_REVIEW_STATUS.json',status)
    return summary
