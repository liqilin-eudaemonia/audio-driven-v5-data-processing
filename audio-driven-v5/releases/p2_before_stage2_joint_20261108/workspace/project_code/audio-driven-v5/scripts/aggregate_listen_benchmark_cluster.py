#!/usr/bin/env python3
"""Export comparable summaries without merging engine versions or hiding failures."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,digest,read
from listen_labeling.schema import CLASSES
from listen_benchmark.runner import select_samples
from listen_benchmark.evaluate import subtract


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
    samples=select_samples(root/'manifest50.json',None);all_ids={s['sample_id'] for s in samples}
    phases={'transformers_v2':root/'full50_v2/results','vllm_v3':root/'vllm_v3/full50',
            'tail_fixed_v4':root/'tail_v4/results','captioner_template_v6':root/'captioner_template_v6/full50',
            'native_vllm_v5':root/'vllm_native_v5/full50'}
    if (root/'phases.json').exists():
        phases={name:root/path for name,path in json.loads((root/'phases.json').read_text()).items()}
    rows=[];failures=[]
    for phase,folder in phases.items():
        grouped={};identities={}
        for path in folder.glob('items/*/*/summary.json'):
            summary=read(path);name=summary['system'];grouped.setdefault(name,[]).extend(summary['results'])
            identities.setdefault(name,[]).append(read(path.parent/'inputs.json'))
        for name,results in grouped.items():
            first=identities[name][0];system=first['system'];stages=['auxiliary'] if system['auxiliary'] else ['discovery','audit']
            if len({(r['sample_id'],r['stage']) for r in results})!=len(results):raise ValueError('Duplicate stage results')
            if any(r['sample_id'] not in all_ids for r in results):raise ValueError('Unexpected sample')
            config_ids={digest({k:v for k,v in identity.items() if k not in ('samples','run_id')}) for identity in identities[name]}
            if len(config_ids)!=1:raise ValueError('Mixed configuration/code/model identity within phase')
            identity=dict(first,samples=samples,per_sample_run_ids={i['samples'][0]['sample_id']:i['run_id'] for i in identities[name]})
            identity.pop('run_id',None);run_id=digest(identity);identity['run_id']=run_id
            by_id={}
            for result in results:by_id.setdefault(result['sample_id'],{})[result['stage']]=result
            complete=sum(set(item)==set(stages) for item in by_id.values())
            passed=sum(set(item)==set(stages) and all(r['status']=='schema_pass' for r in item.values()) for item in by_id.values())
            aux=sum(any(r['status']=='auxiliary_only' for r in item.values()) for item in by_id.values())
            empty_aux=sum(not r.get('raw_response',{}).get('text','').strip() for r in results) if system['backend']=='captioner' else 0
            supported=0;support_seconds=Counter();errors=Counter()
            for item in by_id.values():
                audit=item.get('audit',{})
                if audit.get('status')=='schema_pass':
                    durations=[]
                    for label in CLASSES[system['branch']]:
                        value=audit['parsed'];excluded=value['unknown'][label]+[[e['start'],e['end']] for e in value['events'] if e['type']==label and e['boundary_censored']]
                        seconds=sum(b-a for a,b in subtract(value['reviewed'][label],excluded));support_seconds[label]+=seconds;durations.append(seconds)
                    supported+=int(all(x>0 for x in durations))
                for result in item.values():
                    if result['status']=='failed':
                        message=result.get('error',{}).get('message','unknown');errors[message]+=1
                        failures.append(dict(phase=phase,system=name,sample_id=result['sample_id'],stage=result['stage'],error=message,path=result['path']))
            destination=root/'comparisons'/phase/name
            atomic_json(destination/'inputs.json',identity)
            atomic_json(destination/'summary.json',dict(schema='listen-benchmark-v1',run_id=run_id,system=name,branch=system['branch'],auxiliary=system['auxiliary'],expected_requests=len(samples)*len(stages),completed=len(results),complete=len(results)==len(samples)*len(stages),results=results,human_ground_truth=False,training_admitted=False))
            rows.append(dict(phase=phase,system=name,completed_samples=complete,expected_samples=len(samples),complete_schema_pass=passed,
                             auxiliary_completed=aux,empty_auxiliary_outputs=empty_aux,audit_supported_all_target_classes=supported,checked_support_seconds=dict(support_seconds),
                             failed_requests=sum(errors.values()),failure_reasons=dict(errors),completed_requests=len(results),
                             inference_wall_seconds_sum=sum(r.get('inference_wall_seconds',0) for r in results),summary=str(destination/'summary.json')))
    report=dict(utc=datetime.now(timezone.utc).isoformat(),rows=sorted(rows,key=lambda x:(x['phase'],x['system'])),
                human_ground_truth=False,training_admitted=False,notes=['结构通过和模型自报覆盖不等于事件准确率','不同引擎版本分开统计，不混合计分'])
    atomic_json(root/'comparison_latest.json',report);atomic_json(root/'failed_requests.json',failures)
    text=['# 三机事件打标对照：实时结果','',f"更新时间：{report['utc']}",'','所有结果仍是待审草稿；没有人工 GT 的 precision/recall。','',
          '| 运行 | 配置 | 已完成样本 | 两阶段结构通过 | audit 有效覆盖 | 失败请求 |','|---|---|---:|---:|---:|---:|']
    for row in report['rows']:
        value=str(row['complete_schema_pass']) if not row['auxiliary_completed'] else str(row['auxiliary_completed'])+'（辅助）'
        text.append(f"| {row['phase']} | {row['system']} | {row['completed_samples']}/{row['expected_samples']} | {value} | {row['audit_supported_all_target_classes']} | {row['failed_requests']} |")
    text+=['','原始回答、失败记录和每次调用的输入证据保存在对应 `items/<sample>/<system>/requests/`；统一评测入口在 `comparisons/`。']
    (root/'实时对照结果.md').write_text('\n'.join(text)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
