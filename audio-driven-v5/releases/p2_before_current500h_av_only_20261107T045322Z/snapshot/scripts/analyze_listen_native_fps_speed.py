#!/usr/bin/env python3
"""Audit completed paired timing; reject partial runs and mismatched input pairs."""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiments/listen_visual_input50_20261029/native_fps_decision/speed'


def audit(folder):
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['state']=='complete','Do not turn partial timing into a comparison'
    measurements=json.loads((folder/'measurements.json').read_text())
    plan=json.loads((folder/'plan.json').read_text())
    blocks=sorted(set(m['batch'] for m in measurements));rows=[]
    for b in blocks:
        ms=sorted((m for m in measurements if m['batch']==b),key=lambda m:m['pass'])
        expected=['video8','video16','video16','video8'] if b%2==0 else ['video16','video8','video8','video16']
        assert [m['arm'] for m in ms]==expected,(b,'order / repetitions')
        inputs=[[ (r['sample_id'],r['view'],r['duration']) for r in m['records']] for m in ms]
        assert all(x==inputs[0] for x in inputs),(b,'different input members or duration')
        row={'batch':b,'inputs':inputs[0]}
        for arm in ('video8','video16'):
            pair=[m for m in ms if m['arm']==arm]
            assert [[r['input_sha256'] for r in m['records']] for m in pair][0]==[[r['input_sha256'] for r in m['records']] for m in pair][1]
            t=[m['elapsed_s'] for m in pair];assert all(x>0 for x in t)
            row[arm]={'elapsed_s':t,'mean_seconds':statistics.mean(t),
                      'repeat_time_ratio':max(t)/min(t),
                      'output_tokens':[sum(r['output_tokens'] for r in m['records']) for m in pair],
                      'valid':[all(r['valid'] for r in m['records']) for m in pair]}
        row['ratio_16_to_8']=row['video16']['mean_seconds']/row['video8']['mean_seconds'];rows.append(row)
    ids={r['sample_id'] for m in measurements for r in m['records']}
    assert ids==set(plan['samples']),'Missing selected samples'
    for arm,totals in summary['totals'].items():
        ms=[m for m in measurements if m['arm']==arm];rs=[r for m in ms for r in m['records']]
        assert len(rs)==totals['requests']
        assert sum(r['valid'] for r in rs)==totals['valid_requests']
        assert abs(sum(m['elapsed_s'] for m in ms)-totals['seconds'])<1e-7
        if plan['views']=='full':assert all(r['view']=='full' for r in rs)
    ratios=[r['ratio_16_to_8'] for r in rows]
    process_peak=0;device_peak=0;util=[]
    for line in (folder/'resources.jsonl').read_text().splitlines():
        r=json.loads(line)
        if 'error' in r:continue
        gpu=[x.strip() for x in r['device'].split(',')]
        if len(gpu)==7:device_peak=max(device_peak,float(gpu[3]));util.append(float(gpu[5]))
        for p in r['processes'].splitlines():
            fields=[x.strip() for x in p.split(',',2)]
            if len(fields)==3 and 'VLLM' in fields[2]:
                try:process_peak=max(process_peak,float(fields[1]))
                except ValueError:pass
    result={'passed':True,'shared_gpu':plan['shared_gpu'],'views':plan['views'],'batch_size':plan['batch_size'],
            'samples':len(ids),'blocks':len(blocks),'model_response_semantics_not_scored':True,
            'ratio_16_to_8_total':summary['ratio_16_to_8_wall_time'],
            'ratio_16_to_8_median_block':statistics.median(ratios),'ratio_range':[min(ratios),max(ratios)],
            'blocks_16_slower':sum(r>1 for r in ratios),'per_block':rows,
            'max_repeat_time_ratio':max(r[a]['repeat_time_ratio'] for r in rows for a in ['video8','video16']),
            'sampled_vllm_process_peak_mib':process_peak,'sampled_total_device_peak_mib':device_peak,
            'mean_device_gpu_utilization_including_other_training':statistics.mean(util) if util else None,
            'limits':['Targeted GPU remains shared with training; timing ratios are conditional observations.',
                      'Different generated output lengths contribute to elapsed time; this is not pure encoder timing.',
                      'Only prepared-input generation is timed. No end-to-end or maximum-throughput claim.']}
    (folder/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',nargs='?',type=Path,default=DEFAULT)
    result=audit(parser.parse_args().folder)
    print(json.dumps({k:v for k,v in result.items() if k!='per_block'},ensure_ascii=False,indent=2))
