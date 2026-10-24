"""Review separate modality predictions and preserve provenance during integration."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.integration import integrate
from listen_benchmark.prompts import annotation_prompt
from build_listen_localization_review import read_result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);a=parser.parse_args()
    run=Path(a.run).resolve();out=ROOT/'data_previews/listen_benchmark_review/split_validation';out.mkdir(parents=True,exist_ok=True)
    old=json.loads((out.parent/'review.json').read_text());media={s['id']:s for s in old['samples']}
    previous=json.loads((out.parent/'localization/review.json').read_text());previous={s['id']:s for s in previous['samples']}
    notes={s['sample_id']:s['note'] for s in json.loads((run/'human_feedback.json').read_text())['records']}
    samples=[];lookup={};stats={k:Counter() for k in ['visual_windows','audio_windows','visual_full','audio_full']}
    def count(name,result):
        stats[name][result['status']]+=1
        stats[name]['valid_candidate_rows']+=sum(e['valid'] for e in result['events'])
        stats[name]['invalid_candidate_rows']+=sum(not e['valid'] for e in result['events'])
        stats[name]['truncated']+=result.get('finish')=='length'
    for row in json.loads((run/'manifest50.json').read_text()):
        sid=row['sample_id'];assert row['media_hashes']==media[sid]['source_hashes']
        sample=dict(id=sid,duration=row['duration_s'],video='../'+media[sid]['video'],audio='../'+media[sid]['audio'],
                    note=notes.get(sid,''),head_pitch=previous[sid]['head_pitch'],baseline=previous[sid]['baseline'],
                    windows={'visual_windows':[],'audio_windows':[]},word_windows=[])
        for name in ['visual_full','audio_full']:
            result=read_result(run/'results/items'/sid/name/'summary.json',row['duration_s'])
            sample[name]=result;count(name,result)
        path=run/'word_alignment'/(sid+'.json')
        sample['alignment']=json.loads(path.read_text()) if path.exists() else {'status':'pending','words':[]}
        sample['alignment'].pop('input_evidence',None)
        samples.append(sample);lookup[sid]=sample
    for row in json.loads((run/'windows50.json').read_text()):
        sample=lookup[row['parent_id']]
        for name in ['visual_windows','audio_windows']:
            result=read_result(run/'results/items'/row['sample_id']/name/'summary.json',row['duration_s'],row['window']['start_s'],row['window'])
            result.update(id=row['sample_id'],window=row['window']);sample['windows'][name].append(result);count(name,result)
        path=run/'word_alignment'/(row['sample_id']+'.json')
        if path.exists():
            result=json.loads(path.read_text());result.pop('input_evidence',None);result['window']=row['window'];sample['word_windows'].append(result)
    timelines=[]
    for s in samples:
        s['integrated']=integrate(s['windows']['visual_windows'],s['windows']['audio_windows'],s['word_windows'],s['duration'])
        s['word_candidates']=[dict(e,text=e['behavior']) for e in s['integrated']['events'] if e['source']=='asr_aligner']
        for name in stats:
            results=s['windows'][name] if name in s['windows'] else [s[name]]
            stats[name]['positive_parent_clips']+=any(e['valid'] for r in results for e in r['events'])
        timelines.append(dict(sample_id=s['id'],duration_s=s['duration'],**s['integrated']))
    samples.sort(key=lambda s:not bool(s['note']))
    analysis_path=run/'analysis.json'
    data={'run':run.name,'samples':samples,'stats':stats,'human_ground_truth':False,'tolerance_s':.2,
          'alignment_completed':sum(s['alignment']['status']!='pending' for s in samples)+sum(len(s['word_windows']) for s in samples),
          'analysis':json.loads(analysis_path.read_text()) if analysis_path.exists() else {'paragraphs':['验证进行中：视觉与音频独立识别，所有时间映射回原片；整合结果是待复核候选，不是人工真值。']}}
    config=json.loads((run/'config.json').read_text())
    data['prompts']={s['name']:annotation_prompt(s,3,'discovery') for s in config['systems'] if s['name'].endswith('windows')}
    (out/'review.json').write_text(json.dumps(data,ensure_ascii=False))
    (out/'candidate_timelines.json').write_text(json.dumps(dict(run_id=run.name,human_ground_truth=False,samples=timelines),ensure_ascii=False,indent=2))
    (run/'candidate_timelines.json').write_text((out/'candidate_timelines.json').read_text())
    html=(ROOT/'scripts/listen_split_review.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (out/'index.html').write_text(html)
    print(json.dumps({'stats':stats,'alignment_completed':data['alignment_completed']}))

if __name__=='__main__':main()
