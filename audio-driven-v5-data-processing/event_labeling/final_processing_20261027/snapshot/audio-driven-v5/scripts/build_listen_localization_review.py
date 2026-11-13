"""Preserve window-local raw responses and expose mapped parent-clip candidates."""
import json, re, math, shutil
from statistics import median
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/listen_localization_20261024T070145Z'
OUT=ROOT/'data_previews/listen_benchmark_review/localization'

def seconds(value):
    if isinstance(value,bool):raise ValueError('boolean timestamp')
    if isinstance(value,str) and ':' in value:
        parts=value.split(':')
        if len(parts) not in (2,3):raise ValueError('timestamp components')
        value=sum(float(x)*60**i for i,x in enumerate(reversed(parts)))
    value=float(value)
    if not math.isfinite(value):raise ValueError('nonfinite timestamp')
    return value

def extract(text):
    # Only syntactic normalization: fences/prefix, root array, timestamp strings.
    # Never invent missing times, repair truncated JSON, or discard invalid intervals.
    cleaned=re.sub(r'^\s*(?:Assistant:\s*)?```(?:json)?\s*','',text,flags=re.I)
    cleaned=re.sub(r'\s*```\s*$','',cleaned)
    try:
        value=json.loads(cleaned)
        events=value if isinstance(value,list) else value['events']
        return events if isinstance(events,list) else [],True
    except (ValueError,KeyError,TypeError):return [],False

def read_result(path, duration, offset=0, window=None):
    if not path.exists():return {'status':'pending','events':[]}
    record=json.loads(path.read_text())['results'][0]
    raw=record.get('raw_response',{});text=raw.get('text','');events,parseable=extract(text)
    result={'status':record['status'],'raw':text,'finish':raw.get('finish_reason'),'error':record.get('error'),
            'parseable':parseable,'events':[],'window':window,'seconds':record.get('inference_wall_seconds')}
    request=Path(record['path']).with_name('request.json')
    if request.exists():result['prompt']=json.loads(request.read_text()).get('prompt')
    for ev in events:
        if not isinstance(ev,dict):continue
        item=dict(ev,window=window,valid=False)
        try:
            start,end=seconds(ev['start']),seconds(ev['end'])
            item.update(start=start+offset,end=end+offset,local_start=start,local_end=end,
                        valid=0<=start<end<=duration and raw.get('finish_reason')=='stop')
        except (ValueError,TypeError,KeyError):pass
        result['events'].append(item)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((OUT.parent/'review.json').read_text());old_samples={x['id']:x for x in old['samples']}
    feedback=json.loads((RUN/'human_feedback.json').read_text())
    notes={x['sample_id']:x['note'] for x in feedback['records']}
    samples=[];lookup={}
    for row in json.loads((RUN/'manifest50.json').read_text()):
        sid=row['sample_id'];prev=old_samples[sid];assert row['media_hashes']==prev['source_hashes']
        sample={'id':sid,'duration':row['duration_s'],'video':'../'+prev['video'],'audio':'../'+prev['audio'],
                'note':notes.get(sid,''),'windows':{'omni_windows':[],'vl_windows':[]},'diagnostic':{},'word_windows':[]}
        sample['baseline']=read_result(ROOT/'experiments/listen_native_simple50_20261024T064600Z/results/items'/sid/'native_simple_v2/summary.json',row['duration_s'])
        sample['vl_fullclip']=read_result(RUN/'vl_fullclip_results/items'/sid/'vl_windows/summary.json',row['duration_s'])
        for name in ['vllm_guided','vllm_free','hf_free']:
            sample['diagnostic'][name]=read_result(RUN/'diagnostic_results/items'/sid/name/'summary.json',row['duration_s'])
        alignment=RUN/'word_alignment'/(sid+'.json')
        sample['alignment']=json.loads(alignment.read_text()) if alignment.exists() else {'status':'pending','words':[]}
        # Keep the original historical visual reference, including its raw source link.
        sample['visual_reference']=old['results']['qwen3_vl_visual'][sid]['discovery']
        sample['head_pitch']=[]
        face_path=OUT.parent/'raw/face_landmarker'/(sid+'_auxiliary.json')
        if face_path.exists():
            face=json.loads(face_path.read_text())
            for frame in face.get('raw_response',{}).get('frames',[]):
                matrices=frame.get('transformation_matrices',[])
                if frame.get('face_detected') and matrices:
                    matrix=matrices[0]
                    sample['head_pitch'].append([frame['time_s'],math.degrees(math.atan2(matrix[2][1],matrix[2][2]))])
        samples.append(sample);lookup[sid]=sample
    stats={name:Counter() for name in ['omni_windows','vl_windows']}
    for row in json.loads((RUN/'windows50.json').read_text()):
        word_path=RUN/'window_word_alignment'/(row['sample_id']+'.json')
        if word_path.exists():
            words=json.loads(word_path.read_text());words.pop('input_evidence',None)
            words['window']=row['window'];lookup[row['parent_id']]['word_windows'].append(words)
        for name in stats:
            result=read_result(RUN/'window_results/items'/row['sample_id']/name/'summary.json',row['duration_s'],row['window']['start_s'],row['window'])
            result['id']=row['sample_id'];lookup[row['parent_id']]['windows'][name].append(result)
            result['window']=row['window']
            stats[name][result['status']]+=1
            stats[name]['valid_candidate_rows']+=sum(x['valid'] for x in result['events'])
            stats[name]['invalid_candidate_rows']+=sum(not x['valid'] for x in result['events'])
            stats[name]['truncated']+=result.get('finish')=='length'
    for name in stats:
        stats[name]['positive_parent_clips']=sum(any(e['valid'] for w in s['windows'][name] for e in w['events']) for s in samples)
    for sample in samples:
        clusters=[];invalid=[]
        for w in sample['word_windows']:
            offset=w['window']['start_s'];duration=w['window']['end_s']-offset
            for word in w['words']:
                start,end=word['start_time'],word['end_time']
                if not 0<=start<end<=duration:
                    invalid.append(dict(word,window=w['window']));continue
                candidate=dict(start=start+offset,end=end+offset,text=word['text'],window=w['window'])
                key=re.sub(r'\W','',word['text']).casefold()
                found=next((c for c in clusters if c['key']==key and all(abs(x['start']-candidate['start'])<=.2+1e-8 and abs(x['end']-candidate['end'])<=.2+1e-8 and min(x['end'],candidate['end'])>max(x['start'],candidate['start']) and x['window']!=candidate['window'] for x in c['sources'])),None)
                if found is None:clusters.append({'key':key,'sources':[candidate]})
                else:found['sources'].append(candidate)
        sample['word_candidates']=sorted([dict(text=c['sources'][0]['text'],start=median(x['start'] for x in c['sources']),end=median(x['end'] for x in c['sources']),support=len(c['sources']),sources=c['sources']) for c in clusters],key=lambda c:c['start'])
        sample['invalid_words']=invalid
    samples.sort(key=lambda s:not bool(s['note']))
    data={'run':RUN.name,'samples':samples,'stats':stats,'human_ground_truth':False,'tolerance_s':.2,
          'analysis':json.loads((RUN/'analysis.json').read_text()) if (RUN/'analysis.json').exists() else {'paragraphs':['测试进行中；重叠窗口条目可能重复，不等于独立事件数或召回率。']}}
    (OUT/'review.json').write_text(json.dumps(data,ensure_ascii=False))
    html=(ROOT/'scripts/listen_localization_review.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(html)
    print(json.dumps({'stats':stats,'alignments':sum(s['alignment']['status']!='pending' for s in samples)}))

if __name__=='__main__':main()
