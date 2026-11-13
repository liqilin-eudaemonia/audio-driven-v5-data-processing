"""Prepare automatic integration inputs. No human-feedback file is read."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]

def regions(audio,rate):
    step=max(1,round(rate*.02));energy=np.array([np.sqrt(np.mean(audio[i:i+step]**2)) for i in range(0,len(audio),step)])
    if len(energy)==0 or energy.max()<1e-6:return []
    threshold=max(float(np.percentile(energy,20))*4,float(energy.max())*.04,1e-5)
    active=np.flatnonzero(energy>threshold);groups=[]
    for i in active:
        if groups and (i-groups[-1][-1])*step/rate<=.2:groups[-1].append(int(i))
        else:groups.append([int(i)])
    result=[]
    for g in groups:
        a=g[0]*step/rate;b=min(len(audio)/rate,(g[-1]+1)*step/rate)
        if b-a>=.06:result.append(dict(start=a,end=b,threshold=threshold,role='acoustic_activity_candidate_not_speech_truth'))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);rows=json.loads((ROOT/'experiments/listen_split_validation_20261024T074733Z/manifest50.json').read_text())
    align=ROOT/'experiments/listen_split_validation_20261024T074733Z/word_alignment';source=json.loads((ROOT/'experiments/listen_response_episodes_20261024T120000Z/review.json').read_text());byid={s['id']:s for s in source['samples']};jobs=[];samples=[]
    for row in rows:
        sid=row['sample_id'];audio,rate=sf.read(row['target_audio'],dtype='float32');assert audio.ndim==1
        duration=len(audio)/rate;bounds=regions(audio,rate);asr=[]
        for f in sorted(align.glob(sid+'*.json')):asr.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),result=json.loads(f.read_text())))
        samples.append(dict(sample_id=sid,audio=row['target_audio'],video=row['target_video'],duration_s=duration,regions=bounds,asr=asr,old_observations=byid[sid]['decisions']))
        spans=[(0.,duration,'full')]
        # Retain whole-clip listening; activity crops only add evidence, never gate recall.
        for j,r in enumerate(bounds):
            start=max(0,r['start']-.35);end=min(duration,r['end']+.35)
            if end-start>=.3 and end-start<duration-.1:spans.append((start,end,f'local{j:02d}'))
        for start,end,view in spans:jobs.append(dict(job_id=f'{sid}__{view}',sample_id=sid,audio=row['target_audio'],start=start,end=end,duration_s=end-start,view=view))
    (run/'inputs.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2));(run/'omni_jobs.json').write_text(json.dumps(jobs,indent=2));(run/'input_policy.json').write_text(json.dumps(dict(human_feedback_read=False,sample_specific_rules=False,source_manifest=str(ROOT/'experiments/listen_split_validation_20261024T074733Z/manifest50.json'),samples=len(samples),audio_recheck_jobs=len(jobs)),indent=2));print(len(samples),len(jobs))
if __name__=='__main__':main()
