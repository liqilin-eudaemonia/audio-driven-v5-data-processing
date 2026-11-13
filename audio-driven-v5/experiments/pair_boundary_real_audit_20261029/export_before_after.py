"""Matched-start source excerpts before and after the current production cut."""
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent/'review'
result={}
for n in (1,2):
    evidence=json.loads((OUT/f'case{n}_evidence.json').read_text())
    short=json.loads((OUT/'closeups.json').read_text())[f'case{n}']
    current=next(c for c in evidence['clips'] if c['version']=='current')
    start=max(short['source_start'],current['start'])
    sides={}
    for side,end in [('before',short['source_end']),('after',current['end'])]:
        target=OUT/f'case{n}_{side}_local.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start-evidence['context'][0]),
            '-i',str(OUT/f'case{n}_context.mp4'),'-t',str(end-start),'-c:v','libx264',
            '-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)],check=True)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(target),'-f','null','-'],check=True)
        poster=OUT/f'case{n}_{side}_local.jpg'
        subprocess.run(['ffmpeg','-v','error','-y','-ss','0.1','-i',str(target),'-frames:v','1',str(poster)],check=True)
        sides[side]=dict(file=target.name,poster=poster.name,start=start,end=end,duration=end-start)
    result[f'case{n}']=dict(sides,definition='Before: continuous source context. After: the same context restricted to the actual production pair; no repair applied.',
                           omitted_relative_start=evidence['finding'].get('added_words',[{'start':335.787}])[0]['start']-start)
(OUT/'before_after.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Four matched-start before/after excerpts exported and decoded.')
