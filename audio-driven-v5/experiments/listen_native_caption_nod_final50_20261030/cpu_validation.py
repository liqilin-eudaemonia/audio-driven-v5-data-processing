import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.native_caption_nod import render
from listen_benchmark.native_caption_quality import generation_failures
rows=json.loads((Path(__file__).parent/'manifest.json').read_text())
for r in rows:
 text=render(r)
 assert 'listener_audio_is_silent: '+json.dumps(r['audio_preflight']['full_silence']) in text
 assert 'PRIOR PARAGRAPH' not in text
 assert all(set(c)=={'id','start','end','observation'} for c in json.loads(text.split('SUPPLEMENTARY NODDING INTERVALS:\n')[1]))
assert not generation_failures('The listener nods and says "yeah, yeah, yeah, yeah, yeah".', 'stop')
assert 'repeated_generation' in generation_failures('The listener looks down and then looks back up again. '*3,'stop')
assert 'generation_not_finished' in generation_failures('The listener nods','length')
assert 'empty_caption' in generation_failures(' ','stop')
old=ROOT/'experiments/listen_native_caption_nod_direct50_20261030/results'
failed=[]
for r in rows:
 x=json.loads((old/(r['sample_id']+'.json')).read_text()); f=generation_failures(x['caption'],x['finish_reason'])
 if f:failed.append(dict(sample_id=r['sample_id'],failures=f))
assert len(failed)==1 and 'S1806' in failed[0]['sample_id'],failed
report=dict(passed=True,prompts_checked=len(rows),prior_generation_failures_detected=failed,legitimate_short_word_repetition_preserved=True)
(Path(__file__).parent/'cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
