import json,random,html
from pathlib import Path
OUT=Path(__file__).resolve().parent;review=OUT/'review';review.mkdir(exist_ok=True);media=review/'media';media.mkdir(exist_ok=True)
observations=json.loads((OUT/'observations.json').read_text());targets={r['sample_id']:r for r in json.loads((OUT/'targets.json').read_text())}
import sys
ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import job_key
rows=[];mapping=[]
def link(name,path):
 d=media/name
 if not d.exists():d.symlink_to(Path(path).absolute())
 return 'media/'+name
for i,obs in enumerate(observations):
 sid=obs['sample_id'];gt=targets[sid]
 for seed in [137,251]:
  arms=['E0','E1','E0F','E1F'];random.Random(f'blind20261027:{sid}:{seed}').shuffle(arms)
  cards=[]
  for label,arm in zip('ABCD',arms):
   folder=OUT/arm/'generation/generated'/job_key(sid,seed);v=link(f'{i}-{seed}-{label}.mp4',folder/'generated.mp4');a=link(f'{i}-{seed}-{label}.wav',folder/'listener.wav')
   cards.append(f'<div><b>{label}</b><video controls preload="none" src="{v}"></video><audio controls preload="none" src="{a}"></audio></div>')
   mapping.append(dict(sample_id=sid,seed=seed,label=label,arm=arm))
  sv=link(f'{i}-speaker.wav',obs['speaker_audio']);tv=link(f'{i}-gt.mp4',gt['target_video']);ta=link(f'{i}-gt.wav',gt['target_audio'])
  rows.append(f'<section><h2>样本 {i+1} · seed {seed}</h2><p>{html.escape(sid)}</p><p>Speaker 条件音频 <audio controls preload="none" src="{sv}"></audio></p><div class="grid">'+''.join(cards)+f'</div><details><summary>显示原 listener 参考</summary><video controls preload="none" src="{tv}"></video><audio controls preload="none" src="{ta}"></audio></details><p>建议分别记录：自然度、反应时机、身份稳定、角色保持、音频干净程度；不要要求唯一参考的逐处复现。</p></section>')
(review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>训练集24条四组盲看</title><style>body{font-family:system-ui;max-width:1500px;margin:25px auto;background:#f4f4f4;color:#222}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}video{width:100%;max-width:350px}audio{max-width:100%;height:32px}section{padding:20px;background:white;margin:20px 0;border-radius:10px}p{overflow-wrap:anywhere}summary{cursor:pointer}</style><h1>24条训练样本 · 四组输出盲看</h1><p>每个样本/seed重新打乱A–D，训练条件映射另存。尚无人类盲评结果。视频含生成listener音频；请先听speaker条件，再分别看输出。</p>'+''.join(rows))
(OUT/'blind_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(review/'index.html')
