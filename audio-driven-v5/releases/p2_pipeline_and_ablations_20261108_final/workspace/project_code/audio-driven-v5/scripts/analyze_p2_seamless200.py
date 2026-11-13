"""Read the finished ten-way benchmark; paired analysis and explicit demo sampling."""
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
OUT=ROOT/'experiments/p2_seamless200_analysis_20261105'
SELECTION=ROOT/'experiments/seamless_quick200_latest_20261105/selection.jsonl'
LABELS={
 'av_only_6000':'AV · 6000','av_only_8927':'AV · 8927',
 'av_event_6000':'AV＋Event · 6000','av_event_8927':'AV＋Event · 8927',
 'base_feature_6000':'AV＋Feature · 6000','base_feature_8927':'AV＋Feature · 8927',
 'base_event_feature_7000':'AV＋Event＋Feature · 7000',
 'sft_event_6000':'SFT H＋Event · 6000','sft_event_8927':'SFT H＋Event · 8927',
 'sft_on_base_event_6000':'SFT规划器＋原始H Event · 6000'}


def load():
 protocol=json.loads((RUN/'protocol.json').read_text())
 summary=json.loads((RUN/'RESULTS.json').read_text())
 selection={x['id']:x for x in map(json.loads,SELECTION.read_text().splitlines())}
 video={};audio={};plans={}
 for arm in protocol['arms']:
  name=arm['name'];folder=RUN/'metrics'/name/'seamless'
  video[name]={x['id']:x for x in map(json.loads,(folder/'pairs.jsonl').read_text().splitlines())}
  audio[name]={x['id']:x for x in map(json.loads,(folder/'audio_pairs.jsonl').read_text().splitlines())}
  assert set(video[name])==set(audio[name])==set(selection)
 for group in ('base_standard','sft_event'):
  plans[group]={sid:json.loads((RUN/group/'conditions/seamless'/sid/'qwen/plan.json').read_text()) for sid in selection}
 return protocol,summary,selection,video,audio,plans


def contribution(row,metric):
 if metric not in row['metrics']:return None
 if metric in ('psnr','ssim','lpips'):weight=row['media']['aligned_frames']
 elif metric=='emo_acc':return row['emotion']['matches'],row['emotion']['valid_frames']
 else:weight=row['response']['valid_phase_duration']
 return row['metrics'][metric]*weight,weight


def paired(left,right,metric,selection):
 ids=[sid for sid in selection if contribution(left[sid],metric) and contribution(right[sid],metric)]
 groups=sorted({selection[sid]['group_id'] for sid in ids});index={g:i for i,g in enumerate(groups)}
 sums=np.zeros((len(groups),4),np.float64)
 for sid in ids:
  sums[index[selection[sid]['group_id']]]+=np.array([*contribution(left[sid],metric),*contribution(right[sid],metric)])
 rng=np.random.default_rng(20261105)
 weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
 draws=weights@sums;point=sums.sum(0)
 differences=draws[:,2]/draws[:,3]-draws[:,0]/draws[:,1]
 return dict(pairs=len(ids),source_groups=len(groups),difference_right_minus_left=float(point[2]/point[3]-point[0]/point[1]),
   ci95=[float(x) for x in np.quantile(differences,[.025,.975])],bootstrap_repeats=2000,
   support='common valid clips; each metric keeps its original frame/phase weighting; no distribution-metric bootstrap')


def select_cases(selection,video,audio):
 ids=list(selection);used=set();groups=set();chosen=[]
 E='av_event_6000';F='base_feature_6000';C='base_event_feature_7000';S='sft_on_base_event_6000';H='sft_event_6000'
 def m(arm,sid,key):return video[arm][sid]['metrics'].get(key,float('nan'))
 def ac(arm,sid):return audio[arm][sid]['audio']['comparison']['acoustic']
 def pick(category,title,a,b,score,valid=lambda sid:True,rule='',large=True):
  candidates=[sid for sid in ids if sid not in used and selection[sid]['group_id'] not in groups and valid(sid)
              and (not large or selection[sid]['face_p10_px']>=224) and np.isfinite(score(sid))]
  if not candidates:
   candidates=[sid for sid in ids if sid not in used and valid(sid) and np.isfinite(score(sid))]
  if not candidates:raise ValueError('No case available for '+title)
  sid=max(candidates,key=score);used.add(sid);groups.add(selection[sid]['group_id'])
  chosen.append(dict(id=sid,index=len(chosen)+1,category=category,title=title,a=a,b=b,selection_rule=rule,
                     selection_score=float(score(sid)),selection_kind='purposeful_metric_selection',
                     human_visual_verification=False))
 pick('组合监督','组合监督：画面相似度改善较明显',F,C,lambda s:m(F,s,'lpips')-m(C,s,'lpips'),rule='最大化 Feature6000 LPIPS − 组合7000 LPIPS；先取不同源group和人脸≥224')
 pick('组合监督','组合监督：画面相似度反而下降',F,C,lambda s:m(C,s,'lpips')-m(F,s,'lpips'),rule='最大化组合7000 LPIPS − Feature6000 LPIPS；作为反例')
 pick('单项损失','Feature：相对Event的画面相似度优势',E,F,lambda s:m(E,s,'lpips')-m(F,s,'lpips'),rule='最大化 Event6000 LPIPS − Feature6000 LPIPS')
 pick('单项损失','Event：相对Feature的运动幅度优势',F,E,lambda s:m(F,s,'delta_resp')-m(E,s,'delta_resp'),rule='最大化 Feature6000 ΔResp − Event6000 ΔResp')
 pick('SFT对照','更换SFT规划器：表情一致率提高',E,S,lambda s:m(S,s,'emo_acc')-m(E,s,'emo_acc'),rule='最大化 SFT规划器交叉组 EmoAcc − 原始Event6000 EmoAcc')
 pick('SFT对照','更换SFT规划器：运动误差增加',E,S,lambda s:m(S,s,'delta_resp')-m(E,s,'delta_resp'),rule='最大化 SFT规划器交叉组 ΔResp − 原始Event6000 ΔResp')
 pick('SFT H对照','同为SFT规划器：SFT H表情优势例',S,H,lambda s:m(H,s,'emo_acc')-m(S,s,'emo_acc'),rule='同一SFT规划器，比较SFT H训练与原始H训练的表情一致率')
 pick('SFT H对照','同为SFT规划器：SFT H运动误差例',S,H,lambda s:m(H,s,'delta_resp')-m(S,s,'delta_resp'),rule='最大化 SFT H6000 ΔResp − SFT规划器交叉组 ΔResp')
 pick('音频检查','GT有回应：SFT规划器找回活动匹配',E,S,
      lambda s:ac(S,s)['tolerant']['tp']+sum(x['tiou'] for x in ac(S,s)['tolerant']['matches']),
      valid=lambda s:ac(E,s)['missed_entire_response'] and ac(S,s)['tolerant']['tp']>0,
      rule='原始Event完全无声而SFT交叉组至少一条活动通过±0.2秒或tIoU匹配；不据此认定说词正确',large=False)
 pick('音频检查','GT严格全零：SFT出现额外声音',E,S,lambda s:ac(S,s)['generated_active_s'],
      valid=lambda s:selection[s]['audio_group']=='silent' and not ac(E,s)['generated_intervals'] and bool(ac(S,s)['generated_intervals']),
      rule='原始净化GT波形严格全零，原始Event无声，SFT交叉组有声学活动',large=False)
 pick('共同失败','GT有声音：十组都未产生声学活动',E,H,lambda s:ac(E,s)['gt_active_s'],
      valid=lambda s:ac(E,s)['gt_active_s']>.4 and all(not ac(a,s)['generated_intervals'] for a in video),
      rule='GT活动>0.4秒且十组生成声学活动均为空；重点试听，不把GT声音直接视为词级真值',large=False)
 pick('安静倾听','GT静音：比较安静倾听的画面',C,S,lambda s:-abs(m(C,s,'lpips')-.35),
      valid=lambda s:selection[s]['audio_group']=='silent' and not ac(C,s)['generated_intervals'] and not ac(S,s)['generated_intervals'],
      rule='GT全零、两组均无声，选组合LPIPS接近0.35的不同源样本')
 pick('训练步数','Event：6000与8927步的差异',E,'av_event_8927',lambda s:m(E,s,'lpips')-m('av_event_8927',s,'lpips'),rule='最大化Event6000 LPIPS − Event8927 LPIPS')
 pick('训练步数','Feature：继续训练后的运动变化',F,'base_feature_8927',lambda s:m(F,s,'delta_resp')-m('base_feature_8927',s,'delta_resp'),rule='最大化Feature6000 ΔResp − Feature8927 ΔResp')
 rng=np.random.default_rng(20261105)
 for silent in (True,False):
  candidates=[s for s in ids if s not in used and selection[s]['group_id'] not in groups and (selection[s]['audio_group']=='silent')==silent]
  if not candidates:candidates=[s for s in ids if s not in used and (selection[s]['audio_group']=='silent')==silent]
  sid=str(rng.choice(candidates));used.add(sid);groups.add(selection[sid]['group_id'])
  chosen.append(dict(id=sid,index=len(chosen)+1,category='随机核对',title='固定种子随机抽查：'+('GT静音' if silent else 'GT非静音'),a='av_only_6000',b=C,
       selection_rule='seed20261105，在未展示的对应音频分层中随机抽取，优先不同源group；未按模型指标择优',selection_kind='seeded_random',human_visual_verification=False))
 assert len(chosen)==len(used)==16
 return chosen


def main():
 OUT.mkdir(exist_ok=True)
 p,summary,selection,video,audio,plans=load()
 comparisons=[('Event6000_vs_AV6000','av_only_6000','av_event_6000'),('Feature6000_vs_AV6000','av_only_6000','base_feature_6000'),
  ('EventFinal_vs_AVFinal','av_only_8927','av_event_8927'),('FeatureFinal_vs_AVFinal','av_only_8927','base_feature_8927'),
  ('Combined_vs_Event6000','av_event_6000','base_event_feature_7000'),('Combined_vs_Feature6000','base_feature_6000','base_event_feature_7000'),
  ('SFTplanner_vs_baseEvent','av_event_6000','sft_on_base_event_6000'),('SFTH_vs_sameSFTplanner','sft_on_base_event_6000','sft_event_6000'),
  ('SFTH_final_vs_6000','sft_event_6000','sft_event_8927')]
 paired_results={key:dict(left=a,right=b,metrics={m:paired(video[a],video[b],m,selection) for m in ['psnr','ssim','lpips','emo_acc','delta_resp']}) for key,a,b in comparisons}
 (OUT/'paired_bootstrap.json').write_text(json.dumps(paired_results,indent=2)+'\n')
 records=[]
 for arm in p['arms']:
  name=arm['name'];s=summary[name+'/seamless'];activity=s['audio']['activity']['acoustic']
  records.append(dict(arm=name,label=LABELS[name],step=arm['step'],**s['metrics'],
   generated_active_clips=sum(bool(x['audio']['generated']['acoustic']) for x in audio[name].values()),
   gt_active_clips=activity['active_reference_pairs'],gt_silent_clips=activity['silent_reference_pairs'],
   completely_silent_on_active_gt=activity['missed_entire_response'],missed_response_rate=activity['missed_response_rate'],
   false_sound_on_silent_gt=activity['false_activity_on_silent_clip'],false_sound_clip_rate=activity['false_sound_clip_rate'],
   strict_audio_f1=activity['strict']['f1'],tolerant_audio_f1=activity['tolerant']['f1'],
   gt_active_s=activity['gt_active_s'],generated_active_s=activity['generated_active_s'],
   emotion_pairs=s['coverage']['emotion']['video_identities'],motion_phases=s['coverage']['delta_resp']['valid_phases']))
 (OUT/'summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 selected=select_cases(selection,video,audio)
 (OUT/'showcase_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
 stats={group:dict(samples=len(v),explicit_remains_silent=sum('remains silent' in x['description'].lower() for x in v.values()),
                    exact_remains_silent=sum(x['description'].strip().lower()=='the listener remains silent.' for x in v.values()),
                    median_word_count=float(np.median([len(x['description'].split()) for x in v.values()]))) for group,v in plans.items()}
 (OUT/'planning_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
 print('Prepared10-system analysis,9 paired contrasts,16 disclosed examples')
 for x in records:print(x['arm'],x['generated_active_clips'],x['completely_silent_on_active_gt'],x['false_sound_on_silent_gt'])
 for key in ['Feature6000_vs_AV6000','Combined_vs_Feature6000','SFTplanner_vs_baseEvent','SFTH_vs_sameSFTplanner']:
  print(key,{m:(round(x['difference_right_minus_left'],5),[round(t,5) for t in x['ci95']]) for m,x in paired_results[key]['metrics'].items()})
 for x in selected:print(x['index'],x['title'],x['id'])


if __name__=='__main__':main()
