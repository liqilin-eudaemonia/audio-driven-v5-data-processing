import json,hashlib,subprocess,math
from pathlib import Path
import numpy as np
import soundfile as sf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];old=ROOT/'experiments/p2_train24_fourway_eval_20261027'
read=lambda p:json.loads(p.read_text());jobs=read(OUT/'validation_jobs.json');rows=[]
for job in jobs:
 p=OUT/'validation_outputs'/job['name'];assert (p/'complete.json').exists();m=read(p/'completed.json');assert m['prompt_variant']==job['variant'];assert m['target_annotation_access']==(job['variant']=='guided');assert m['gt_media_access'] is False
 w,sr=sf.read(p/'listener.wav');key=hashlib.sha256(json.dumps([job['sample_id'],job['seed']],sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()[:32];b,br=sf.read(old/job['arm']/'generation/generated'/key/'listener.wav');assert sr==br and w.shape==b.shape
 streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,codec_name,sample_rate,channels,duration','-of','json',str(p/'generated.mp4')],text=True))['streams'];assert {x['codec_type'] for x in streams}=={'video','audio'}
 rms=float(np.sqrt(np.mean(w*w)));base=float(np.sqrt(np.mean(b*b)));row=dict(**job,rms=rms,rms_dbfs=20*math.log10(max(rms,1e-15)),fixed_baseline_rms=base,fixed_baseline_dbfs=20*math.log10(max(base,1e-15)),streams=streams,output=str(p),prompt_sha256=m['prompt_sha256'])
 if job['variant']=='fixed':
  row.update(wave_bit_identical=np.array_equal(w,b),wave_rms_difference=float(np.sqrt(np.mean((w-b)**2))),wave_max_abs_difference=float(np.max(abs(w-b))))
  def pixels(path):return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-']),dtype=np.uint8)
  a=pixels(p/'generated.mp4');bb=pixels(old/job['arm']/'generation/generated'/key/'generated.mp4');assert a.shape==bb.shape
  row.update(video_pixel_mae=float(np.mean(np.abs(a.astype(np.float32)-bb.astype(np.float32)))),video_pixel_max_abs=int(np.max(np.abs(a.astype(np.int16)-bb.astype(np.int16)))))
 rows.append(row)
for i in range(8):
 worker=read(OUT/f'worker-{i}.json');assert worker['finished'] and all(r['returncode']==0 for r in worker['results'])
report=dict(status='passed_engineering_validation',generation_outputs=10,guided_outputs=9,fixed_controls=1,
            weights=['E0','E1','E0F','E1F'],seeds=[137],source_only_benchmark=False,
            source_code_unchanged=True,training_started=False,html_changed=False,
            cpu_tests=dict(guidance=35,existing_glare=37),rows=rows)
(OUT/'validation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# 原文标注指导推理：实现与GPU验证','',
'按确认方案保留原文及时间，合并音频描述/发声文本。24条原train样本和24条test样本都已导出指导文本并完成Gemma缓存，不修改原数据或固定prompt缓存。','',
'原train24的文本长度分档：18条1024、5条2048、1条4096；test24：16条1024、4条2048、4条4096。分档是padding长度；实际最长分别2931和2872 tokens。均检查完整token ID序列，没有截断。','',
'**工程验证完成：四组权重各2条短样本（8条）、E0F一条4096档长文本、一条E0固定prompt重放，共10条输出。不是四组全部24条的正式效果评测。**','',
'| 输出 | 配置 | 新音轨RMS | 对应旧固定prompt RMS | 视频与原文 |','|---|---|---:|---:|---|']
for row in rows:
 name=row['name'];lines.append(f'| {name} | {row["variant"]} | {row["rms"]:.8f} | {row["fixed_baseline_rms"]:.8f} | [视频](validation_outputs/{name}/generated.mp4) · [音频](validation_outputs/{name}/listener.wav) · [prompt](validation_outputs/{name}/prompt.txt) |')
f=next(r for r in rows if r['variant']=='fixed');lines+=['',
f'固定prompt重放不是逐位一致：音频RMS差值（波形差的RMS）为{f["wave_rms_difference"]:.8g}，最大逐采样差{f["wave_max_abs_difference"]:.8g}；视频像素MAE {f["video_pixel_mae"]:.6g}/255，最大差{f["video_pixel_max_abs"]}/255。原配置、权重、正负文本与seed一致；保留实际差异，不声称位级复现。','',
'两条短样本的指导版音频电平明显增加。RMS仅说明幅度，不证明发声内容、音色、时间遵循或与speaker的适当性；后续需试听并独立核验。指导使用目标标注，全部明确标记annotation-guided diagnostic，不能混入旧source-only E0/E1成绩。','',
'- [完整验证记录](validation_summary.json)、[全部prompt及缓存校验](cache_validation.json)、[验证任务](validation_jobs.json)。',
'- [逐样本prompt](prompts/)，`.txt`是实际Gemma输入，`.json`包含完整原标注和每行来源。',
'- [实现与命令](../../related/P2_原文标注Gemma指导推理实现.md)。',
'- 35项新增CPU测试及37项现有评测入口测试通过。没有修改训练实现、权重或已有HTML。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
other=ROOT/'experiments/p2_guided_prompt_test24_20261027';(other/'README.md').write_text('# 独立test24：逐样本原文指导与Gemma缓存\n\n24份原文prompt与24份文本缓存已完成，见[prompts](prompts/)和[缓存校验](cache_validation.json)。本目录尚未执行指导版视频生成。\n\n这是使用目标标注的指导诊断输入，不是source-only正式test条件。\n\n[实现与命令](../../related/P2_原文标注Gemma指导推理实现.md)；[train样本的10条GPU验证输出](../p2_guided_prompt_train24_20261027/README.md)。\n')
print(json.dumps({'outputs':len(rows),'fixed_replay':f},ensure_ascii=False,indent=2))
