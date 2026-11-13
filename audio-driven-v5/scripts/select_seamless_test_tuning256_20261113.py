"""Freeze a distribution-matched test tuning subset; never inspect model scores."""
import collections,csv,hashlib,json,math,random,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/seamless_test_tuning256_20261113'
SEED=20261113
N=256

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def stratum(r):
 return (r['pair'].split('_')[0],bool(r['provenance']['processed_listener_waveform_nonzero']),0 if r['duration_s']<4 else 1 if r['duration_s']<8 else 2)
def label(k):return f'{k[0]}|'+('nonzero' if k[1] else 'zero')+'|'+['2-4','4-8','8-11.95'][k[2]]
def summary(rs):
 return dict(n=len(rs),mean_duration_s=sum(r['duration_s'] for r in rs)/len(rs),total_duration_s=sum(r['duration_s'] for r in rs),source_pairs=len({r['pair'] for r in rs}),source_groups=len({r['group_id'] for r in rs}),max_clips_per_group=max(collections.Counter(r['group_id'] for r in rs).values()),genre=dict(collections.Counter(stratum(r)[0] for r in rs)),waveform=dict(collections.Counter('nonzero' if stratum(r)[1] else 'zero' for r in rs)),duration=dict(collections.Counter(['2-4','4-8','8-11.95'][stratum(r)[2]] for r in rs)),strata=dict(sorted(collections.Counter(label(stratum(r)) for r in rs).items())),mean_face_p10_px=sum(r['face']['short_side_p10_px'] for r in rs)/len(rs))
def choose(rs):
 counts=collections.Counter(map(stratum,rs));quotas={k:int(v*N/len(rs)) for k,v in counts.items()}
 order=sorted(counts,key=lambda k:(-(counts[k]*N/len(rs)%1),k))
 for k in order[:N-sum(quotas.values())]:quotas[k]+=1
 old=rows(ROOT/'experiments/seamless_test_tuning64_20261113/selection.jsonl')
 by_id={r['sample_id']:r for r in rs};selected=[by_id[r['id']] for r in old]
 assert len(selected)==64
 existing=collections.Counter(map(stratum,selected));assert all(existing[k]<=quotas[k] for k in existing)
 used_groups=collections.Counter(r['group_id'] for r in selected);used_pairs={r['pair'] for r in selected}
 buckets={k:[] for k in counts}
 for r in sorted(rs,key=lambda r:r['sample_id']):buckets[stratum(r)].append(r)
 rng=random.Random(SEED)
 for k in sorted(buckets):rng.shuffle(buckets[k])
 priorities={r['sample_id']:i for k in sorted(buckets) for i,r in enumerate(buckets[k])}
 schedule=[k for k in sorted(quotas) for _ in range(quotas[k]-existing[k])];rng.shuffle(schedule)
 for k in schedule:
  available=[r for r in buckets[k] if r['pair'] not in used_pairs]
  if not available:raise ValueError('Cannot satisfy quotas with unique original source pairs')
  r=min(available,key=lambda r:(used_groups[r['group_id']],priorities[r['sample_id']]))
  selected.append(r);used_pairs.add(r['pair']);used_groups[r['group_id']]+=1
 return selected,quotas

def main():
 current=read(ROOT/'evaluation/SEAMLESS_CURRENT.json');parent=Path(current['prepared_full6624']['prepared']);seal=read(parent/'seal.json')
 assert sha(parent/'references.jsonl')==seal['references_sha256'] and sha(parent/'source.test.jsonl')==seal['source_sha256']
 refs=rows(parent/'references.jsonl');sources=rows(parent/'source.test.jsonl')
 assert len(sources)==len(refs)==6624 and {r['id'] for r in refs}=={r['sample_id'] for r in sources}
 assert all(r['split']=='test' for r in sources)
 selected,quotas=choose(sources);assert [r['sample_id'] for r in choose(sources)[0]]==[r['sample_id'] for r in selected]
 assert len(selected)==len({r['pair'] for r in selected})==256
 counts=collections.Counter(map(stratum,selected));assert counts==quotas
 assert all(abs(counts[k]-sum(stratum(r)==k for r in sources)*256/len(sources))<1 for k in quotas)
 WORK.mkdir(parents=True,exist_ok=True)
 if (WORK/'COMPLETE.json').exists():
  old=rows(WORK/'selection.jsonl');assert [r['id'] for r in old]==[r['sample_id'] for r in selected]
  print('Existing frozen selection matches deterministic recomputation');return
 refmap={r['id']:r for r in refs};annmap={r['id']:r for r in rows(parent/'annotations.jsonl')};inputmap={r['id']:r for r in rows(parent/'inference_inputs.jsonl')};genmap={r['id']:r for r in rows(parent/'generation_map.template.jsonl')}
 prepared=WORK/'prepared_256';prepared.mkdir(exist_ok=True)
 shutil.copyfile(parent/'crop_config.json',prepared/'crop_config.json')
 output_refs=[];output_inputs=[];output_anns=[];output_maps=[];selection=[];checks=[]
 offset=0
 with (prepared/'source.test.jsonl').open('wb') as out,(parent/'source.test.jsonl').open('rb') as source:
  for i,r in enumerate(selected,1):
   sid=r['sample_id'];old=refmap[sid];ref=old['gt_reference'];source.seek(ref['offset']);raw=source.read(ref['bytes'])
   assert hashlib.sha256(raw).hexdigest()==ref['row_sha256'] and json.loads(raw)==r
   out.write(raw)
   newref=dict(ref,manifest=str(prepared/'source.test.jsonl'),offset=offset,crop_config=str(prepared/'crop_config.json'));offset+=len(raw)
   output_refs.append(dict(old,gt_reference=newref))
   output_inputs.append(dict(inputmap[sid],gt_reference=newref));output_anns.append(annmap[sid]);output_maps.append(genmap[sid])
   frame=r['inputs']['listener_first_frame'];audio=r['inputs']['speaker_audio']
   assert Path(frame['source']['archive']).is_file()
   assert Path(audio['source_result_file']).is_file()
   assert sha(r['target']['source_file'])==r['target']['source_sha256']
   checks.append(dict(id=sid,parent_row_sha256=ref['row_sha256'],source_archive_exists=True,source_result_exists=True,caption_hash_verified=True))
   selection.append(dict(rank=i,id=sid,split='test',purpose='checkpoint_tuning',pair=r['pair'],group_id=r['group_id'],genre=stratum(r)[0],listener_waveform_nonzero=stratum(r)[1],duration_bin=['2-4','4-8','8-11.95'][stratum(r)[2]],duration_s=r['duration_s'],source_start_s=old['source_start'],source_end_s=old['source_end'],speaker_id=old['speaker_id'],listener_id=old['listener_id'],face_p10_px=old['face_p10_px'],stratum=label(stratum(r)),reference_row_sha256=ref['row_sha256']))
 for name,records in [('references.jsonl',output_refs),('inference_inputs.jsonl',output_inputs),('annotations.jsonl',output_anns),('generation_map.template.jsonl',output_maps)]:
  (prepared/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 (WORK/'selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selection))
 with (WORK/'selection.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
 (WORK/'ids.txt').write_text(''.join(r['id']+'\n' for r in selection))
 # Preserve exact selected GT rows, geometry and media; only rebase manifest offsets.
 save(prepared/'seal.json',dict(schema='seamless-approved-subset-v1',dataset='seamless',expected_clips=256,parent_prepared=str(parent),parent_seal_sha256=sha(parent/'seal.json'),parent_references_sha256=sha(parent/'references.jsonl'),selected_ids_sha256=hashlib.sha256(json.dumps(sorted(r['id'] for r in output_refs),separators=(',',':')).encode()).hexdigest(),references_sha256=sha(prepared/'references.jsonl'),source_sha256=sha(prepared/'source.test.jsonl'),crop_config_sha256=sha(prepared/'crop_config.json'),selection_sha256=sha(WORK/'selection.jsonl'),source_version=current['version'],purpose='test_checkpoint_tuning_not_independent_dev'))
 full,sub=summary(sources),summary(selected)
 old200={r['id'] for r in rows(Path(current['prepared_200']['prepared'])/'references.jsonl')}
 report=dict(full_test=full,selected256=sub,stratum_quotas={label(k):v for k,v in sorted(quotas.items())},seed=SEED,old_diagnostic200_overlap_ids=[r['id'] for r in selection if r['id'] in old200],audio_definition='Stored processed_listener_waveform_nonzero; nonzero is a waveform property, not a human speech/laughter judgment',selection_policy='Hamilton proportional quotas for genre x waveform-zero/nonzero x duration; seeded random priority within cells; prefer unused source groups; retain original64; prefer least represented group; one clip per original pair; no model outcomes read',test_used_for_selection='Aggregate input metadata only; no generation, metric, or model ranking accessed',limitations=['This is explicitly a test tuning subset, not independent dev. Full6624 scores include these tuning examples.','256 clips only support coarse screening, not confident ranking of nearly tied checkpoints.','No new human reaction labels or full-media decode performed; existing frozen metadata and media references retained.'])
 save(WORK/'DISTRIBUTION.json',report)
 save(WORK/'PROTOCOL.json',dict(seed=SEED,count=256,parent=str(parent),source_version=current['version'],selection_script=str(Path(__file__).resolve()),selection_script_sha256=sha(__file__),parent_reference_sha256=sha(parent/'references.jsonl'),parent_source_sha256=sha(parent/'source.test.jsonl'),selected_ids_sha256=sha(WORK/'ids.txt'),selection_sha256=sha(WORK/'selection.jsonl'),quotas=report['stratum_quotas'],policy=report['selection_policy'],purpose='test_checkpoint_tuning',inference='All candidate weights must use identical selected IDs, fixed Qwen conditions, seed and sampling settings; target annotation never enters generation inputs',future_expansion='Contains the original64 IDs unchanged; 192 new IDs selected without model outputs',does_not_launch_inference=True))
 save(WORK/'SELECTION_AUDIT.json',dict(success=True,count=256,retained64=64,added192=192,all_from_current_full_test=True,deterministic_reselection_equal=True,each_stratum_rounding_error_lt_one=True,unique_original_pairs=256,max_per_group=sub['max_clips_per_group'],source_groups=sub['source_groups'],checks=checks,full_media_decode_performed=False))
 lines=['# Seamless test 固定256条快速筛选集','',f'已固定256条，来自当前完整test的6624条。随机种子{SEED}；直接用于选择Qwen/LTX权重，**不是独立dev**。完整test最终分数包含这256条，需披露调参重叠。','','抽样按naturalistic/improvised × listener处理后音轨全零/非零 × 时长（<4、4–8、≥8秒）共12层，以最大余数法分配256个名额。层内使用固定随机优先级，优先不同来源组，保留原64条、优先补充当前样本较少的来源组，每条原长视频只取1个pair；不读取任何模型预测或评分。','','| 项目 | 完整test | 固定256 |','|---|---:|---:|']
 for field,title in [('genre','来源'),('waveform','音轨'),('duration','时长')]:
  for k in sorted(full[field]):lines.append(f'| {title}：{k} | {full[field][k]}（{full[field][k]/6624:.2%}） | {sub[field].get(k,0)}（{sub[field].get(k,0)/256:.2%}） |')
 lines.extend([f'| 平均时长 | {full["mean_duration_s"]:.3f}s | {sub["mean_duration_s"]:.3f}s |',f'| 平均原始人脸短边p10 | {full["mean_face_p10_px"]:.1f}px | {sub["mean_face_p10_px"]:.1f}px |','',f'256条来自{sub["source_pairs"]}条原长视频、{sub["source_groups"]}个来源组，总计{sub["total_duration_s"]/60:.2f}分钟。与旧固定200重叠{len(report["old_diagnostic200_overlap_ids"])}条；未按历史效果挑选。','','- `prepared_256/`：与现有Seamless评测器兼容的references、inference_inputs、annotations、seal；原GT行内容、时间、裁框及版本保持。','- `selection.csv`：256条ID、原视频、起止时间及分层信息，供人工检查。','- `selection.jsonl` / `ids.txt`：固定机器清单。','- `DISTRIBUTION.json`：完整分布和12层配额。','- `PROTOCOL.json` / `SELECTION_AUDIT.json`：规则、指纹与核验。','','音轨非零只代表生产处理后波形非零，不等同于人工确认有语音；本次没有重新识别具体反应、重解码整段媒体或启动GPU推理。本轮包含原64条，以便直接复用生成。所有权重固定同一Qwen、seed和采样配置。','','重算核对：`python scripts/select_seamless_test_tuning256_20261113.py`；已有完成清单只校验确定性，不覆盖。'])
 (WORK/'README.md').write_text('\n'.join(lines)+'\n')
 save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),selected=256,selection_sha256=sha(WORK/'selection.jsonl'),prepared=str(prepared),source_version=current['version'],purpose='test_checkpoint_tuning'))
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
