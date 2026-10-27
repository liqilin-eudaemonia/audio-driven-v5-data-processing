#!/usr/bin/env python3
"""Build a portable, evidence-linked HTML review of frozen benchmark results."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.evaluate import subtract
from listen_labeling.schema import CLASSES
from p2_listen.contracts import sha
from listen_benchmark_retest_review import retest_comparison, retest_template, retest_analysis
from listen_discovery_review import analyze_discovery, discovery_markdown, discovery_panel

LABELS = {
    'joy_visual':'Joy · 视觉', 'qwen3_visual':'Qwen3-Omni · 视觉',
    'qwen3_audio':'Qwen3-Omni · 音频', 'qwen3_omni_native':'Qwen3-Omni · 原生视听',
    'minicpm_visual':'MiniCPM · 视觉', 'minicpm_audio':'MiniCPM · 音频',
    'minicpm_omni_native':'MiniCPM · 原生视听 8 fps', 'qwen2_audio':'Qwen2-Audio',
    'qwen3_vl_visual':'Qwen3-VL · 视觉', 'captioner_audio':'Qwen3-Omni Captioner',
    'qwen3_asr':'Qwen3-ASR', 'face_landmarker':'Face Landmarker',
    'qwen3_omni_separate':'Qwen3-Omni · 视听普通拼接',
    'minicpm_omni_separate':'MiniCPM · 视听普通拼接',
    'minicpm_omni_native_1fps':'MiniCPM · 原生视听 1 fps',
    'minicpm_visual_1fps':'MiniCPM · 视觉 1 fps',
}
PAIRS = [
    ('pair_joy', 'Joy ＋ Omni 音频', ['joy_visual','qwen3_audio'], '诊断基线 · 待验收'),
    ('pair_vl', 'Qwen3-VL ＋ Omni 音频', ['qwen3_vl_visual','qwen3_audio'], '重点挑战者'),
    ('pair_omni', 'Omni 视觉 ＋ Omni 音频', ['qwen3_visual','qwen3_audio'], '分路对照'),
    ('pair_mini', 'MiniCPM 视觉 ＋ 音频', ['minicpm_visual','minicpm_audio'], '暂不采用'),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def valid_intervals(result, target):
    if result.get('status') != 'schema_pass':
        return []
    value = result['parsed']
    excluded = value['unknown'][target] + [[e['start'], e['end']] for e in value['events']
                                         if e['type'] == target and e['boundary_censored']]
    return subtract(value['reviewed'][target], excluded)


def build_media(task):
    row, out = task
    sid = row['sample_id']; target = out / 'media' / sid; target.mkdir(parents=True, exist_ok=True)
    hashes = {k:sha(row[k]) for k in ('target_audio','target_video')}
    crop = row['video_transform']['crop']; duration = row['duration_s']
    identity = dict(source_sha256=hashes, crop=crop, duration_s=duration,
                    preview='original-frame-rate-crop-512-bicubic-h264-crf20-no-audio', audio='unaltered-independent-wav')
    cache = target / 'provenance.json'
    if not (cache.exists() and json.loads(cache.read_text()) == identity and (target/'video.mp4').is_file() and (target/'audio.wav').is_file()):
        command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-i',row['target_video'],
                   '-map','0:v:0','-an','-vf',f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},scale=512:512:flags=bicubic,setsar=1",
                   '-filter_threads','1','-t',str(duration),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
                   '-pix_fmt','yuv420p','-movflags','+faststart',str(target/'video.mp4')]
        subprocess.run(command, check=True, capture_output=True)
        shutil.copy2(row['target_audio'], target/'audio.wav'); write(cache, identity)
    assert sha(target/'audio.wav') == hashes['target_audio']
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(target/'video.mp4')]))
    assert len(probe['streams']) == 1 and probe['streams'][0]['codec_type'] == 'video'
    assert probe['streams'][0]['width'] == probe['streams'][0]['height'] == 512
    return dict(id=sid, duration=duration, video=f'media/{sid}/video.mp4', audio=f'media/{sid}/audio.wav',
                provenance=f'media/{sid}/provenance.json', source_hashes=hashes,
                cohort='自然对话' if sid.startswith('naturalistic') else '即兴对话')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--output',default=str(ROOT/'data_previews/listen_benchmark_review'))
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    final=json.loads((run/'final_comparison.json').read_text());manifest=json.loads((run/'manifest50.json').read_text())
    assert final['configuration_count']==16 and final['samples']==50 and final['completed_requests']==1450
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        samples=list(pool.map(build_media, [(r,out) for r in manifest]))
    print('Prepared and verified 50 media previews',flush=True)
    duration=sum(s['duration'] for s in samples);systems={};results={};source_hashes={}
    for row in final['rows']:
        name=row['system'];path=Path(row['summary']);path=path if path.is_absolute() else ROOT/path
        summary=json.loads(path.read_text());identity=json.loads(path.with_name('inputs.json').read_text())
        for sample in identity['samples']:
            actual=next(s['source_hashes'] for s in samples if s['id']==sample['sample_id'])
            assert sample['media_hashes']==actual, 'Media fingerprint mismatch'
        cfg=identity['system']; source_hashes[name]=sha(path)
        stats=dict(row);stats.pop('summary');stats.pop('inference_wall_seconds_sum')
        stats.update(name=LABELS[name], branch=summary['branch'], fps=cfg.get('media_override',{}).get('fps',identity['media']['fps']),
                     representation=cfg['representation'], max_slice_nums=cfg.get('max_slice_nums'), engine=cfg.get('runtime_engine','Transformers'),
                     versions=identity.get('versions',{}), audit_pass=0, discovery_pass=0,event_counts={},positive_clips={},censored_events=0)
        events=Counter();positive=Counter();by_sample={}
        for record in summary['results']:
            sid=record['sample_id'];stage=record['stage'];raw=record.get('raw_response',{});rel=f'raw/{name}/{sid}_{stage}.json'
            destination=out/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(record['path'], destination)
            request_path=Path(record['path']).with_name('request.json')
            request=json.loads(request_path.read_text()) if request_path.exists() else {}
            item=dict(status=record['status'],error=record.get('error'),parsed=record.get('parsed'),raw=raw.get('text',''),
                      finish=raw.get('finish_reason'),tokens=raw.get('completion_tokens'),seconds=record.get('inference_wall_seconds'),
                      prompt=request.get('prompt'),raw_url=rel,coverage={},
                      frame_count=len(record.get('input_evidence',{}).get('video',{}).get('frames',[])))
            if stage=='audit' and record['status']=='schema_pass':
                stats['audit_pass']+=1
                for target in CLASSES[summary['branch']]:
                    item['coverage'][target]=valid_intervals(record,target)
                for event in record['parsed']['events']:
                    events[event['type']]+=1;stats['censored_events']+=int(event['boundary_censored'])
                for target in {e['type'] for e in record['parsed']['events']}:positive[target]+=1
            if stage=='discovery' and record['status']=='schema_pass':stats['discovery_pass']+=1
            if name=='face_landmarker':
                frames=raw.get('frames',[])
                item['face']=dict(total=len(frames),detected=sum(f['face_detected'] for f in frames),
                                  trajectory=[[f['time_s'],f['landmarks'][0][1][1]] for f in frames if f['face_detected'] and f.get('landmarks')])
            by_sample.setdefault(sid,{})[stage]=item
        stats['event_counts']=dict(events);stats['positive_clips']=dict(positive);results[name]=by_sample;systems[name]=stats
        print(name,'exported',flush=True)
    schemes=[]
    for ident,label,members,badge in PAIRS:
        schemes.append(dict(id=ident,name=label,members=members,group='scheme',badge=badge))
    for name,model in systems.items():
        group='auxiliary' if model['auxiliary'] else ('control' if model['tier']=='control' else model['branch'])
        native_badge=('时序已修正 · 已重测' if model['phase']=='native_aligned_v7' else '时序配置问题 · 需重测')
        schemes.append(dict(id=name,name=LABELS[name],members=[name],group=group,
                            badge=native_badge if name=='qwen3_omni_native' else ''))
    for scheme in schemes:
        members=scheme['members']; passed=0;audit_pass=0;both=0;seconds=Counter();counts=Counter();positive=Counter()
        auxiliary=all(systems[m]['auxiliary'] for m in members)
        for sample in samples:
            stages=[v for m in members for v in results[m][sample['id']].values()]
            audits=[results[m][sample['id']].get('audit') for m in members]
            passed+=int(all(r['status'] in ('schema_pass','auxiliary_only') for r in stages))
            audit_pass+=int(not auxiliary and all(a and a['status']=='schema_pass' for a in audits))
            coverage={};types=set()
            for m,a in zip(members,audits):
                if not a:continue
                types.update(CLASSES[systems[m]['branch']])
                for target,intervals in a['coverage'].items():
                    coverage[target]=sum(b-a for a,b in intervals);seconds[target]+=coverage[target]
                if a['status']=='schema_pass':
                    counts.update(e['type'] for e in a['parsed']['events']);positive.update(set(e['type'] for e in a['parsed']['events']))
            both+=int(bool(types) and all(coverage.get(t,0)>0 for t in types))
        scheme.update(passed=passed,audit_pass=audit_pass,both=both,coverage={k:v/duration for k,v in seconds.items()},
                      events=dict(counts),positive_clips=dict(positive),auxiliary=auxiliary,failed_samples=50-passed)
        if len(members)==1 and not auxiliary:
            assert passed==systems[members[0]]['complete_schema_pass']
            for key,value in scheme['coverage'].items():assert abs(value-systems[members[0]]['audit_supported_seconds_fraction'][key])<1e-8
    # Differences are inspection priorities, never labels or precision/recall estimates.
    for sample in samples:
        sid=sample['id'];vectors=[];has_failure=False;poor=False
        for name in ('joy_visual','qwen3_vl_visual','qwen3_visual','qwen3_omni_native'):
            stages=results[name][sid];audit=stages.get('audit');has_failure|=any(v['status']=='failed' for v in stages.values())
            vectors.append(None if not audit or audit['status']!='schema_pass' else sum(e['type']=='head_nod' for e in audit['parsed']['events']))
            poor|=not audit or sum(b-a for a,b in audit['coverage'].get('head_nod',[]))<sample['duration']*.5
        for stage in results['qwen3_audio'][sid].values():has_failure|=stage['status']=='failed'
        sample.update(nod_counts=vectors,disagreement=len(set(v for v in vectors if v is not None))>1,
                      failure=has_failure,low_coverage=poor,disagreement_score=max([v for v in vectors if v is not None] or [0])-min([v for v in vectors if v is not None] or [0]))
    resources=[]
    for line in (run/'cluster_resources.jsonl').read_text().splitlines():
        record=json.loads(line);point=dict(utc=record['utc'],nodes={})
        for node,info in record['nodes'].items():
            if not info.get('gpus'):continue
            point['nodes'][node]=dict(cpu=info.get('cpu_percent'),gpu=sum(g['gpu_percent'] for g in info['gpus'])/len(info['gpus']),
                                     memory=sum(g['memory_used_mib'] for g in info['gpus'])/1024)
        if point['nodes']:resources.append(point)
    data=dict(revision='listen-benchmark-review-v1',run_id=run.name,built_utc=datetime.now(timezone.utc).isoformat(),
              experiment_utc=final['utc'],duration=duration,samples=samples,systems=systems,results=results,schemes=schemes,
              resources=resources,resource_summary=json.loads((run/'resource_summary.json').read_text()),
              provenance=dict(final_comparison_sha256=sha(run/'final_comparison.json'),summaries_sha256=source_hashes),
              human_ground_truth=False,training_admitted=False)
    template=(ROOT/'scripts/listen_benchmark_review.html').read_text()
    if (run/'retest_plan.json').exists():
        data['retest']=retest_comparison(run,data)
        write(run/'retest_comparison.json',data['retest']);write(out/'retest_comparison.json',data['retest'])
        template=retest_template(template,data)
    data['discovery_analysis']=analyze_discovery(data)
    write(out/'discovery_analysis.json',data['discovery_analysis'])
    write(run/'discovery_analysis.json',data['discovery_analysis'])
    discovery_report=discovery_markdown(data['discovery_analysis'])
    (out/'开放描述分析.md').write_text(discovery_report)
    (run/'开放描述分析.md').write_text(discovery_report)
    template=template.replace('__DISCOVERY_ANALYSIS__',discovery_panel(data['discovery_analysis']))
    write(out/'review.json',data)
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    (out/'index.html').write_text(template.replace('__REVIEW_DATA__',encoded))
    # Bundle a small Chinese font subset for browser hosts without CJK fonts.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font_path=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.exists():
        font=TTFont(font_path,fontNumber=2);sub=subset.Subsetter();sub.populate(text=template+encoded);sub.subset(font)
        font.flavor='woff';font.save(out/'review.woff')
    for name in ('最终对照结果.md','final_comparison.json','paired_scheme_summary.json','resource_summary.json'):
        shutil.copy2(run/name,out/name)
    if (run/'native_alignment_diagnosis.json').exists():
        shutil.copy2(run/'native_alignment_diagnosis.json',out/'native_alignment_diagnosis.json')
    analysis=analysis_text(data)
    (out/'选型分析.md').write_text(analysis);(run/'选型分析.md').write_text(analysis)
    write(out/'build_validation.json',dict(samples=len(samples),systems=len(systems),requests=sum(len(stages) for v in results.values() for stages in v.values()),
          media_hashes_verified=True,stats_reconciled=True,html_bytes=(out/'index.html').stat().st_size,
          human_ground_truth=False,source=run.name,utc=data['built_utc']))
    print('READY',out,flush=True)


def analysis_text(data):
    if data.get('retest'):
        return retest_analysis(data)
    return '''# 打标方案选型分析：当前建议与证据边界

**当前没有已通过质量验收的正式打标方案。撤回此前对 Joy＋Omni 的“主候选”推荐，仅保留为诊断基线。** 用户复核反馈大量点头漏标；原始回答也存在备注描述持续点头、事件列表却为空的矛盾。结构通过率和模型自报覆盖率不足以支持生产选型。

## 本次复查发现

**原生视听时序问题已复现，旧结果需重测。** vLLM 0.19 的缓存预处理路径在拼接音视频 token 时读取显式 `second_per_grid_ts`，缺省使用 2.0 秒；本次 8 fps、每组 2 帧应为 0.25 秒。旧适配器只传 fps，没有传此字段。CPU 预处理对照中，旧配置将 2 秒音频集中排在第一组视频之后，显式传入 0.25 后恢复与 HF 处理器一致的交错顺序。当前适配器已修正；冻结实验快照和旧输出保留。此对照未运行模型生成，尚不能量化该问题造成的漏检，也不能声称修复后质量合格。

- 样本 naturalistic_V03_S0882_I00000471_local_664e95f3508c：同一 WAV，纯音频报告 1.21–1.85 秒笑声；页面采用的 vLLM 原生视听没有报告事件；保留的 Transformers 原生视听却报告 2.41–3.55 秒笑声。三个时间结论尚未经人工真值验证。不能把这解释为联合输入必然损害听觉，需控制推理实现、预处理和解码约束复验。
- 50 条最终结果中，纯音频在 9 条里报告 12 个发声事件，原生视听在 4 条里报告 4 个；7 条纯音频阳性在原生视听中没有发声事件。此处是模型输出分歧，不是已测得的漏检率。
- Joy 的 naturalistic_V03_S1460_I00000182_local_bfdaf1092a0a：notes 明确描述开头至约 2.4 秒持续有节律地点头，events 却为空。结构校验没有检查这类语义矛盾。
- 页面灰色区域原名“未检查”，现改为“未声明有效覆盖”。它表示 reviewed 扣除 unknown 和边界截断后的覆盖为空，不表示模型没有收到视频，也不能当作可靠负例。有的 Joy 输出只填写了事件窗口，有的全片未填覆盖。

本次包含既有记录复查和 CPU 预处理验证，没有重新进行模型生成或修补原始事件。

## 为什么不能直接换成结构更稳的 Qwen3-VL

50 条视频共 330.2365 秒。Joy 在 24 条中报告 41 个点头；Qwen3-VL 在 1 条中报告 2 个；Qwen3-Omni 视觉在 6 条中报告 6 个；Qwen 原生视听报告 1 个。Qwen3-VL 虽然自报有效检查时长为 98.3%，仍可能存在系统性漏检；Joy 则可能把一般头动当点头。没有逐事件人工真值，不能以更多事件证明召回，也不能以更少事件证明精度。

## 音频目前更倾向 Qwen3-Omni

Qwen3-Omni 音频两阶段结构通过 50/50，有效检查时长 73.3%，在 9 条中报告 12 个发声事件；但仍有 13 条 audit 没有任何有效检查时长，不能将其空事件当可靠负例。MiniCPM 音频虽然两阶段通过 49/50，实际有效检查时长仅 4.2%；57 个报告事件中有 12 个边界截断。Qwen2-Audio 有效检查时长 17.8%，通过校验的 audit 结果没有报告发声事件，另有 11 次时间范围错误。这些证据支持优先使用 Omni 做音频草稿，不构成其声学准确率领先的证明。

## 原生视听旧配置存在时序问题，需重测

Qwen 原生视听两阶段通过 49/50，点头／发声有效检查时长分别为 60.1%／62.6%，只报告 1 个点头和 4 个发声；Joy＋Omni 对应时长覆盖为 89.8%／73.3%。此版存在上述 token 时序错配，不用于公平评价联合模型能力。MiniCPM 原生视听 8 fps 两阶段通过 29/50，点头有效覆盖为 0；1 fps 对照也只有约 0.3%。在当前 prompt、解码和输入配置下，不建议直接用于批量标签生产；不能把该结论推广成模型本身不会感知。

## 建议落地方式

1. 所有配置均待质量验收，Joy＋Omni 仅作诊断基线。discovery 与 audit 独立，不互相喂答案；检查两阶段矛盾，但不把它们直接合并当真值。
2. 对视觉分歧片段做盲评：区分明确点头、一般头动、边界不清及无法判断；既查 Joy 的误报，也查 Qwen3-VL／联合模型的漏报。不要把多模型共识自动当真值。
3. 对 Omni 音频零有效覆盖和模型发声分歧片段重点复核。Captioner 与 ASR 只提供声音描述／文字线索，不能直接替代事件时间边界；人脸轨迹只提供几何证据。
4. 用独立人物/session 的新留出集确认逐类事件 precision、recall、F1、边界误差和人工修订耗时后，再定最终生产方案。此次 50 条是诊断样本。

## 比较限制

覆盖来自模型自报 reviewed，扣除 unknown 和边界截断区域；不是人工验证的正确率。事件数包含通过结构校验的边界截断事件。不同引擎、原生时序编码与并发部署会影响结果；未做等资源速度对照，不把本轮总耗时排名成模型速度。页面的人工偏好记录也不是事件 GT。
'''


if __name__=='__main__':main()
