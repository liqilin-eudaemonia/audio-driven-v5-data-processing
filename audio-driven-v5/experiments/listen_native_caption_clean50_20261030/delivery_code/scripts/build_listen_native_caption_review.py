#!/usr/bin/env python3
"""Publish the natural-language pilot without replacing historical event results."""
import argparse
import json
import re
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_audio_postprocess.recipes import read_listener_audio, masks
from p2_listen.contracts import atomic_json, sha
from build_listen_visual_prompt_versions_review import compact_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    base=ROOT/'data_previews/listen_benchmark_review/native_simple';out=base/'natural_caption';out.mkdir(exist_ok=True)
    old=json.loads((base/'comparison.json').read_text());historical={s['id']:s for s in old['samples']}
    rows=json.loads((run/'manifest.json').read_text());samples=[]
    for row in rows:
        sid=row['sample_id'];recipe=json.loads(Path(row['listener_audio_recipe']).read_text())
        wave,sr,meta=read_listener_audio(recipe);gate,extra,info=masks(recipe)
        media=out/'media'/sid;media.mkdir(parents=True,exist_ok=True)
        paths={'listener.wav':Path(recipe['source']['path']), 'video.mp4':(base/historical[sid]['video']).resolve(),
               'speaker.wav':Path(row['source_reference']['roles']['speaker']['audio_wav']['path'])}
        for name,path in paths.items():
            dest=media/name
            if dest.is_symlink(): assert dest.resolve()==path.resolve()
            elif dest.exists(): raise ValueError(f'Unexpected physical media: {dest}')
            else: dest.symlink_to(path)
        result_path=run/'results'/(sid+'.json');result=json.loads(result_path.read_text()) if result_path.exists() else None
        if result:
            assert result['recipe_sha256']==row['recipe_sha256']
            assert result['input_evidence']['audio_identity']['processed_waveform_sha256']==meta['processed_waveform_sha256']
            result['review_flags']=list(result['review_flags'])
            if re.search(r'\b(contemplative|pensive|thoughtful|attentive|engaged|curious|curiosity|concentrat\w*|interested|interest)\b',result['caption'],re.I):
                result['review_flags'].append('mental_state_wording_review')
        focus={
            'naturalistic_V00_S0805_I00000126_local_f7c1fb6d13ca':'此前你确认过两次 Right 和五次 yeah。本轮文字也保留了这些词；请检查发声同时的点头是否仍被遗漏或只写成转头。',
            'naturalistic_V02_S5193_I00000190_local_60bde84ffa78':'此前你确认过 Uh huh 和 Yeah。本轮只明确写出 Yeah；请听净化音轨核对 Uh huh 是否遗漏，并检查 contemplative、pensive、thoughtful 等推演措辞。',
            'naturalistic_V00_S0699_I00000500_local_3491a3549642':'本轮写出两次 Okay，请核对是否确实两次、先后关系是否正确，以及笑和点头是否完整。',
            'naturalistic_V03_S2043_I00000338_local_32d3f86103cd':'注意区分可见笑容与实际笑声：请检查 while still laughing 是否把短暂笑声延长到了后面的遮嘴动作。',
        }
        sample=dict(id=sid,short=sid.split('_')[2],duration=row['duration_s'],video=f'media/{sid}/video.mp4',
                    audio=dict(original=f'media/{sid}/listener.wav',speaker=f'media/{sid}/speaker.wav',sample_rate=sr,
                               source_samples=len(wave),gate=compact_mask(gate),extra=compact_mask(extra),
                               processed_full_silence=row['audio_preflight']['full_silence'],
                               processed_pcm_sha256=meta['processed_waveform_sha256']),
                    recipe_provenance=row['audio_preflight']['provenance']['mode'],result=result,
                    focus=focus.get(sid,''),
                    historical_events=historical[sid]['results']['native_baseline'].get('raw_events',[]),
                    historical_url='../?sample='+sid)
        samples.append(sample)
    finished=[s for s in samples if s['result']]
    stats=dict(samples=len(samples),completed=len(finished),nonempty=sum(bool(s['result']['caption'].strip()) for s in finished),
               finished_normally=sum(s['result']['finish_reason']=='stop' for s in finished),
               full_silence=sum(s['audio']['processed_full_silence'] for s in samples),
               possible_interpretation=sum('possible_interpretation_review' in s['result']['review_flags'] for s in finished),
               total_inference_seconds=sum(s['result']['seconds'] for s in finished),human_ground_truth=False)
    payload=dict(run_id=run.name,stats=stats,samples=samples,protocol=json.loads((run/'protocol.json').read_text()))
    atomic_json(out/'review.json',payload);atomic_json(run/'review_summary.json',stats)
    shutil.copy2(ROOT/'scripts/listen_native_caption_review.html',out/'index.html')
    js=(ROOT/'scripts/listen_visual_prompt_review_audio.js').read_text()
    # Original historical clips are PCM24; completed full-corpus sources are PCM16.
    # Decode integer PCM exactly, then apply the same two float32 multiplies.
    js=js.replace("bits !== 16", "![16,24].includes(bits)").replace("bytes / 2 !== s.audio.source_samples", "bytes / (bits/8) !== s.audio.source_samples")
    js=js.replace("const n = bytes / 2,", "const n = bytes / (bits/8),")
    js=js.replace("(dv.getInt16(start + 2 * i, true) / 32768)", "readSample(i)")
    js=js.replace("    for (let i = 0; i < n; i++) out[i]", "    const readSample = i => { if(bits===16) return dv.getInt16(start+2*i,true)/32768; const p=start+3*i; let v=dv.getUint8(p)|(dv.getUint8(p+1)<<8)|(dv.getUint8(p+2)<<16); if(v&0x800000)v-=0x1000000; return v/8388608; };\n    for (let i = 0; i < n; i++) out[i]")
    (out/'audio.js').write_text(js)
    html=(out/'index.html').read_text()
    fontpath=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if fontpath.exists():
        from fontTools.ttLib import TTFont
        from fontTools import subset
        f=TTFont(fontpath,fontNumber=2);options=subset.Options();options.desubroutinize=True;options.layout_features=[]
        ss=subset.Subsetter(options);ss.populate(text=html+js+json.dumps(payload,ensure_ascii=False));ss.subset(f);f.flavor='woff';f.save(out/'review.woff')
    index=base/'index.html';text=index.read_text();marker='<!-- natural-caption-clean50-entry -->'
    if marker not in text:
        shutil.copy2(index,base/'index.before_natural_caption_20261030.html')
        entry=marker+'<section class="panel"><strong>新增：净化 listener 音轨＋原生视听 · 整段自然语言描述</strong><p>同一批50条，描述动作顺序、持续状态、同时发生与真实重复，不输出逐事件时间戳表。</p><a href="natural_caption/">打开有声复核页面 →</a></section>'
        assert '<main>' in text;index.write_text(text.replace('<main>','<main>'+entry,1))
    print(json.dumps(stats,ensure_ascii=False))


if __name__=='__main__': main()
