#!/usr/bin/env python3
"""Playable blind review packet; previews NEVER enter the detector or scorer."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_event_presence.contracts import CLASSES,NAMES,digest,read,rows,save,sha
from p2_event_presence.reporting import review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();work=args.work.resolve();out=args.output.resolve();review(work)
    entries=rows(work/'blind_review_media.jsonl');identity=digest(entries)
    if (out/'review_packet_contract.json').exists() and read(out/'review_packet_contract.json')['media_index_identity']!=identity:
        raise ValueError('Existing review packet belongs to another input')
    out.mkdir(parents=True,exist_ok=True);(out/'preview').mkdir(exist_ok=True);(out/'audio').mkdir(exist_ok=True)
    shutil.copyfile(work/'human_labels.template.csv',out/'human_labels.template.csv')
    pages={}
    style='body{font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 16px;color:#222}article{border:1px solid #ccc;padding:16px;margin:18px 0;border-radius:8px}video{max-width:100%;width:512px}audio{width:min(100%,500px)}code{overflow-wrap:anywhere}li{margin:8px 0}'
    for mod in CLASSES:
        content=[f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>匿名事件核验 · {mod}</title><style>{style}</style>',
            f'<h1>匿名事件核验 · {"视觉" if mod=="visual" else "音频"}</h1>',
            '<p>这里隐藏方法、GT／生成角色和检测器答案。只标记可观察事件；重复同类仍只记存在。无法判断填unknown；留空是未标注。</p>',
            '<p><a href="human_labels.template.csv">下载七类空白标注CSV</a>，填写state（present／absent／unknown）、evidence和rater_id，保留完整media_id。</p>',
            '<p>视频为仅供人工观看的无声H.264预览，检测仍用原无损媒体；音频为原准备波形，不按GT窗口裁生成声音。</p>',
            '<ul>'+''.join('<li>'+html.escape(c)+'：'+html.escape(NAMES[c])+'</li>' for c in CLASSES[mod])+'</ul>']
        if mod=='audio':content+=['<p>yeah／okay／right／no／wow等独立短词属于词语回应；mhm／uh-huh／hmm／uh／ah／oh属于非词汇回应。笑声单列；咳嗽、普通呼吸和噪声不自动归回应。持续长句另标，短词边界不清填unknown。</p>']
        for entry in entries:
            mid=entry['media_id'];content+=[f'<article><p>media_id：<code>{mid}</code> · {entry["duration_s"]:.3f}秒</p>']
            if mod=='visual' and entry.get('video'):
                target=out/'preview'/(mid+'.mp4')
                if not target.exists():subprocess.run(['ffmpeg','-v','error','-threads','1','-i',entry['video'],'-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)],check=True)
                content+=[f'<video controls preload="metadata" src="preview/{mid}.mp4"></video>']
            elif mod=='audio' and entry.get('audio'):
                target=out/'audio'/(mid+'.wav')
                if not target.exists():target.symlink_to(entry['audio'])
                content+=[f'<audio controls preload="metadata" src="audio/{mid}.wav"></audio>']
            else:content+=['<p>此模态素材不可用，请填unknown并说明。</p>']
            content+=['</article>']
        content+=['</html>'];name=f'review_{mod}.html';(out/name).write_text('\n'.join(content));pages[mod]=name
    (out/'index.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>整段事件匿名核验</title><style>{style}</style><h1>整段事件匿名核验</h1><p>人工标注只核验事件类别；语境适当性及视听协调性需另行评价。请先独立标注，再查看检测结果，避免共享错误。</p><p><a href="review_visual.html">视觉四类核验</a> · <a href="review_audio.html">音频三类核验</a> · <a href="human_labels.template.csv">空白标签表</a></p></html>')
    save(out/'review_packet_contract.json',dict(media_index_identity=identity,media=len(entries),pages=pages,
        builder_sha256=sha(Path(__file__)),previews_used_for_detection=False,method_and_side_blinded=True,
        original_media_index_sha256=sha(work/'blind_review_media.jsonl'),human_validation_complete=False))
    print(out)


if __name__=='__main__':main()
