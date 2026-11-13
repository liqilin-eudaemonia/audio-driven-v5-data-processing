"""Publish normalized <=10s subturns on the existing current preview routes."""
import json
from pathlib import Path
import publish_speaker_span_preview as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data_previews/seamless_subturns_v5"


def main():
    manifest=json.loads((OUT/"manifest.json").read_text())
    for p in manifest["pairs"]:
        for c in p["clips"]:
            if not (OUT/c["path"]/"complete.json").is_file():
                raise ValueError("Cannot publish incomplete media")
            if json.loads((OUT/c["path"]/"complete.json").read_text()).get("audio_mux_revision")!=2:
                raise ValueError("Complete-audio remux must finish before publication")
    base.OUT=OUT
    base.main()
    path=OUT/"index.html"
    t=path.read_text()
    changes={
      "Seamless · Speaker 起止 V4":"Seamless · 子话轮与统一音量 V5",
      "SPEAKER SPAN 04":"SUBTURNS 05",
      "Speaker 起止 V4 · 倾听音视频":"子话轮与统一音量 V5 · 倾听音视频",
      "Speaker 起止 V4 · 原 50 个会话":"子话轮 V5 · 原 50 个会话",
      "按 Speaker 首尾词统一裁剪 · 保留内部停顿与重叠反馈 · 512×512":"子话轮不超过 10 秒 · 优先停顿处分割 · 统一语音响度 · 512×512",
      "'seamless_speaker_span_v4'":"'seamless_subturns_v5'",
      "segmentation_revision:4":"segmentation_revision:5",
      "key:p.pair+'/'+c.turn_id+'/'+c.part_index":"key:p.pair+'/'+c.turn_id+'/'+c.part_index+'/'+c.subturn_index",
      "${c.turn_id}.${c.part_index}</span>":"${c.turn_id}.${c.part_index}.${c.subturn_index}</span>",
      "const reasons={":"const reasons={speaker_pause:'Speaker 停顿',speaker_word_gap:'Speaker 词间隙',hard_10s_no_word_gap:'无合适词间隙，按 10 秒切分',",
      "去掉开头 ${c.head_removed_seconds.toFixed(3)} 秒、结尾 ${c.tail_removed_seconds.toFixed(3)} 秒。":"子话轮 ${c.subturn_index}/${c.subturn_count}；speaker 增益 ${c.normalized_audio.speaker.gain_db.toFixed(1)} dB，listener 增益 ${c.normalized_audio.listener.gain_db.toFixed(1)} dB。",
      "完整会话与切点上下文（含双方声音）":"完整会话与切点上下文（原始音量，含双方声音）"
    }
    for a,b in changes.items():
        if a not in t: raise ValueError("Missing preview anchor: "+a)
        t=t.replace(a,b)
    old='基于 V2 的 457 个片段，统一裁至 Speaker 第一个转写词开始、最后一个转写词结束；双方视频和独立音轨使用相同区间。去掉区间外的 Listener 发声，保留区间内重叠反馈与 Speaker 内部停顿。已有排除区间仍保留，不运行新模型。词时间戳为自动标注；25 FPS 视频终点存在帧量化误差。'
    new='基于 V4 的 457 个连续区间，长片段进一步切成不超过 10 秒的子话轮。优先选择 Speaker 停顿或词间隙，尽量避开 Listener 短反馈；子话轮连续且不重叠，父区间的时间完整保留。按原始参与者整条音轨测量响度并施加统一增益，目标 −18 LUFS；峰值限制 −1 dB，静音不单独放大。完整会话上下文仍使用原始音量。'
    if old not in t: raise ValueError("Missing notice")
    t=t.replace(old,new)
    path.write_text(t)
    # Regenerate font after introducing the V5 labels.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=t);sub.subset(font);font.flavor='woff2';font.save(OUT/'assets/review-chinese.woff2')
    for name in ("rule_turns","speaker_span"):
        link=ROOT/"data_previews/seamless_face_listening_v1"/name
        if link.is_symlink(): link.unlink()
        elif link.exists(): raise ValueError("Refuse replacing a real directory")
        link.symlink_to(OUT,target_is_directory=True)
    (ROOT/"data_previews/seamless_face_listening_v1/index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=rule_turns/">'
        '<title>子话轮与统一音量 V5</title><a href="rule_turns/">打开当前 V5 预览</a></html>')
    print("Updated /rule_turns/ and /speaker_span/ to V5")


if __name__=="__main__":main()
