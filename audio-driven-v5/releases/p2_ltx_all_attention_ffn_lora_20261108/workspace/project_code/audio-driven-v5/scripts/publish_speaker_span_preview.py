"""Publish V4 speaker-bound clips without changing the historical V2 route."""
import json
from pathlib import Path
import publish_rule_turns_preview as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/seamless_speaker_span_v4'
V1=ROOT/'data_previews/seamless_face_listening_v1'


def main():
    base.OUT=OUT; base.main()
    path=OUT/'index.html'; t=path.read_text()
    replacements=[
        ('Seamless · 规则话轮 V2','Seamless · Speaker 起止 V4'),
        ('RULE TURNS 02','SPEAKER SPAN 04'),
        ('规则话轮 V2 · 倾听音视频','Speaker 起止 V4 · 倾听音视频'),
        ('停顿不单独切断话轮 · 插话独立标记 · 完整会话上下文 · 512×512',
         '按 Speaker 首尾词统一裁剪 · 保留内部停顿与重叠反馈 · 512×512'),
        ("'seamless_rule_turns_v2'","'seamless_speaker_span_v4'"),
        ('剔除插话后的连续倾听子区间','按 Speaker 起止裁剪的连续倾听区间'),
        ('规则话轮 V2 · 原 50 个会话','Speaker 起止 V4 · 原 50 个会话'),
        ('<a href="../index_v1.html">第一版对照</a>',
         '<a href="../rule_turns/">当前预览</a>'),
        ('本版已重做话轮切分，仅使用现有转写、VAD 和明确规则，没有运行标注模型。主话轮允许停顿；插话、短问句及不确定发言从倾听导出区间中排除，并保留原时间。以下仍是规则候选，请用完整会话检查接管边界。',
         '基于 V2 的 457 个片段，统一裁至 Speaker 第一个转写词开始、最后一个转写词结束；双方视频和独立音轨使用相同区间。去掉区间外的 Listener 发声，保留区间内重叠反馈与 Speaker 内部停顿。已有排除区间仍保留，不运行新模型。词时间戳为自动标注；25 FPS 视频终点存在帧量化误差。'),
        ("const reasons={","const reasons={speaker_first_word_start:'Speaker 第一个已转写词开始',speaker_last_word_end:'Speaker 最后一个已转写词结束',"),
        ('current.listener_events.map(e=>[Math.max(0,e.start-current.start)',
         "current.listener_events.filter(e=>e.label!=='untranscribed_vocalization').map(e=>[Math.max(0,e.start-current.start)"),
        ('Listener 转写 / 规则保留的声音事件','Listener 转写'),
        ('c.listener_events.length?c.listener_events.map(e=>',
         "c.listener_events.some(e=>e.label!=='untranscribed_vocalization')?c.listener_events.filter(e=>e.label!=='untranscribed_vocalization').map(e=>"),
        ('原有 VAD 与 listener 转写独立展示','原有 VAD 与文字转写独立展示；VAD-only 不作为转写'),
        ('segmentation_revision:2','segmentation_revision:4'),
        ('end:c.end,parent_turn:',
         'end:c.end,previous_start:c.previous_start,previous_end:c.previous_end,head_removed_seconds:c.head_removed_seconds,tail_removed_seconds:c.tail_removed_seconds,boundary_policy:c.boundary_policy,parent_turn:'),
        ('终点：${reasons[c.end_reason]||c.end_reason}。',
         '终点：${reasons[c.end_reason]||c.end_reason}。去掉开头 ${c.head_removed_seconds.toFixed(3)} 秒、结尾 ${c.tail_removed_seconds.toFixed(3)} 秒。'),
        ("${e.exclude?'排除':'保留候选'}", "${e.exclude?'规则排除':e.in_export?'本段保留':'本段范围外'}"),
        ('40+i/Math.max(1,e[k].length-1)*800',
         '40+(current.emotion_sample_times[i]/current.duration)*800'),
    ]
    for a,b in replacements: t=base.replace(t,a,b)
    path.write_text(t)
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter(); sub.populate(text=t); sub.subset(font)
    font.flavor='woff2'; font.save(OUT/'assets/review-chinese.woff2')
    link=V1/'speaker_span'
    if not link.exists(): link.symlink_to(OUT,target_is_directory=True)
    print('Published at /speaker_span/; /rule_turns/ can be updated after validation')


if __name__=='__main__': main()
