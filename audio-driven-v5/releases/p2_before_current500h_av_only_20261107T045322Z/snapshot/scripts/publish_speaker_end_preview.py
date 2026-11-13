"""Publish V3 with explicit speaker-end provenance and separate VAD/transcript rows."""
import json
from pathlib import Path
import publish_rule_turns_preview as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_previews/seamless_speaker_end_v3'
OLD = ROOT / 'data_previews/seamless_rule_turns_v2'
V1 = ROOT / 'data_previews/seamless_face_listening_v1'


def main():
    base.OUT = OUT
    base.main()
    p = OUT / 'index.html'
    t = p.read_text()
    replacements = [
        ('Seamless · 规则话轮 V2', 'Seamless · Speaker 结束 V3'),
        ('RULE TURNS 02', 'SPEAKER END 03'),
        ('规则话轮 V2 · 倾听音视频', 'Speaker 结束 V3 · 倾听音视频'),
        ('停顿不单独切断话轮 · 插话独立标记 · 完整会话上下文 · 512×512',
         '裁至 Speaker 最后一个词结束 · 保留话轮内部停顿 · 512×512'),
        ("'seamless_rule_turns_v2'", "'seamless_speaker_end_v3'"),
        ('剔除插话后的连续倾听子区间', '裁至 Speaker 结束的连续倾听区间'),
        ('规则话轮 V2 · 原 50 个会话', 'Speaker 结束 V3 · 原 50 个会话'),
        ('<a href="../index_v1.html">第一版对照</a>',
         '<a href="../rule_turns_v2/">V2 裁尾前对照</a> · <a href="../index_v1.html">第一版对照</a>'),
        ("const reasons={", "const reasons={speaker_last_word_end:'Speaker 最后一个已转写词结束',"),
        ('current.listener_events.map(e=>[Math.max(0,e.start-current.start)',
         "current.listener_events.filter(e=>e.label!=='untranscribed_vocalization').map(e=>[Math.max(0,e.start-current.start)"),
        ('原有 VAD 与 listener 转写独立展示', '数据集原有 VAD；转写行仅显示有文字的事件，不含 VAD-only'),
        ('本版已重做话轮切分，仅使用现有转写、VAD 和明确规则，没有运行标注模型。主话轮允许停顿；插话、短问句及不确定发言从倾听导出区间中排除，并保留原时间。以下仍是规则候选，请用完整会话检查接管边界。',
         '本版在 V2 的同一批 457 个片段上收紧终点：裁至 Speaker 最后一个已转写词结束，不补尾部静音。主话轮及内部停顿保留，没有运行新模型。末尾的 Listener 反馈也可能被截去；可在完整会话或 V2 对照中查看。转写与 VAD 均为原有自动标注，需试听复核。'),
        ('segmentation_revision:2', 'segmentation_revision:3'),
        ('end:c.end,parent_turn:', 'end:c.end,previous_end:c.previous_end,tail_removed_seconds:c.tail_removed_seconds,end_policy:c.end_policy,parent_turn:'),
        ('终点：${reasons[c.end_reason]||c.end_reason}。',
         '终点：${reasons[c.end_reason]||c.end_reason}。本次去掉尾部 ${c.tail_removed_seconds.toFixed(3)} 秒（原终点 ${fmt(c.previous_end)}）。'),
    ]
    for a,b in replacements:
        if a not in t: raise ValueError('Missing anchor '+a[:80])
        t = t.replace(a,b)
    p.write_text(t)
    # Regenerate the local font subset after adding V3 copy.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font = TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', fontNumber=2)
    sub = subset.Subsetter(); sub.populate(text=t); sub.subset(font)
    font.flavor='woff2'; font.save(OUT/'assets/review-chinese.woff2')
    for name,target in [('speaker_end',OUT), ('rule_turns_v2',OLD)]:
        if not (V1/name).exists(): (V1/name).symlink_to(target, target_is_directory=True)
    print('V3 published at /speaker_end/; current entry unchanged until validation')


if __name__ == '__main__': main()
