# 当前500h Whisper与Feature teacher独立增量核验

覆盖全部284,218段、保留train 281,060＋dev 3,158，test不加入。逐样本分别核对speaker原轨／归一化与时间观测、listener视觉来源/时间/最终裁框和音频配方。旧模型、原代码与数值规则必须匹配；旧缓存完整验收、逐样本contract/identity、SHA记录、文件存在与字节数必须有效。旧payload在读取时重算SHA；本次成员审计没有为两项旧缓存再次全量读取约2TB数据做哈希。

| 缓存 | 整份复用 | 仅复用音频、重算视频 | 新成员 |
|---|---:|---:|---:|
| speaker Whisper | 277,676 | 不适用 | 6,542 |
| listener Feature teacher | 276,780 | 896 | 6,542 |

因此Whisper增量6,542段；teacher视觉7,438段、音频6,542段。身份异常0。结果见SUMMARY.json，逐样本映射reuse_map.jsonl；三份增量清单为whisper_increment_ids.json、teacher_video_increment_ids.json、teacher_audio_increment_ids.json。这里只读核验，不修改旧缓存或训练成员；实际提取位于对应独立新目录。
