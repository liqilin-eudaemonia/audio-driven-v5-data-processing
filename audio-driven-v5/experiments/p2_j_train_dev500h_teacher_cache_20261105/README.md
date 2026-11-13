# 当前500h listener Feature teacher增量缓存（2026-11-05）

当前284,218段train＋dev：276,780份旧teacher整份引用；896个已转正样本仅重提视觉teacher、保留原音频teacher；6,542个新成员提取视音频。生产只需7,438段视频塔、6,542段音频塔，不重提整批500h。状态见chain_status.json，最终完整视图以MIXED_COMPLETE.json为准，COMPLETE.json只记录7,438个新payload。

原始冻结Qwen2.5-Omni-3B视觉／音频塔，BF16、SDPA、不加载LoRA、不运行语言规划。视觉遵循最终512裁框→25fps原生网格选8fps→224×224，4Hz时间组和8×8空间网格、2048维；listener音轨使用同一官方净化配方并重采样16k，聚合8Hz、2048维。静音保留有效支持，mask仅排除时间尾部不完整支持。

复用core.extract_batch、core.persist、原媒体选帧/音频/processor，沿用已验证的4线程短seek解码。新适配器只接当前train＋dev和转正来源；没有修改原teacher目录。CPU producer各有独立processor，双模型进程／卡，共64个，4个producer／进程、预取16、原生batch8逐片段tower前向、2线程后台写盘、共享动态队列每块32条。原生逐片段数学定义不改为合批视觉前向。

先完成8条真实对照：加速解码与完整25fps源解码的像素／PTS一致；新teacher和保留旧音频的结果均与完整原生视音频塔逐值一致。生产每份新payload原子写、SHA和逐张量回读；最后32进程对全部新payload做SHA、张量和时间几何验收。旧引用绑定独立aux审计及旧全量验收，读取时继续校验SHA。

```python
import sys
sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/scripts")
from p2_j_current_teacher_cache import Reader
r = Reader("/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train_dev500h_teacher_cache_20261105")
x = r.load(sample_id)  # video/audio/video_valid/audio_valid，带来源audit
```

每个进程独立构造Reader；完整成员和原split取当前H来源，默认不再按旧train-only过滤。新payload在cache/<sample_id>；不复制旧teacher文件。未启动训练。

**完整500h已完成并验收：** 284,218段全覆盖，见[MIXED_COMPLETE.json](MIXED_COMPLETE.json)。3个CPU供数超时块保留原记录后，仅重试64个缺失成员，原生单线程解码＋无损NumPy IPC；最终独立单线程CPU主进程执行32进程SHA／张量／几何审计通过。新旧teacher与Qwen、Gemma、AV、Whisper联合回读通过。未重新提取整批数据，当前无未解决失败。
