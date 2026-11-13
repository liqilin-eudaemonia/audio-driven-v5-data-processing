# Qwen3-Omni：净化 listener 音轨＋视频的整段自然语言描述

2026-10-30 完成。用户明确要求自然语言描述，不做“净化音轨效果”的消融实验，也不输出逐事件时间戳表。

- 范围：`native_simple` 历史页面同一批 50 条，ID、listener 裁框、8 fps 帧时间和 JPEG 哈希逐条与历史输入一致。
- 模型：`/private/ckpts/Qwen3-Omni/Instruct`；原生视听输入为裁剪后的 512×512 视频数组＋同步 listener 波形，`use_audio_in_video=true`、`second_per_grid_ts=[0.25]`。
- 音频：31 条复用已完成全量后处理的配方，19 条旧测试样本按同一已确认规则补齐。官方词／句／VAD 并集、100 ms 容差、10 ms 淡入淡出，以及既有多字中文 ASR 静音规则不变。40 条全静音，10 条仍有声音。
- 复用的部分全量音轨为 PCM16，历史测试包为 PCM24；已核对同一来源时段、采样长度及波形差值不超过一个 PCM16 量化步长。模型使用配方指向的原音轨，不隐式改绑来源。
- 音轨在内存中按配方处理，不删采样，不移动时间轴，不新增处理后 WAV，不修改训练标签或缓存。
- 模型只接收媒体与任务 prompt，未喂入旧标注、官方转写文本或人工反馈。官方信息仅用于音频净化。

## 当前结果与入口

当前有效试验目录为 [`text8/`](text8/)，而非根目录的首次尝试结果。

- [有声复核 HTML](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption/)，原 `native_simple` 页面顶部也已增加入口。左侧视频＋可切换音轨，右侧英文原始描述、完整 prompt、旧事件参考、人工复核问题与 JSON 导出。
- [自动验收](text8/audit.json)：50/50 非空自然语言段落、50/50 正常结束，未截断；每段 34–146 个英文词。50 条帧时间和 JPEG 哈希均与历史输入相同。
- [原始结果](text8/results/)：逐样本 caption、prompt、渲染后的请求、输入证据、来源哈希、生成时长和 token 数完整保留。
- [冻结配置](text8/protocol.json)、[完整 prompt](text8/prompt.txt)、[输入清单](text8/manifest.json)、[静音配方](recipes/)、[代码快照](text8/code/)、[源码哈希](text8/code_hashes.json)。
- [浏览器验证](browser_validation.json)：有声播放正常；浏览器生成的净化 PCM 与 Python 读取入口逐样本 SHA256 一致，50/50；无页面 JS 错误，手机宽度无横向溢出。

自然语言重点表达动作顺序、持续状态、同时发生与真实重复；引用听清的回应原词，不预设行为类别。完整 prompt 以文件及实际请求为准。

## 八卡执行与初次失败

node4 `172.29.225.51` 最初仅后四卡空闲；随后用户授权全八卡，前四卡的其他任务也已释放。本轮没有停止其他人的进程。

初次自由解码存在空行循环、空回复和长文本重复，根目录 `results/` 保留这些失败／部分完成记录，不冒充最终结果。切换前只停止了本轮自己的进程。

最终 `text8` 使用 0–7 卡各一副本，分配为 6/6/6/7/6/6/6/7 条。保持原生视听的单请求完整 prefill，CPU 每卡两个准备线程、按时长均衡；启用 decode-only CUDA Graph。输出约束为单段文本正则 `[A-Za-z][^\n\r]*`，temperature=0、repetition_penalty=1.05、4096 token 预算，未限定段落中的动作内容。

各 worker 模型加载后执行耗时约 21.4–44.8 秒，包含媒体准备；50 次推理耗时之和约 98.2 秒。以上不含模型加载、前置准备、第一次失败、页面构建，也不是全量吞吐测量。显存占用不是计算利用率，不能声称 GPU 持续满载。所有八卡均实际生成了结果，完成后进程正常退出。

## 仍需人工复核

自动验收只验证输入、格式与执行完成，不证明感知正确。页面保留了完整原文，未事后去掉 `Assistant:` 前缀、人物外观细节或推演措辞。

- S0805 描述包含两次 Right 和五次 yeah，但发声时点头是否漏掉仍需看画面。
- S5193 只写出了 Yeah，没有写此前人工确认过的 Uh huh，并包含 contemplative、pensive、thoughtful 等措辞。
- S2043 需区分可见笑容与实际笑声，检查是否把短暂声音延伸到遮嘴动作。
- 50 条里 8 条触发措辞复核提示；这是词项提示，既不是人工错误数，也不覆盖全部错误。

## 代码与复验

入口分别为 `scripts/run_listen_native_caption_pilot.py`、`scripts/build_listen_native_caption_review.py`、`scripts/audit_listen_native_caption_pilot.py`。本次实际八卡运行的代码在 `text8/code/` 冻结。首次四卡代码在根目录 `code/` 留档。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/audit_listen_native_caption_pilot.py --run experiments/listen_native_caption_clean50_20261030/text8
python scripts/build_listen_native_caption_review.py --run experiments/listen_native_caption_clean50_20261030/text8
```

不得把本轮候选段落自动覆盖到正式训练集。本轮没有启动全量再打标或训练。
