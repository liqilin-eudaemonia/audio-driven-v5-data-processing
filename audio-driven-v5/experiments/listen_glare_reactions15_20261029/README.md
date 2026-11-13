# GLARE 六类视觉检测补充：15 段

**后续决定（2026-10-29）：** 用户确认剩余两条摇头仍是误判，已选择“原生视频 8 fps＋高召回点头补充”。本文件保留历史实验过程，摇头及其他未确认的专用检测器不进入当前默认输出。参见总索引顶部的当前方案交接。

**当前查看版本：摇头方向检查 v1。** 用户指出点头被误报为摇头后，已增加物理角度主方向与反向运动检查。摇头从 20 条暂缩减为 2 个待复核候选；其他五类原样保留，当前补充列表 45 条。[新版本、原因和代码](headshake_direction_v1/README.md)。下表为原始六类 63 条基线；新版入口为 `headshake_direction_v1/supplementary_events.jsonl`。

用户认可上一轮点头高召回候选基本正确，并要求补齐参考工作的其他工具。沿用相同 15 段，点头结果逐项复用，新增摇头、微笑、视觉大笑、皱眉和惊讶样表情。作为 VLM 的独立候选补充，尚未写入生产标签或训练缓存。

[原页面已更新为六类对照](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/)。Cursor 转发 8876。历史点头三参数结果折叠保留；各类别可分别反馈、导出。视频、原始 listener／处理后 listener／双人参考音轨均沿用原页面，不新增音视频文件。

## 结果

| 类型 | 有候选片段 / 15 | 候选区间 | 区间总时长 |
|---|---:|---:|---:|
| 点头（高召回） | 13 | 16 | 10.920 秒 |
| 摇头 | 12 | 20 | 24.358 秒 |
| 微笑 | 10 | 14 | 23.765 秒 |
| 视觉大笑 | 4 | 4 | 6.168 秒 |
| 皱眉 | 4 | 6 | 22.149 秒 |
| 惊讶样表情 | 3 | 3 | 8.762 秒 |

15/15 完成，共 63 个候选区间。时长按各类别单独计算，不能跨类别相加理解为不同视频时长。609 帧保留多类别同时发生；未采用参考代码的逐帧最大值互斥合并。

单路 CPU FER，Torch/TF 最多各 2 个计算线程、OpenCV 单线程，无 GPU。正式补跑墙钟约 77.5 秒，其中首段复用约 9.5 秒首测结果；不含环境安装、初始化、页面检查。此前 MediaPipe 无需重算。

## 具体方法及限制

- 点头：复用 `../listen_glare_nodding50_20261029/results/` 的 `recall` 完整帧级分数与候选区间，未重新调参。用户认可记作配置级反馈，不伪造逐事件人工真值。
- 摇头：原 `head_shaking/run_head_shaking_intensity_mt.py`，偏航动态及俯仰抑制，使用 Python 默认参数。
- 微笑：FER happy 与 mouthSmile blendshape。
- 视觉大笑：FER happy、mouthSmile、jawOpen 与下颌动态；**没有音频输入，不代表存在笑声**。
- 皱眉：FER sad/angry、mouthFrown、browDown、AU04 proxy；AU04 proxy 不是专业 AU 测量真值。
- 惊讶样表情：FER surprise、jawOpen、eyeWide、browInnerUp。标签是外观候选，不能据此推断人物真实心理。

完整权重、阈值、平滑、事件最短长度取自本地参考 Python 实现，包括代码中已经修改过的皱眉、惊讶和摇头权重；没有擅自切回注释里的旧公式。`protocol.json` 保存参数和源代码 SHA256。

FER 环境：`fer==25.10.3`，`FER(mtcnn=True, use_tflite=True)`，TensorFlow CPU 2.15.1，facenet-pytorch 2.6.0；参考仓库没有锁定 FER 版本，因此不能声称逐版本复现其论文结果。保留模型、MTCNN 权重及 FER 代码哈希。每帧 BGR 输入，完整保留所有人脸结果；多脸时选与同帧 MediaPipe 框 IoU 最大的人脸，无 MediaPipe 框时选最大脸。情绪分数保持 FER 返回的数值，不自行改写为行为语言。

**已发现的实际覆盖问题：**2296 帧中 FER 有 277 帧未检出人脸；S0694 仅 1/121 帧，S0343 为 45/125，S0284 为 185/234，S0333 为 88/104。原规则将缺失 FER 项按零处理，依旧可能通过 MediaPipe 信号触发表情候选。这可能导致漏检或区间断裂，不能把空结果当成已确认无表情；页面显示覆盖率，并对 FER 覆盖低于 90% 的片段提示。90% 仅为展示提示阈值，不改变算法输出。点头和摇头不使用 FER，不受这一层缺失影响。本轮保留参考算法基线，未偷偷改用其他框或模型补分数。

摇头在 12/15 段中检出，优先核对是否混入普通转头／姿态调整；皱眉和惊讶的组合分数可能饱和，不是校准后的概率。各类别强度不能直接竞争取最高分。微笑和视觉大笑可能描述同一时段的不同层次，最终生成指导不应直接重复拼接所有候选文字。

## 输出与接入

- `supplementary_events.jsonl`：每段一行，事件带 `type/start/end/modality/source/status/configuration`，`status=candidate`、`training_admitted=false`，与原 VLM 观察分别保存。
- `results/<id>.json`：六类完整帧级分数、区间、FER 检出掩码、输入哈希和候选事件。
- `fer_raw/<id>.json`：每帧所有人脸、FER 概率、选中索引；没有跳过未检出帧。
- `fer/<id>.csv`：零起点逐帧七情绪分数与 `detected` 标记，供参考工具读取；不是七类 backchannel 标签。
- `scores/<type>/<id>.csv`：五类新增检测器原生输出。点头 CSV 保留在上一轮 `../listen_glare_nodding50_20261029/scores/recall/`。
- `protocol.json`、`summary.json`、`requirements.freeze.txt`、`run.log`：参数、统计、环境和运行记录。
- 原 VLM 原文、原时间区间、音频配方、训练缓存均保留。没有用新类别硬映射或重写已有自然语言。

参考仓库其他工具的处理边界：MediaPipe 特征和 FER 已纳入；`combine_reactions.py` 的互斥策略不采用；HDF5 注入/裁剪、抽样可视化、路径修复属于导出维护工具，不是新增反应检测器。Qwen2-Audio prosody 分析 **speaker 音轨**，不属于 listener backchannel，未混入本轮标签。旧 py-feat 脚本不作为额外独立投票重复计数。

## 代码与复现

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/tmp/glare-reaction-env/bin/python scripts/run_glare_reaction_pilot.py
/tmp/glare-nod-pilot-env/bin/python scripts/check_glare_reaction_results.py
python scripts/build_glare_nodding_review.py --with-reactions
FONTCONFIG_FILE=/tmp/native-fps-fonts.conf /tmp/native-fps-review-qa/bin/python scripts/check_glare_nodding_review.py
```

独立 venv `/tmp/glare-reaction-env` 只读继承 qwenomni 的现有 Torch/torchvision，CPU 运算；TF、FER、OpenCV 等安装在此 venv 内，没有更新训练环境。`requirements.freeze.txt` 记录实际版本。facenet-pytorch 的发布依赖限制与继承 Torch 版本不完全一致，安装时避免自动替换基础 Torch；本轮实际 MTCNN/FER 推理和逐帧验收通过，不据此宣称所有版本组合兼容。

验收：`audit_results.json` 逐帧核对 FER 原始值和 CSV、原样重放五个检测器函数、确认点头结果不变、所有区间有界、保留多类并发和来源；`audit_browser.json` 检查 15 段六类面板、分类筛选、按类别反馈保存/导出，以及音视频 HTTP206、同步播放、慢放、手机布局和无 JS 错误。自动测试反馈不进入人工反馈数据。
