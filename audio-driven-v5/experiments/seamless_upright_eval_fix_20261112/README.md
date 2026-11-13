# Seamless 评测转正来源修复（2026-11-12）

旧训练／打标流程已完成转正，但评测仍引用旧 TAR。此次建立新冻结版本，不覆盖旧素材、旧输入清单或历史生成。

## 数据入口

- [当前机器入口](../../evaluation/SEAMLESS_CURRENT.json)。新固定200：[prepared_200/references.jsonl](prepared_200/references.jsonl)。
- 同一200条、同一角色与时间：4条更换为已发布 upright 来源；核心48涉及1条；完整6624涉及33条。
- 原始样本已在 `listen_j_upright_response_sft_20261105/source` 中存在，复用其校验读取与已修复 TAR/NUT，不重新转正或重新打训练标签。
- 修改读取器以识别已转正来源记录，并维持原 PTS、裁框、25fps 最近帧采样。派生短片尾部的最近帧允许补到原目标网格，仍受一个原生帧间距约束，超出则报错。
- [新旧首帧对照](first_frame_comparison.jpg)：上排转正来源、下排旧输入。四条首帧与新版完整GT的第0帧逐像素相同。

## 生成与人工页面

原人工核验使用 model6000＋SFT2752。`generation6000/` 对全部4条受影响的固定200样本重提首帧、Whisper、Qwen预测规划及H、Gemma条件，并按原checkpoint、seed137、30步、CFG3/7重新生成与解码。没有将旧生成视频旋转充当重新生成结果。

`full200_model6000/` 提供修复后完整200条绑定：4条重生成、196条已核对来源不变后复用。**旧200条总分不属于新版本，不将旧指标拷为新结果。** 其他模型历史运行亦不能直接声称已迁移；其受影响输出应按新来源重生成后再比较。

新人工核验入口和最终完成状态见 `REVIEW_PUBLISHED.json`。网页替换新素材时使用新媒体身份，旧标签不自动套到新素材；对完全相同的媒体保留浏览器已填标注。旧页面另存为 legacy，原自动检测与分数保留在旧目录。

## 执行范围

只在 node11 GPU5 执行本次4条增量生成及新pilot检测，不重训，不停止其他任务。当前运行状态见 `generation6000/status.json`。新事件检测使用本地已修复的提示版本，不能把与旧pilot的分数差解释为单一转正效果。

## 代码与验收

- `scripts/migrate_seamless_upright_eval.py`：固定200／核心48／完整6624迁移，校验角色、时间、成员及替换集。
- `scripts/repair_seamless_upright_outputs.py`：复用原checkpoint加载、规划、采样和解码代码，仅替换输入来源。
- `scripts/finalize_seamless_upright_eval.py`：完整200输出绑定。
- `scripts/publish_upright_event_review.py`：新pilot媒体与人工页发布。
- `before/` 保存修改前的读取器与设备检查；原事件推理把 nvidia-smi 设备写死为7，本轮改为读取配置中的 physical_gpu，并继续校验该卡UUID及空闲显存。
- Seamless adapter原有4项、事件评测22项测试通过。真实读回、最终生成、网页与CSV验证分别见对应 JSON，不把自动化测试标签当成人工结果。

## 已完成结果

- `generation6000/COMPLETE.json`：4条原模型重生成完成，speaker波形身份不变。
- `full200_model6000/pairs.jsonl`：可直接传给评测的完整200条新绑定，尚未重算完整200总分。
- [新版人工核验](http://localhost:8876/listen_benchmark_review/event_presence_upright_20261112/review_visual.html)：原链接自动跳转，8段共32项视觉判断。
- `REVIEW_BROWSER_CHECK.json`：8个视频实际加载；模拟旧浏览器标注时6段保留、2段换媒体后清空；刷新恢复、CSV导出和移动端检查通过。`TEST_ONLY_review.csv`仅为自动化测试，不是真实人工标签。
- `event_presence/` 为新来源、新提示版本的独立检测运行，状态见 `RUN_STATUS.json`；未完成前不宣称获得新分数。

新8媒体事件检测已结束，技术错误0，语义unknown按原响应保留；状态 `execution_complete_partial` 不表示仍在运行。已导入用户人工核验45项，剩11项须针对换后的媒体补标。[人工结果与逐项分歧](../event_presence_human_review_20261112/README.md)。
