# ViCo43：共同输入的统一本地比较

**自动测评与展示已完成（2026-11-07）。** 我们联合8000、DyStream、Avatar Forcing在全部43条正式test上完成视频比较；新增基线43条生成、BFM/DECA提取与官方评分全部通过。L2L四套人物迁移动作与文献数字分表，暂停中的重训诊断不计为正式基线。

[打开全部43条同步视频与评分](http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/) · [结果与解释](RESULTS.md) · [完整结果JSON](RESULTS.json) · [画质CSV](image_metrics.csv) · [逐条DECA动作CSV](deca_per_clip.csv) · [BFM独立补充CSV](bfm_metrics.csv) · [A/B盲评入口](http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/blind.html)。盲评仅完成工具准备，目前人工评分0份。

我们的画质/身份匹配与表情参考距离较好；两种baseline的CPBD更高，头姿差值区间跨0。我们的表情运动方差约为GT的37%，仍需检查反馈幅度是否不足。不能把本轮分数写成“所有指标领先”或完整自主AV任务的优越性证明。

[方案与边界](DESIGN.md) · [当前状态](status.json) · [生成协议及指纹](avatarforcing/protocol.json) · [输入验收](INPUT_AUDIT.json) · [权重实际加载检查](avatarforcing/WEIGHTS_LOADED.json) · [最短/最长片段试跑](avatarforcing/SMOKE_COMPLETE.json)

[阶段状态](pipeline_status.json) · [正式43条生成完成](avatarforcing/GENERATION_COMPLETE.json) · [最终评分协议及旧版本来源链](evaluation_protocol.json) · [试跑/正式运行逐字节一致性](SMOKE_FORMAL_REPRODUCIBILITY.json)。早期[两模型配对差值](EXISTING_DYSTREAM_PAIRED.json)保留为历史，最终三模型结果以RESULTS.json为准。

[全部86份视频完整解码](MEDIA_AUDIT.json) · [301条动作输出指纹及独立数值检查](NUMERICAL_AND_MEDIA_AUDIT.json) · [浏览器播放、切换、移动端及盲评导出检查](BROWSER_AUDIT.json) · [6项协议／并发回归测试](TESTS.log) · [执行环境与批次](IMAGE_BACKEND_AUDIT.json)。完整源码及数值的核验范围见各记录，浏览器播放为代表片段检查，不等于逐条人工观看。

[完成审计与结论边界](COMPLETION_AUDIT.md) · [最终交付指纹](COMPLETE.json)。三种视频模型的129份BFM输出也已逐项检查并独立复算FD/L1，最大误差0。

本轮使用node11（172.29.189.119）物理GPU7，单生成worker、主要CPU计算2线程、PyTorch分配上限10%显存；保留已有进程。CPU清晰度评分随已完成视频接续，最多4个CPU子进程。没有恢复L2L训练。

训练条件、额外输入和表示空间必须披露。本实验不是DIM原表复现；基线不生成listener声音，不为其计算自主声音指标。旧586/200探索子集与本次ViCo43无关，测试UUID不按模型成绩筛选。

## 执行入口

- [生成适配器](../../compare/deployment/vico_avatarforcing.py)：`prepare`、`smoke`、`generate`。正式结果逐条断点复用须通过协议和媒体指纹检查；同一时段不同listener保留独立UUID。
- [评分与统计](../../compare/deployment/vico_unified_evaluate.py)：`freeze`固定提取器和评分代码；`extract`提取新模型BFM/DECA；`images`调用原官方ViCo评分；`motions`、`summarize`聚合与来源聚类bootstrap。
- [完整页面构建](../../scripts/build_vico43_unified_review.py)、[独立媒体/数值验收](../../scripts/check_vico43_unified.py)、[时间轴与统计测试](../../tests/test_vico_unified_protocol.py)。
- [接续控制器](../../compare/deployment/vico_unified_finish.py)：观察指定的实际生成PID退出后顺序接评分，不自动重启失败生成或任何训练；[浏览器验收](../../scripts/check_vico43_unified_browser.py)、[视觉适当性盲评材料准备](../../scripts/prepare_vico43_blind_review.py)。盲评材料不等于已收集人工结果。

生成环境`compare/.envs/avatarforcing`隔离继承DyStream基础环境，单独固定transformers4.30.2/timm1.0.9。适配器显式将旧wav2vec weight_norm的g/v映射到当前PyTorch命名并逐张量验证；未修正的首次试跑已移至`avatarforcing/revisions/invalid_wav2vec_names_smoke/`，不计入正式结果。

最短1秒、最长约35秒的修正后试跑已完成。完整波形直接调用官方模型，避免CLI30秒截断。正式生成不读取listener真值音频或后续视频；评分阶段才使用GT。

两条试跑与正式输出的原生视频、评分副本均逐字节一致。首条新增视频330帧提取已完成，漏检0帧；旧GT的8帧DECA对照最大误差1.907e-6、裁剪变换误差0。[实测提取对照](avatarforcing/extractor_parity_0.json)。

全部43条新增片段漏检0帧，评分网格共16288帧；RTLCC覆盖42条、RWTLCC覆盖41条，其余43条。L2L补充保留172条原生动作，加上三种视频模型共301条动作输出，已核验各文件及独立复算L1，最大误差0。

CPU提前评分曾与接续脚本重复领取同一指标；已停止重复子进程，保留原CPU任务的全43条最终输出，并补充指标级flock与并发回归测试。[受控评分恢复及版本链](SCORING_RESUME.json) · [原日志及协议归档](revisions/scoring_lock/REASON.json)。模型、生成样本、评分算法及采样配置未因此调整。

旧BFM最初提取协议记录了初版评分源码；最终`SCORING_PROTOCOL.json`明确保留了短片段N/A处理的第二版指纹及初版源码归档。本次逐项核验后沿用最终版，没有跳过不一致检查或覆盖旧结果。

## 原有结果

[旧统一比较](../common_method_comparison_20261106/README.md) · [原ViCo43结果](../p2_compare_vico43_20261106/RESULTS.md) · [复现协议审计](../l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md)

新目录不覆盖旧视频、权重或评分。完成状态最终以实际43条生成、评分、媒体解码、数值及浏览器验收为准。
