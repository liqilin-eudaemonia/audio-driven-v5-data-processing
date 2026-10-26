# 打标 HTML 对照页

入口：[本地预览](http://127.0.0.1:8875/listen_benchmark_review/)；[独立 HTML](../data_previews/listen_benchmark_review/index.html)；[选型分析](../experiments/listen_benchmark_retest_20261024T032648Z/选型分析.md)。

首页默认展示开放描述分析，包含独立 discovery 统计和 5 个复核样本。点击样本卡进入开放描述并排比较；“点头／发声审计”标签保留原完整方案统计。点击“打开点头分歧样本”，查看本轮视觉事件数量不一致的诊断样本；这不是已确认错误集。逐样本页默认使用 discovery，可选 2–4 个方案及 discovery/audit 阶段。复核样本卡会选中对应的三个配置；“恢复默认对照”恢复 Joy＋Omni、Qwen3-VL＋Omni 和 Qwen 原生视听。点击事件会跳转并循环对应区间；点击时间轴其他位置取消区间循环。

播放器使用原帧率 512×512 裁剪预览及独立原始 WAV；不是各模型内部抽帧/拼图编码的复刻。所有媒体来源指纹均与最终实验一致。音频单独播放，预览视频静音，无重叠音轨。原始请求、回答和输入证据可展开查看或下载；人脸大文件仅按需下载。

人工判断按样本、阶段、所选方案组合保存在当前浏览器 localStorage，可导出 JSON；未保存的编辑不会自动落盘。隐藏名称仅遮盖界面模型名，不是正式随机盲评。人工比较意见不会写回实验数据，也不能代替逐事件 GT。

`index.html` 内嵌全部比较数据，直接打开可看总览；完整媒体复核建议使用 HTTP 地址。移动整个 `data_previews/listen_benchmark_review/` 目录后也可运行静态服务；复制单个 HTML 不会携带音视频。`listen_benchmark_review_bundle.zip` 是该目录的便携副本。项目原 8875 服务目录通过一个独立软链接提供新页面，没有重启服务或模型。

## 复查修订（2026-10-24）

撤回 Joy＋Omni 的主候选推荐，全部方案仍待质量验收。灰色“未检查”改为“未声明有效覆盖”，仅指模型输出的覆盖区间；“本阶段通过”改为“结构校验通过”。已发现 Joy 备注描述持续点头但事件为空的语义矛盾。

vLLM 0.19 原生视听旧适配器未传 `second_per_grid_ts`，缓存路径的音视频 token 交错使用默认 2 秒，实际应为 0.25 秒（8 fps、每组 2 帧）。CPU 预处理对照已复现；当前适配器补充显式参数，旧实验快照和初轮页面保留。本轮已在三节点 24 卡重跑全部 16 种配置、50 条样本，共 1450 次请求；修正版原生视听发声事件 4→14，点头仍为 1。主页面现展示本轮结果，并可展开上轮原生视听回答。事件准确率仍未经过人工真值验证。诊断脚本：`scripts/check_vllm_native_alignment.py`；结果：`experiments/listen_benchmark_cluster_20261024T014809Z/native_alignment_diagnosis.json`。

## 复现与检查

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
BENCH_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$BENCH_PY" scripts/build_listen_benchmark_review.py \
  --run experiments/listen_benchmark_retest_20261024T032648Z

PYTHONPATH=/tmp/v8_browser_deps \
LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu \
"$BENCH_PY" scripts/check_listen_benchmark_review.py
```

检查报告见预览目录内 `build_validation.json` 与 `browser_validation.json`；截图为 `overview_desktop.png`、`compare_desktop.png`、`compare_mobile.png`。浏览器测试的模拟判断仅存在于独立测试上下文，不是人工验收结果。

完整重测目录：`experiments/listen_benchmark_retest_20261024T032648Z`；`rerun_validation.json` 核对全部 1450 次请求的样本、prompt、预算和媒体一致性，100 次原生视听调用均携带 0.25 秒时序参数。`release_check.json` 记录结束后三节点全部 GPU 已释放。

## 访问连接修复（2026-10-24）

原服务仅运行在 node12；本地转发若连接 node17 或 node4，远端 8875 没有监听，可能导致浏览器 ERR_EMPTY_RESPONSE。现已在 node17、node12、node4 的 8875 端口提供同一份页面，均验证完整 HTML 返回 200、音视频 Range 请求返回 206；新增服务的进程记录和验证结果见 `experiments/listen_benchmark_preview_service/`。

`http://127.0.0.1:8875/listen_benchmark_review/` 用于浏览器所在电脑已将远端 8875 转发到本地 8875 的情况。若本地转发使用其他端口，应使用实际本地端口。20342/20382/20359 是原机器连接信息中的外部端口，不应直接当作此页面的 HTTP 端口。

### 本地动态转发再次恢复

用户实际访问地址为 `http://localhost:63034/listen_benchmark_review/`。旧会话的 Cursor IPC 已拒绝连接；三节点 HTTP 服务持续正常。通过当前在线 Cursor 的 `openExternal`，为 node12 的新远端端口 **8876** 打开页面。服务器收到带 `access_check=fresh63034` 的浏览器页面请求（200），随后字体和音视频请求正常（206）。这次验证经过实际编辑器／浏览器链路，而非仅服务器内自测。

本地端口由 Cursor 分配，应使用新打开标签页的实际地址，不能假定本地端口等于远端的 8875 或 8876。恢复记录见 `experiments/listen_benchmark_preview_service/client_access_recovery.json`。

开放描述分析只统计结构通过的 discovery 回答，失败保留原文；眼部词语、长区间和连续重复是诊断指标，不是自动真值或删除规则。三个视觉例子仅抽看原帧率裁剪预览，未做完整人工事件标注，音频未听审。结果与口径见预览目录的 `开放描述分析.md` 和 `discovery_analysis.json`。本次未调用模型重新打标。


## 原生视听逐次事件 prompt 对照

后续页面位于 `data_previews/listen_benchmark_review/native_prompt/`，主页面顶部已有入口。对应实验为 `experiments/listen_native_prompt50_20261024T063000Z`：node8 的 8 卡完成 50 段 × 新旧两版 prompt，共 100 次请求。页面含独立音轨能量图、模型区间、原始回答和用户反馈；能量图不等于人工事件真值。新 prompt 能拆开部分短词，但有定位偏差、视觉描述减少及机械重复问题，尚不推荐生产使用。

构建：`$BENCH_PY scripts/build_listen_prompt_review.py --run experiments/listen_native_prompt50_20261024T063000Z`。


## 简短 prompt 复测

最新页面为 `data_previews/listen_benchmark_review/native_simple/`，主页面顶部有入口。对应 `experiments/listen_native_simple50_20261024T064600Z`：51 词 prompt、四字段事件输出、50 段重新推理。校验失败但可解析的原始事件仍展示；零长度区间禁用循环播放；截断不当作无事件。默认对比原版和简短版，可加入上一版长 prompt。

构建：`$BENCH_PY scripts/build_listen_prompt_review.py --run experiments/listen_native_simple50_20261024T064600Z --compare-run experiments/listen_native_prompt50_20261024T063000Z --output-name native_simple --template listen_simple_review.html`。浏览器检查使用相同依赖环境运行 `scripts/check_listen_simple_review.py`。

资源对照见 `experiments/listen_native_prefetch8_20261024T065200Z/performance_result.json`：8 副本每个相同 8 段，4 个单客户端、4 个提前准备；不含模型加载，平均 45.0 秒降为 30.7 秒，8 副本原始回答一致。仅是小样本实测，不宣称 GPU 满载。
