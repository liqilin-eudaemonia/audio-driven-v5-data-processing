# 原生视频8／16 fps：轻量复核与独立测速

本轮复用已完成的50条输出，不重新填写训练标签。用户只需复核关键问题；原文、时间、失败响应全部保留。助手对照抽帧给出的判断不是独立人工真值。

## 页面与操作

[打开轻量复核页](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/fps_decision/)；也可从原三组对照页顶部进入。

服务运行在当前 node12：`172.29.252.37`。Cursor连接这台机器后转发远端8876；本地若分配了不同端口，使用Cursor给出的本地地址，路径不变。旧node12地址`172.29.252.11`已不可用。

- 推荐队列：9条关键分歧＋6条随机一致项，共15条。无需从零写描述或精确时间。
- 先看左侧视频，右侧回答一个具体问题；可选哪份更符合画面。没有预选答案，允许无法判断。
- 甲／乙映射按样本固定随机排列，默认隐藏帧率。候选判断默认折叠，建议作答后再展开。
- 随机项先核对高亮描述，再播放整段找共同遗漏。文字相同不等于正确。
- 保存到当前浏览器；导出JSON后交回复核。导出记录包含甲乙映射、是否显示帧率、是否打开助手判断、来源摘要和非真值状态。
- 可切换全部50条、整段／短窗，并试听原始listener、处理后listener、双人参考。试听不代表视觉模型输入了音频。

关键项：S0157、S0333、S0321、S1833、S1033、S2147、S1052、S2128、S0343。

随机项：S1236、S0284、S1668、S0339、S0694、S0127。固定种子20261030，从非关键样本中具有“文本相似度≥0.90、两端时间差均≤0.2秒”的片段先抽片段、再抽事件对。这个文本筛选仅用于省事抽查，不作语义归类、准确率或召回率。发现错误后不更换抽中样本。

`review_audit.json`记录输入摘要、抽样总体与具体事件对；`review_cases.json`记录问题和助手候选判断。`frames/`为随机项抽帧证据，原先关键样本证据在上级`qualitative_review/`。4fps抽帧不足以确认所有短瞬动作；眨眼等关键争议优先使用已有更密集帧序列并人工慢放复核。

## 独立测速

**当前状态：三次试跑均被SIGKILL终止，尚无完整速度比较。** node12两次、node4一次；单请求重试也未完成。已停止继续重启，待确认终止来源或提供能持续运行的资源条件。页面会显示中断，不能将已有少量批次当作最终速度倍数。

用户允许在现有训练上叠加，但控制并发。本机node12两次被SIGKILL后，切换node4（172.29.225.51）的GPU4；一个模型进程、两条CPU计算线程、一路原视频解码。两种帧率在同一卡上分别计时，不再共用混合批次耗时。

- 12条按时长三层固定随机抽样。当前node4正式重试只测整段，两个配置各运行两遍，共48次请求；不据此声称整段＋短窗全流水线速度。脚本也支持`--views full-and-middle`，前两次中断记录采用该模式。
- 输入与冻结实验逐帧记录一致：原listener视频裁512×512、JPEG85、同C prompt、同模型和生成参数。
- 每批按ABBA或BAAB交替，两个配置各重复两遍。预热不计时；禁用prefix缓存和多模态processor缓存，避免重复请求缓存造成顺序偏差。
- 只比较准备好输入后的`LLM.generate`时间；源视频采用一次16fps解码供两组共享，其耗时另记，不冒充各组端到端时间。
- `speed/resources.jsonl`每3秒记录GPU总负载及进程显存；总显存包含原训练，不能全算作打标占用。
- 原训练保持运行，因此只能报告共享GPU条件下的配对观察，不能外推独占GPU速度、全量工期或最大吞吐。
- 首轮batch=3、显存配额20%的父进程与引擎收到SIGKILL。无Python异常或本机cgroup OOM证据；不推断具体终止来源。未完成结果保留在`speed_interrupted_batch3/`，不混入最终统计。
- node12第二次batch=1、显存配额17%的尝试仍以SIGKILL退出，保留于`speed_interrupted_batch1_node12/`。终止前仍有约10GB空闲显存，不据此推断SIGKILL来源。node4重试继续采用batch=1、显存配额17%、max_model_len=32768。减少并发和缓存预算，没有改变帧率、512分辨率、prompt、max_tokens或源样本。此设置不用于与旧批量混合实验比较绝对吞吐。

状态和结果分别为`speed/status.json`、`speed/summary.json`；页面中可刷新查看。进程完成后退出并释放本轮GPU资源。有效结构不代表语义正确，截断／无效输出仍计入耗时并单列。测速产生的新响应只用于记录性能，不替换复核页冻结的50条结果。代码快照与SHA256保存在`code/`和`code_manifest.json`。

## 代码与复现

- `scripts/build_listen_native_fps_review.py`：冻结结果读取、随机抽查和页面发布。
- `scripts/listen_native_fps_review.html`、`scripts/listen_native_fps_review.js`：问题复核、盲化排列、保存和导出。
- `scripts/listen_visual_prompt_review_audio.js`：复用原音轨同步及浏览器内静音配方。
- `scripts/benchmark_listen_native_fps.py`：分开计时、顺序平衡、资源记录和输入一致性校验。
- `scripts/analyze_listen_native_fps_speed.py`：完成后检查配对、重复轮次、实际输入和资源峰值，拒绝用中断结果生成完整对比。
- `scripts/check_listen_native_fps_review.py`：原文完整性、原始窗口、浏览器反馈映射、播放和导出验收。
- `scripts/serve_listen_review.py`：服务入口，复用已有HTTP Range支持。

在项目根目录执行：

```bash
python scripts/build_listen_native_fps_review.py
# 仅在8876尚未运行服务时启动：
python scripts/serve_listen_review.py --port 8876

python scripts/benchmark_listen_native_fps.py --plan-only
# GPU资源确认可叠加后，使用新输出目录；脚本拒绝覆盖既有测速记录。
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
/private/software/conda/envs/vllm/bin/python scripts/benchmark_listen_native_fps.py \
  --gpu 4 --memory .17 --batch-size 1 --max-model-len 32768 --views full --shared-gpu \
  --out experiments/listen_visual_input50_20261029/native_fps_decision/speed_repeat
```

这批定向问题和文本近似样本不能作为全数据集的代表性准确率评测。最终选择需要结合用户复核与测速，不能用条目数量、同句比例或单个正例直接宣布8／16fps胜出。
