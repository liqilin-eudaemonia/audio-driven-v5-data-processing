# 四组最终权重：训练集随机 24 条快速验证

**2026-10-27 已完成：** [结论与完整对照表](report.md)、[机器可读对比](comparison.json)、[指标图](comparison.png)、[盲看页](review/index.html)。192/192条生成成功，AV指标192/192条计算成功；GLARE完整集合因检测覆盖不足不可报告，共同子集5/24条另列。未得到Event改善回应能力的可靠证据；Feature存在UTMOS下降信号，具体边界见报告。

用户明确要求训练集快速验证。固定从四组共同的 30,415 段原训练清单按 sample_id 排序后，用 Python Random(20261027).sample 无放回抽取 24 条，不做类别富集；保留原 split=train。输入和 GT 均使用原训练版本，不替换为后来完成的全量音频静音版本。

四组为 E0=AV、E1=AV＋0.05 Event、E0F=AV＋0.1视频Feature＋0.1音频Feature、E1F=AV＋Feature＋0.05 Event，均为最终第 1,268 步、同训练 seed=20261023、相同 text-attention LoRA 配置。Feature 组的共享 Connector/Event/LoRA 初始化继承原无 Feature 组；新增 decoder 的初始化使整体模型初始化哈希不同。

固定生成 seed=137、251，每组 48 个输出，共 192 个；原生 30 采样步、video CFG=3、audio CFG=7。三机24卡，每GPU同一分片依次执行四组。每次只读取 speaker 音频、listener 首帧、固定任务文本和时长；GT不进入生成。

- `sampling.json`、`observations.json`、`targets.json`：抽样身份、原媒体及类别支持；24条152.261秒，随机样本没有原摇头标签。
- `experiments.json`：四组路径、配置、最终权重哈希。
- `source_restoration.json`：历史 E0/E1 的两处源码从固定备份恢复到独立目录，逐文件SHA必须与原检查点一致；执行实际恢复的训练数据模块。原检查点、原训练目录及当前源码不覆盖，不能跳过源码校验。
- `runner.py`：仅当前进程将历史文件解析到已验证副本；其余源码仍全部验证。`--train-diagnostic` 明确区分训练诊断，正式dev/test隔离检查保持严格。
- `{E0,E1,E0F,E1F}/generation`：生成协议、逐输出检查点/输入哈希、视频和独立listener音频。
- 各组 `glare/`：独立 MediaPipe/FER 检测及 GLARE 公开四指标流程。任一帧漏检则整条失败；完整集合分数不能用成功子集冒充。共同成功子集另报 `glare_common.json`。
- 各组 `omni/`：原 OmniResponse I3D FVD-32、UTMOSv2、SyncNet LSE-D/LSE-C，24个GPU worker不重叠处理行，再按原完整集合聚合，逐项报告共同支持数。

此次只运行现有可直接用于AV的指标，不虚构缺失3DMM特征或文本转写。独立视觉检测分数不是人工真值；UTMOS/LSE不是回应speaker适当性指标；训练集24条、单训练seed和小样本Fréchet均不足以证明泛化或原计划H1。对照计划时将明确列出音频事件时机、误触发、人工盲评、speaker音频替换和独立留出集仍缺的证据。


## HTML 查看（2026-10-27）

- [带配置名称的四列对照](review/comparison.html)：选择24条样本与seed137/251，含speaker输入、四组生成视频/独立音频、真实listener参考及下载入口。
- 在线预览：http://127.0.0.1:8876/p2_train24_fourway/comparison.html ，沿用已有8876服务；远程使用时转发8876端口。
- [隐藏配置盲看](review/index.html)仍保留。HTML、视频及音频HTTP Range请求已核验，视频支持拖动。
