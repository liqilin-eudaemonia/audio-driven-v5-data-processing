**最新状态：6、7卡连续两次确认空闲后，已自动触发在GPU7从step43接续。有效batch32与每50步保存不变。当前等待/启动状态见AUTO_RESUME_STATUS.json，实际训练进度见status.json。**

**当前状态：按用户最新要求，GPU7续训任务已停止。第43步完整checkpoint已保留，STOP标记已恢复。GPU6原训练此前已正常退出。**

**已确认实际更新：** 第1次参数更新完成，loss及五组梯度均正常，单卡累积32条样本的实测峰值显存115.5 GiB。首个周期checkpoint将在step50保存；后台任务持续运行。见`STARTED.json`。

# node11 GPU6 单卡正式训练

本轮使用 node11 / js-dev / 172.29.189.119 的物理 GPU6。CUDA_VISIBLE_DEVICES=6，进程内设备为 cuda:0；WORLD_SIZE=1、RANK=0。没有使用其他GPU。

沿用已验收的当前 train+dev 284,218段、500.001259h、修正后的整段Event标签和五项缓存。模型为当前 AV + LTX hidden整段Event + 两路Feature + speaker Whisper，冻结原始Qwen hidden条件。新训练从初始化开始，不接续16步联合测试权重。

有效global batch仍为32，单卡每次处理1条、累积32条后更新。共8,882次更新，沿用原采样计划、学习率和warmup。每50个optimizer step保存完整checkpoint：step_000050.pt、step_000100.pt等，并生成complete.json哈希凭据。step_000000.pt为初始状态；正常结束或通过STOP停止也保存最终状态。完整文件包含模型、Adam状态、配置身份与随机状态。

控制进程独立后台运行，见CONTROLLER.json；控制日志controller.log；GPU进程日志launch/node11.log；进度status.json；逐步指标metrics.jsonl。首次进入计算前执行全量preflight与最长样本检查。

本轮单卡使用已有FP64梯度累积路径，保持有效batch；每50步指参数更新次数，等于常规完整批次1,600次样本呈现，不是50条样本。

需要正常停止时在本目录创建STOP，训练在当前更新结束后保存退出。不要直接终止进程以免丢失尚未保存的更新。续训需要删除STOP并通过同一config/cluster、显式--resume指向本目录已有完整checkpoint。

源码与配置影响严格接续身份；本轮已保存before/修改前副本及tests_final.xml回归结果。后续修改公共训练源码前应保存本轮可接续代码版本。
