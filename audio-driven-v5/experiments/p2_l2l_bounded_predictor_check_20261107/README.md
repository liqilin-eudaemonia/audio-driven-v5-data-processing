# 有上限的L2L predictor可行性验证

**最新状态：已停止于569步。** 用户要求先按论文和官方材料核对协议。短预热无复现依据，本目录只保留诊断；STOP已写入、完整last保存、EXITED记录退出，不自动继续、不计正式基线。

原VQ已在27749步完整保存并停止。此处复用原始初始化／原官方优化器训练得到的验证最优1000步VQ，不采用码本初始化短实验的权重。VQ满足原先的学习门槛：总目标优于初始化、重建优于训练均值5%以上、使用至少2个码。取消未经充分依据的最少150000步要求。

仅检查predictor训练和46条source-only验证rollout，不跑test43、不启动额外种子。本地训练适配：保留官方网络、损失、Adam／ScheduledOptim和batch32，仅将predictor warmup由100000改为1000；最大10000更新，验证每500，3000后6次无改善停止，总壁钟上限20分钟。必须优于初始化并比train-token Laplace unigram CE改善1%以上，才进入source-only验证rollout；运动有限值／方差检查沿用原明确口径。

这是诊断结果，不是DIM原表复现或已经完成的正式基线。短预热差异在PROTOCOL及检查点identity中保存。

进度见status.json；成功见RESULTS.json；学习失败见seed_137/predictor/FAILED_GATE.json和FAILED.json；无论成功失败都在EXITED.json记录退出。GPU7继续与已有推理共享，上限12GiB。
