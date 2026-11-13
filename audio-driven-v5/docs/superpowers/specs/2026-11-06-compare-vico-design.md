# compare 参考模型 ViCo 测试

用户要求核实现有八项工作的可用性，并实际完成可运行工作的 ViCo test43 推理／评分。已明确确认 DyStream 的 listener 适配：目标自身音频全零，speaker音频进入other分支，同一已有listener首帧，官方process_image(crop=False)，原生512/25→抗混叠256/30评分。也确认Learning2Listen四套人物模型全部适配，单独报告动作结果，不混入写实视频排名。

不更换已封存的联合模型结果。DyStream使用公开官方checkpoint和EMA，官方5步及CFG默认值，随机种子统一137。先1秒最短样本实测，再node8 GPU7四进程共享队列。官方推理每条原生少1帧，保持原始输出，评分按最近时间戳末帧延拓并逐帧记录映射。捕获官方渲染返回的原始RGB，避免额外解码损失后再评分；原生视频仍保存官方结果。DyStream不生成声音，视频带的speaker上下文音轨不能标为listener生成音频。

视频指标沿用前轮官方五项及BFM六组动作补测，复用只读GT／speaker特征缓存。Learning2Listen需DECA50表情＋6姿态，使用独立提取与评测空间，不能把64维BFM截断冒充DECA。四套训练统计、人物VQ-VAE与predictor严格配对；不输入listener未来真值。公开渲染链路缺失则交付动作参数及可解释曲线，不捏造写实视频分数。

另外六项复查官方当前提交、Releases、Issues及本地权重。有缺项则具体记为受阻，给出真实资源链接；不以辅助模型／环境导入冒充完整生成。最终交付各方法43条覆盖、测试协议、来源哈希、独立验收、完整指标、对照可视化和README状态。
