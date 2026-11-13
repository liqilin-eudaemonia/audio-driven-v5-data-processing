# 当前完整损失的四分之三重计算测试

用户授权：测试稳定且有提速可启用，接受接近134GiB进程上限。

保持当前500h train+dev、Qwen原始缓存、AV+LTX hidden整段Event+双路Feature+speaker Whisper、global batch32、每300步保存、样本和noise ordinal。

原训练：`../p2_ltx_clip_train32_after_gpu67_20261106`。候选生产：`../p2_ltx_clip_recompute32_20261106`。

ABBA每轮同一完整模型/Adam开始，最长11.95秒样本两次真实更新，一次预热，12个相同计时batch全部真实更新；测试更新不计入生产。比较完整梯度、损失、模型/Adam状态和所有rank显存峰值。两轮候选均安全、逐值一致且各自快于对应基线才迁移。原检查点保留，失败自动恢复all。

CPU四block完整联合损失测试通过，两次更新逐值一致，第四block前向次数验证确实减少重算。

`state.json`看控制器，`progress.json`看测试阶段，`partial_results.json`看逐轮结果，`RESULT.json`为最终决策。
