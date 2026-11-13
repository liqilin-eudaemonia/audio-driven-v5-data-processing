# AV＋Whisper代码再次复核

结论：未发现需要修改的训练逻辑错误，生产代码保持不变。最终验收见RECHECK_ACCEPTED.json。

- 36项CPU测试通过，包括原生小尺寸LTX、多步真实更新、全重计算梯度、完整模型／Adam／RNG恢复、初始化／原实现前向一致、时间RoPE／mask／CFG路径。
- 新增两进程CPU Gloo测试，使用实际StreamingFP64Reduction和实际Whisper模型，完成两次更新；与单进程同输入平均梯度参考相比，两rank参数及Adam状态最大绝对误差均为0。
- 新实验经过更新后，改变有效speaker特征会影响视频和音频输出；修改masked token不影响输出，证明条件实际接入且mask生效。
- 新冷进程走实际CLI train，配置、源码、数据身份校验全部通过，准确停在“需要GPU”的入口，无导入或身份错误，不创建训练运行状态。
- 参数所有权覆盖Connector、text LoRA、Whisper adapter和注意力分支；Event头冻结且不执行，Feature Decoder未构建，Qwen／Whisper encoder冻结。
- 损失严格为视频AV＋音频AV，辅助损失为0；world归一化与SUM归约匹配，31卡rank布局及GPU7排除／短测停止参数均通过。
- 最终284,218段train＋dev、最新caption条件缓存及Whisper验收收据与已完成全量核验完全一致。此前32条实际读取、CPU预取、284,218条speaker身份／区间／缓存合同审计仍有效。
- 当前AV＋Feature训练所冻结源码逐文件SHA未变。

双进程测试首次使用了不同的随机测试输入，导致对照不一致；固定fixture种子并逐值验证Qwen条件、speaker特征及视频latent相同后通过。该问题属于新增测试夹具，不是训练数据或归约实现问题，没有因此调整训练实现或放宽比较容差。

本次验证全部使用CPU，未占用训练卡；尚未进行本实验H200／NCCL的31卡最长样本显存及真实更新验收，长期稳定性仍需正式启动监控。不能将CPU Gloo验证等同于GPU验收。
