# AV＋Feature 再次复核（2026-11-07）

本次复核对象是最终500h的Qwen-only AV＋Feature消融。正式结论以RECHECK_ACCEPTED.json及下面的全量审计成功记录为准。

## 代码

- 实际训练由p2_current500h_av_feature.py进入p2_current_av_feature.training.train。
- Dataset继承最新Qwen/Gemma/AV读取器，并按同一个sample_id加载已验收的混合teacher。teacher几何与该样本layout一致；CPU预取线程独立打开读取器。
- 配置保留旧data_config以约束原有架构；实际训练extractor配置来自最新AV缓存protocol，实际样本由current_cache_index与sample_list共同决定。旧train-only的training_admitted标记不参与筛选。
- 复用原MatrixModel前向：同一次Connector输出U送入两路Feature Decoder与LTX。总损失严格为AV＋0.025视频Feature＋0.025音频Feature。按microbatch和world size归一化，再进行SUM梯度归约。
- Event头冻结且不执行，不读取Event标签；Qwen参数冻结，Whisper分支不存在。优化器仅包含Connector、text LoRA和两路Decoder。
- 完整恢复核对model、Adam和rank RNG；禁止继承其他实验checkpoint。样本/noise ordinal顺序与AV-only31一致。Feature Decoder的随机去噪也消耗随机数，因此不能宣称两个不同loss实验的所有实际噪声张量逐值相同。

## 本次修正

31卡启动分支原先未传递--stop-after。现通过独立环境变量传给所有rank，并验证rank入口确实使用该限制；不带限制的正式命令会显式清除继承值。跨实验resume与非法停止步数在生成命令前拒绝。

最长样本启动预检增加完整AV＋Feature损失及全部梯度的有限值检查。此前只显式检查AV值，当前完整损失反传后也会拒绝非有限梯度。

这些修改只涉及尚未启动的AV＋Feature代码，当前AV-only运行所冻结的源码SHA保持一致。

## 验证证据

- recheck_tests.xml：35项测试通过，包括原生小尺寸LTX更新、Feature单独反传到Connector、原初始化一致、完整断点恢复、分布式归一化和31卡停止参数传递。
- preflight.json：32条真实缓存读取，覆盖train/dev以及teacher的new/reuse_audio/reuse_both三类；完整caption、AV与teacher几何检查通过。
- RECHECK_FULL.json：逐条核对284,218份最终caption实际文件SHA和内容，与Qwen/Gemma缓存完整标记合同一致；核对所有成员顺序、划分、小时数和最终selection SHA。
- TEACHER_RECHECK_FULL.json：逐条核对最终媒体身份、全部teacher复用/增量映射、缓存合同及payload存在；旧复用payload还核对文件大小和历史SHA记录。
- PREFETCH_VERIFIED.json：实际CPU预取线程读取train、dev及teacher。
- CURRENT_AV_SOURCE_UNCHANGED.json：当前AV-only冻结源码未变。

本轮没有重新读取全部TB级tensor计算SHA；tensor内容完整性由已有全量验收及运行时load_part的payload SHA检查保障。本实验尚未进行31卡GPU最长样本和真实分布式更新验收，不能据此宣称GPU显存/长期稳定性已验证。当前已运行的AV-only未被停止。
