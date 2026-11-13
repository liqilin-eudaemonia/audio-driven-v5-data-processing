# AV + Event 首轮对照实现

按用户已选方案实施：E0 的 Connector 仅由 AV loss 更新，事件头读取 detach(U)；E1 由 AV + 0.05 Event 更新。同样保留弱背景标签、七类接口、确定性计算、FP64 梯度累加及配对数据/噪声规则。

使用 training.feature_loss_enabled=false 显式关闭 feature 分支；旧配置缺省为 true。关闭时复用 Extractor(auxiliary=False)，不实例化特征 decoder，不读取 teacher 缓存，不计算特征损失，不注册 features 优化器组。推理和恢复按该配置构建相同架构。

保持现有缓存身份与抽取源码不变：新增训练读取适配层；cache --part all 根据训练模式选择依赖。旧 teacher 缓存和当前全量抽取任务保留。新实验独立配置与 output，不能把旧完整模型 checkpoint 直接续训为 AV-only。

验证：配置/模块/优化器/无 teacher 读取、E0/E1 梯度、严格恢复和推理构建测试；完整 CPU 回归；两进程 CPU DDP 对照；50 条真实缓存的无 teacher 读取检查。若机器有足够空闲 GPU，再运行新的真实 S0。
