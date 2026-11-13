# 当前官方权重与本地运行权重核验

README下载脚本的两个Google Drive文件已重新下载，内容SHA与本地实际使用文件逐字节一致，没有发现另一套可公开获取的更新权重。

| 官方文件 | 大小 | 本地历史别名 | SHA256 |
|---|---:|---|---|
| motion_autoencoder.pth | 181034955 | flow_transformer.pth | 41391516d761a3768bf43ad4ce10f41c66b3a652394610c1be2bc9617f327c33 |
| flow_transformer.pth | 613993979 | motion_autoencoder.pth | 1c258fdcc534941ef2cf46a53188690305584ca642099c523db08122e05083cc |

本地历史文件名标反，但官方加载器分别遍历模型参数名，不按文件名决定网络子模块。两个包没有交集：小包212个自动编码器状态，大包152个音频投影／FlowTransformer状态；原生运行327个核心参数逐张量相等。因此加载顺序没有覆盖冲突，当前结果不因这个名称问题改变。封存运行依赖原路径和SHA，本次不交换正在使用的文件。

同时核查了checkpoint中两个50×50注意力mask：它们与公开代码构造的allow-mask恰好逐元素互补。运行代码采用SDPA并重建allow-mask；源文件中的mask没有被按参数加载。不能直接以原始buffer不相等为由覆盖运行mask。该检查不改生成端。

[官方文件元信息](PUBLIC_FILE_METADATA.json) · [重新下载后的哈希比较](FRESH_OFFICIAL_CHECKPOINT_COMPARISON.json) · [参数集合交集检查](WEIGHT_LABEL_OVERLAP_AUDIT.json) · [mask表示检查](MASK_BUFFER_SEMANTICS.json) · [结论](SOURCE_CLOSURE.json)。

这些证据排除了“当前官方入口提供的是另一份权重字节”及“标反文件导致参数覆盖”两种原因；文件中没有公开训练步数／DPO运行回执，仍不能单独证明这套checkpoint就是Table4的确切版本。
