# 本轮 Qwen SFT hidden 提取并发优化

原32卡已部署历史配置：每卡2模型进程、每进程8个CPU producer、prefetch8、2个异步writer；直接调用原生逐样本Qwen H提取，复用首帧PNG、Gemma、AV和teacher。历史Gemma batch32不属于当前Qwen H路径。

node17 先额外增加每卡1个模型进程，仅2个CPU producer、prefetch8，不停止原进程。基线120秒窗口node17约85.33条/秒，node8/node4同为85.33；热身后的121.67秒窗口node17为96.65条/秒，node8为84.62、node4为85.15、node12为58.32。node17吞吐提升约13%，相对同期对照约14%。这些是不同片段的生产窗口，并非相同样本严格AB；没有修改权重、数值路径、prompt、样本或缓存身份。

依据此结果，扩展到四机每卡3进程，总96个常驻worker。原进程仍各配8个CPU producer，新增进程各配2个，每机共144个CPU producer；所有worker共享动态任务队列。新增supervisor目录node8_extra/node17_extra/node12_extra/node4_extra仍属于原四台机器，不是增加机器。有效配置见 ../effective_execution.json；原execution.json保留初始配置记录。

当前不声称已达到理论最优，也不把显存占用或单点GPU利用率作为提速结论；以生产成功样本吞吐、失败记录和输出一致性为准。

新H最终验收使用独立32进程CPU审计，完成后只停止本运行冗余的串行审计器。成员、来源、身份、payload哈希回执和文件大小检查与原审计相同；已有写盘SHA和逐张量回读检查保持不变。teacher/hidden两种目录各5项正例与篡改拒绝检查通过。
