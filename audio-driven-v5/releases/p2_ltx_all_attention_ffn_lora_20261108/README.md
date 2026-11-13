# 完整pipeline全attention／FFN LoRA及现有代码备份

逐文件SHA256核验；[MANIFEST.json](MANIFEST.json)保存原绝对路径、副本相对路径、SHA和字节数。

包含此前完整pipeline、所有消融、原第二阶段代码；另加入独立 `src/p2_ltx_all_lora`、启动入口、最终500h配置、来源指纹、12项测试、32条真实缓存回读和实际模型meta覆盖记录。新增版本保留两路speaker Whisper attention全参数训练，全部原生attention（含gate）及FFN共1,632处rank8 LoRA。扩展版尚未开训。

原第二阶段四机31卡GPU debug已完成，相关配置、step0完整标记和调试验收也已保存；该debug沿用原text LoRA范围，没有使用扩展版。原先备份均保留。

文件位于 `workspace/project_code/...`，恢复时按manifest映射到独立目录并逐文件核对SHA，再使用对应配置和启动入口。不要覆盖运行中的代码或重新开启已取消的训练队列。

模型权重、step0大检查点、训练数据和缓存payload没有复制，仍按配置／完整标记使用原路径。本副本不是离线数据与环境镜像。
