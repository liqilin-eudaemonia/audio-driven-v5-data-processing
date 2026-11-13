# 当前 seed238 扩至三机24卡（已授权，尚未激活）

用户明确要求第二个 seed238 使用 node12、node8、node11 各8卡；不是后续完整测试集。保持四组×256条、随机种子238、全部条件和原生生成/评分代码，已完成产物继续复用。

当前两机16卡继续执行。三机候选配置和入口在本目录的 `*.proposed` 中，尚未覆盖运行入口。唯一阻塞：node11 172.29.189.119:22 能连接，8卡显存和计算有占用，但 NVML / nvidia-smi 计算进程列表为空；当前 PID 命名空间的存活进程和 GPU 文件描述符检查均未发现计算任务。没有终止不明进程或重置GPU。已向用户询问占卡任务的启动环境或新SSH入口。

诊断保留在上一级并列的 `p2_joint_qwen_ltx_test256_two_seeds_20261113/NODE11_*`。实际两机状态仍以上级 `status.json` 为准，不能将准备好的24卡配置表述成已启动。

取得对应环境后：记录并停止已授权的 node11 GPU任务及恢复器，确认8卡释放；快照现有运行入口、配置、协议、PREPARED及状态；先停止双seed followup，再确认当前controller和其worker退出；部署本目录候选入口/配置/prepare_impl，只更新执行资源字段与入口指纹；保留数据、模型、seed、条件、完成收据和评分；重算PREPARED指纹；重新启动双seed followup，核验三机24卡实际任务和node11首个产出。两seed自动均值汇总保持。

CPU检查：仅 slots、queue、activate 三个函数改变；worker、publish及其原生数学计算逐AST相同；原16个rank顺序不变；当前全部冻结文件校验通过。生成最多20并发，余下4卡动态解码/评分。详见 CPU_ACCEPTANCE.json。
