# vLLM并存条件下再次验证单／八网卡

用户明确要求保留vLLM并叠加。生产训练完整step871模型／Adam／31份RNG保持，node12/node8/node4各GPU0–7、node11 GPU1–7，GPU0评测不变。

## 已执行

测试脚本为8+8+8+7显式绑定本地CUDA设备。单网卡控制先行，31rank FP64 SUM在1／32／512MiB均通过。随后八网卡分别测试NCCL_CROSS_NIC=0和1，均通信超时，没有进入完整模型ABBA测速，不能宣称端到端提速／降速。

测试允许node11既有七个vLLM EngineCore／api_server共存，所有其他新占卡进程会阻止测试；resource_ownership.jsonl保留逐次检查。该条件不是独占资源测试，结论限于当前并存环境与两套通信参数，不能推断所有多网卡方案都不可用。

## 正式训练恢复

使用原单网卡HCA设置、原始训练源代码／数据／损失／batch31从step871严格恢复。node11每rank分配器上限108GiB，其余节点134GiB；实际命令见PRODUCTION_COMMANDS.json。vLLM保持，未终止其进程。RESUMED_ACCEPTED.json表示已确认恢复后至少8步真实更新；OVERLAY_RUNTIME_CHECK.json记录node11实际环境和显存。

恢复入口为`driver.py resume`（仅在确认现有训练已退出后使用）；它复用原冻结训练入口、严格checkpoint检查，并对node11添加108GiB进程上限，在核验仅有已授权vLLM后台进程后调用原集群launcher。普通旧launch的空卡检查仍保留。

测试的同批ABBA入口与八网卡采用wrapper均已准备，但本轮在通信门槛处拒绝候选，未改生产cluster、config或282核心源码。先维持稳定训练，网络根因仍需另行定位。
