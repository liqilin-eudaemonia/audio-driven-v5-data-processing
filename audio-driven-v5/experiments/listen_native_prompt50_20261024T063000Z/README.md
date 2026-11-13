# Qwen3-Omni 原生视听逐次事件 prompt 对照

本轮已完成 50 段 × 原版/新版两种 prompt，共 100 请求；结构通过 100/100，但效果未通过人工事件验收。结论见 [分析](analysis.txt)，逐样本结果见 `results/items/`，网页为 `data_previews/listen_benchmark_review/native_prompt/`。

## 授权与资源

用户明确要求清理 node8 计算任务并用 8 卡运行。提供的旧地址 `172.29.104.194` 连接超时；交接记录和历史打标部署确认现址为 `172.29.104.220`，hostname 为 `dedicated-developjob-dev-machine-hi3wi`。清理了旧 vLLM 服务及 InfiniteTalk 计算任务树，保留基础服务；清理记录复制保存为 `node8_cleanup.json`。未停止 node17/node12/node4 的三机训练。

`launch_node8.py` 启动 8 个独立模型副本，逐样本请求加锁分配。每个副本同时执行一个原生视听请求，保持已有修复的单请求预填充路径；没有为增加并发而使用已知存在时序问题的混合模态批处理。

监测窗口约 248 秒，包含模型加载和推理；显存峰值约 123–124 GiB/卡。CPU 与 GPU 实测记录保存在 `resources.jsonl`，不宣称 GPU 持续满载。单请求自回归生成阶段利用率较低，8 卡占用期间的采样均值约 10.4%，短时预填充峰值可达 100%。全部进程正常退出，结束后每卡显存约 1 MiB。

## 对照与可追溯性

- `human_feedback.json`：用户原始比较意见，`human_ground_truth=false`，没有转为完整人工标签，也没有发送给模型。
- `manifest.json`：原 50 条开发样本，用户反馈的 4 条优先，其余固定种子排序。不是独立测试集。
- `config.json`：模型、采样、生成参数和两种 prompt 身份。
- `prompts/`：每个样本两份实际 prompt。类型保持开放，要求每次可分辨的短词和点头周期独立列出，不能制造精确边界。
- `code/`、`code_hashes.json`：推理代码快照和校验值；`report_code/` 为报告代码快照。
- `results/items/<sample>/<system>/requests/<sample>/discovery/`：实际请求、输入证据、原始回答与校验结果。
- `status.json`、`launch.json`、`resources_before.json`、`resources_after.json`：任务完成状态、各卡命令和释放情况。
- `comparison_stats.json`：输出统计；视觉/音频条目数不是正确事件数量。

此前准备的 8 段小样本包保留在 `../listen_native_prompt_pilot_20261024T062202Z/`，完成了 8 段媒体准备和 23 项离线回归，但没有单独执行 GPU 推理。本次授权 8 卡后扩展成当前 50 段实验。

页面提供原 prompt、新 prompt 同步媒体比较、可选 Qwen3-VL 历史参考、原始音轨能量图与模型区间叠加、原始回答下载及本轮人工反馈导出。音频能量辅助定位，不直接判定词语或说话人；没有把能量阈值生成的区间写回标签。
