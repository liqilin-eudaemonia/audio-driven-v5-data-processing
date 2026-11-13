# Qwen3-Omni 原生视听：逐次事件提示词小样本对照

状态：已准备，尚未启动 GPU 推理。2026-10-24 资源检查显示 node17、node12、node4 的 24 张 H200 均被现有训练占用。本实验未停止或调整这些训练。

用户提供的 4 条比较反馈完整保存为 `human_feedback.json`。它们不是完整的逐事件人工真值，也不作为模型输入。另从原 50 条开发样本中以固定种子抽取 4 条，共 8 段。此集合用于诊断，不是独立测试集。

## 对照设置

同一 Qwen3-Omni-Instruct 权重、listener 原框裁剪 512×512、8 fps 原生视频、独立 listener WAV、修正后的 0.25 秒时序网格、temperature=0、4096 输出 token、同一 JSON schema。只运行 discovery，共 16 次请求：

- `native_baseline`：原版开放描述 prompt，重新运行作为同期基线。
- `native_occurrences_v1`：在同一输出契约下增加逐次事件、独立视听边界、完整持续状态和不确定性要求。

新版具体要求：每次可分辨的短词分别输出；可分辨的点头周期各有起止时间；点头与声音是独立事件；持续微笑不切成重复定长片段；连续笑声按实际笑声段标记；无法可靠细分时说明不确定，禁止编造均匀间隔和精确时间。类型保持开放。

没有向模型提供用户针对具体样本的判断、约 5 秒微笑等答案或旧模型结果。新版 prompt 中的行为、短词示例只是通用检查要求。

逐样本的完整提示词位于 `prompts/`；配置、输入清单和抽样信息分别为 `config.json`、`manifest.json`、`plan.json`。代码已冻结到 `code/`，校验值为 `code_hashes.json`。旧实验及生产提示词保持原样。

## 启动

在有空闲 H200 且可访问上述共享目录的节点上运行。下面 `--gpu 0` 必须换成实际分配给本次实验的卡号。单张卡共享一个模型实例，顺序完成新旧提示词请求；原生视听保持单请求预填充以避免已知混合批次时序问题。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  experiments/listen_native_prompt_pilot_20261024T062202Z/code/scripts/run_listen_prompt_pilot.py \
  --run experiments/listen_native_prompt_pilot_20261024T062202Z --gpu 0
```

增加 `--check-only` 只检查资源，不加载模型。启动器检查所选 GPU 是否有计算进程、利用率和剩余显存；资源不足时返回，不终止任何已有任务。当前环境的该检查已确认拒绝在忙卡启动。

实际推理时 `process.json` 记录进程与命令，`resources.jsonl` 记录 GPU 资源和系统 load average，`results/items/` 保存各样本的请求、实际输入、原始回答、校验结果。失败和截断回答保留原样，不补写标签。

## 验证与后续判断

`validation.json` 保存 8 段媒体解码、原文件 hash、原生视频采样时间检查和冻结代码离线测试结果；这些检查不代表已完成模型推理。`status.json` 表示实际运行状态。

收到模型回答后，优先复核用户指出的漏点头、每次 okay/yeah 的独立区间及微笑范围，再查看随机样本是否增加误报。统计整段区间、重复片段和格式失败作辅助诊断，不以事件更多或时间更短判定成功。尚不能认定单一原生视听模型是最终最佳方案。
