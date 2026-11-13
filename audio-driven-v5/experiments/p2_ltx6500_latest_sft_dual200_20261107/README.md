**最新安排：已按用户要求恢复，仅使用node11物理GPU7。保留已完成163条latent；此前STOP已归档。下方八卡排队说明为历史，当前以[GPU7接续说明](card7/README.md)为准。**

# 接续评测step6500：复用step6000的最新SFT条件

已排队在 `experiments/p2_ltx6000_latest_sft_dual200_20261107/HANDOFF_COMPLETE.json` 成功之后；node11（172.29.189.119）GPU0–7，固定ResponseNet分层随机200＋Seamless固定200，不中断6000组。

- 权重：`experiments/p2_ltx_clip_train24_reserve_node11_20261107/step_006500.pt`，SHA `dc773f0b0e3867d371d65afd7820902022ff3987acf26453cd355ebfae48ab4f`。注意来自三机24卡续训目录，6000步来自旧31卡目录。
- 架构：LTX主干整段Event＋Feature＋speaker Whisper；与6000组相同。
- Qwen：最新SFT2752 final_adapter，规划、H及Gemma均与6000组读取**同一批已完成文件**；无需再跑Qwen或重提H。
- 400条清单、首帧、speaker音轨、Whisper、seed137、Euler30、CFG3／7、GT缓存和指标均相同。模型训练在6026步后换为batch24，此变化记录保留。

入口 `scripts/p2_ltx6500_latest_sft_dual200_20261107.py` 复用6000组已有的数值采样、解码和评分实现；仅切换原生6500权重加载／身份、工作目录与报告步数。动态继承render的唯一文本变更为metadata.step=6000→6500，采样运算不改；队列报告字段也改为6500。`RUNTIME_ADAPTER.json` 记录来源及等价约束，代码快照和SHA在 `code_snapshot/`。当前6000组源文件未修改。

CPU已完成原权重SHA／有限性、推理版逐张量一致性、6500原生代码身份、模型／数据配置与6000一致性、400条条件指向同一文件、相同采样和正确接续依赖检查。见 `CPU_ACCEPTANCE.json` 与 `RUNTIME_ADAPTER.json`。真实GPU验收在自动接续之后进行，不能把排队视为已测完。

状态：`status.json`、`CONTROLLER.json`、`queue.log`。最终以 `FINAL_AUDIT.json`／`HANDOFF_COMPLETE.json` 为准；原值在 `RESULTS.json`，与6000组比较在 `COMPARISON.json`，逐条在 `metrics/`。

生成媒体：`generated/ltx_hidden_event_feature_whisper_6500_sft2752/{responsenet,seamless}/<id>/{generated.mp4,listener.wav}`。

本轮conditions、Whisper和negative通过符号链接共享6000组文件，两个评测都结束前勿删除6000组条件目录。创建本目录STOP会停止领取／后续阶段，不停止训练或6000组；不要重复启动活跃queue。Goal暂停状态保持。
