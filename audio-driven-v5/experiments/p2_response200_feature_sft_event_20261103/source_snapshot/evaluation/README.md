# 评测索引

这里只登记评测侧的入口。Seamless 训练切分仍以项目根目录 [README.md](../README.md) 里的固定 J v1 为准，本目录不替换那套规则。

## ResponseNet

官方 140 条。说话人固定为 Speaker / Human · 观察对象，Agent 自己说话的区间不反向切。音质模型不跑。

140 条里 135 条有观察对象 pair，共 586 条。测评读取方式：[responsenet_speaker_j/HANDOFF.md](responsenet_speaker_j/HANDOFF.md)。处理代码和文件位置：[responsenet_speaker_j/README.md](responsenet_speaker_j/README.md)，逐条清单是 [responsenet_speaker_j/CUTS.md](responsenet_speaker_j/CUTS.md)。

已放上原视频审核页的两条：

- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=6V1HXfo
- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=9lhidn7

## VidChatBench

本地内容验收过的 8 条分轨，不是官方 501 条。说话人固定为 human。官方 ASR 没有词时间，切分用本地已有的词级转写。音质模型不跑。

处理代码和页面：[vidchatbench_speaker_j/README.md](vidchatbench_speaker_j/README.md)。

## ResponseNet固定200条：新两组四权重

使用用户确认的[200条原始清单](../experiments/responsenet_initial200_20261102/references.jsonl)与[CSV时间范围](../experiments/responsenet_initial200_20261102/selection.csv)。原始H＋AV＋Feature、SFT新H＋AV＋0.05 Event各6000/8927步，合计800个输出。四机32卡自动接续当前训练，沿用GPU加速推理及九项评测。[本轮运行、模型对应关系与结果入口](../experiments/p2_response200_feature_sft_event_20261103/README.md)。

新增交叉对照：SFT LoRA Qwen＋原始H Event6000，同200条并与SFT H Event组共享条件。现为5组1000输出；旧权重兼容重放与SSH恢复详见[本轮记录](../experiments/p2_response200_feature_sft_event_20261103/README.md)。
