# 最终500h原始Qwen AV-only：三机24卡备选

已准备，未启动。协议、备份、完整CPU验收和数据说明见[主实验说明](../p2_current500h_av_only_20261107/README.md)。

只用Qwen条件，不加Whisper/Event/Feature；最终train281060＋dev3158共284218段，使用最新caption对应的H/Gemma和混合AV VAE缓存，从原始预训练初始化开始。node12/node8/node4各8卡，batch24、11843步、尾部14条填充，每500步保存、all全重计算、CPU预取2、512MiB FP64归约、GPU Adam。全部真实sample/noise配对与32卡一致，duration384重组批次，不能认为优化轨迹不变。

[配置](config.json) · [数据准入与回读复用](preflight.json) · [调度和准备验收](PREPARED.json) · [完整验收](ACCEPTANCE.json) · [命令预览](COMMANDS.json)。用户分配机器后才显式启动：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_current500h_av_only.py launch --output experiments/p2_current500h_av_only24_20261107 --logs experiments/p2_current500h_av_only24_20261107/launch
```
