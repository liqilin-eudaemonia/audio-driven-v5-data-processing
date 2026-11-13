**已全部完成并自动切到5500（2026-11-09 16:14:20 UTC／北京时间11月10日00:14:20）：** 200条生成、解码及视频／音频评分通过最终验收；后续5500在同GPU7严格载入1632目标，7号卡实测利用率100%。见`FINAL_AUDIT.json`、`RESULTS.json`与接续总队列。

**已按最新授权改为node11仅GPU7接续（2026-11-09）：** 只在物理GPU7进行生成、解码和评分；GPU4–6不派发本轮任务。复用已验收step5000推理导出及同200条条件。旧4个分片保留为逻辑评分分片，不代表使用4张卡；实际映射由`RUNTIME_GPU7.json`及`LAUNCH_GPU7.json`限定。旧停止收据仅为历史，已备份到`resume_gpu7/`。入口`scripts/p2_all_lora5000_seamless_gpu7_20261109.py`，进度`status.json`，日志`controller_gpu7.log`及`logs/`。

**已按用户要求停止（2026-11-09）：** 本轮step5000的node11 GPU4–7四个评测worker均已退出；STOP保留，禁止自动续跑。现存vLLM未动，仍约28GB／卡；6428的28卡评测继续。见`STOP_ACCEPTED.json`。

# 全attention／FFN LoRA完整组5000：node11 GPU4–7，Seamless200

2026-11-09用户明确追加使用node11后4卡评测同pipeline的step5000。本轮仅GPU4、5、6、7；不改变6428的四机28卡评测，现存vLLM保留并叠加。

完整训练权重：`../p2_ltx_all_attention_ffn_lora31_20261108/step_005000.pt`；SHA256 `f02a6cdd6889688826183e4e918afe51875298feaea4af4116489a8ac2566424`。仍含AV＋整段Event＋双路Feature＋speaker Whisper、原生48层1632处rank8 LoRA。

与6000／6428严格相同的Seamless固定200、SFT2752规划/H/Gemma、speaker Whisper、listener首帧、seed137、Euler30、CFG3／7、512及原FPS／帧数；不重新规划、不读listener未来GT为输入。独立目录保存输出，未修改其他评测。

完整模型／Adam／31RNG及原282源码先校验，模型推理导出逐张量一致；每卡严格载入1632目标。沿用共享领取长片优先、只读mmap、资源准入、每卡4路解码、就绪分片评分和完整200特征合并。生成／解码PYTHONPATH清空，仅指标使用vendor环境。

代码入口：`scripts/p2_all_lora5000_seamless_eval4_20261109.py prepare|queue|worker`；适配`src/p2_all_lora5000_eval_20261109.py`。逻辑rank0–3映射物理GPU4–7，`processes/`与`devices/`记录实际物理卡号。

当前进度`status.json`；真实启动验收`RUNNING4_ACCEPTED.json`（生成后）；媒体`generated/ltx_all_lora_event_feature_whisper_5000_sft2752/seamless/<id>/generated.mp4`及`listener.wav`。指标`RESULTS.json`，对照已完成6000结果`COMPARISON.json`。须通过最终全200绑定对照与`FINAL_AUDIT.json`、`HANDOFF_COMPLETE.json`才算完成。

不启动训练，不添加Goal或定时聊天监控。根目录STOP只停止本轮。
