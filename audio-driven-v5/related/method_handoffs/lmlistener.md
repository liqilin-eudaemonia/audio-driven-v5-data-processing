# LM Listener（ICCV 2023）：运行与交接

更新：2026-11-11。状态：**两份核心预训练权重本地已存在；原生生成接入待完成**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

区别于CVPR2022 L2L的Trevor人物模型。当前本地已有[VQVAE net_iter300000.pth](<../../evaluation/lm-listener/pretrained/VQVAE_l1smooth_c256_p8_datav2/net_iter300000.pth>)与[Trevor net_last.pth](<../../evaluation/lm-listener/pretrained/trevor_hist3/net_last.pth>)，同目录config.json应一起保留。不是“缺checkpoint”。本次尚未验证完整生成；--resume-trans加载模型不自动证明训练优化器/RNG可恢复。

## 官方推理命令和输入参数

[官方README Generation](<../../evaluation/lm-listener/README.md>)，在evaluation/lm-listener目录运行。以下为官方完整参数模板；resume-pth和resume-trans替换为上方实际本地权重，人物与config保持一致：
```bash
python train_t2m_trans.py \
        --exp-name listener_{trevor/conanglobal/stephenglobal} \
        --batch-size 8 \
        --nb-code 256 \
        --drop-out-rate 0.1 \
        --resume-pth output/VQVAE_{trevor/trevorconanstephen}/net_iter300000.pth \
	--vq-name VQVAE_{trevor/trevorconanstephen} \
        --out-dir output \
        --total-iter 0 \
        --lr-scheduler 150000 \
        --lr 0.00005 \
        --dataname face_trevor \
        --down-t 3 \
        --depth 3 \
        --quantizer ema_reset \
        --eval-iter 2000 \
        --pkeep 0.50 \
        --dilation-growth-rate 3 \
        --vq-act relu \
        --max-motion-length 240 \
        --gpt2 gpt2-medium \
        --print_val_pred \
        --gradient_accumulation_steps 2 \
        --manual-bf16 \
        --delay-start-frames 96 \
        --max-time-before 3 \
        --save-name subdir_where_predictions_will_be_saved \
        --seed 50 \
        --resume-trans /path/to/model/checkpoint.pth
```


## 数据处理、缓存及训练／推理适配

官方发布数据为动作tensor和转写；当前Trevor配置GPT2-medium、down_t3、delay96、max_motion240，不要把文本条件等同音频驱动RGB输入。[测试名单/媒体映射与评分器](<../../experiments/lm_listener_readiness_20261109/README.md>)：4107行、4103唯一帧区间、63源视频，半开区间末尾额外tensor帧不计分。缺原始媒体主要阻塞我们的RGB模型在该数据集上的评测，不能据此断言原生LM不能使用已发布tensor/转写。[CPU评分索引加速](<../../compare/deployment/lm_listener_eval_fast.py>)已接入，属于评分器优化；尚无生成加速验收。

## 评测结果与下一步

当前仅有官方评分器GT自对照，无LM模型实测分数。先按本地权重配置打通原生预测，再区分LM官方测试与ViCo43适配；不能把DIM论文Trevor543段与2026发布4107行混为一表。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
