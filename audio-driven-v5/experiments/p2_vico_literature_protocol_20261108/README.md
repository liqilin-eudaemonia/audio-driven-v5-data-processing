# ViCo文献协议对齐与EMOCA实测

**执行完成：固定43条，172组EMOCA特征，两组我们模型评分。** 原论文未公开的执行差异已披露，尚不能称严格复现原表。

[结果与解释](RESULTS.md) · [逐项协议对照](ALIGNMENT.md) · [完整JSON](RESULTS.json) · [论文表格PDF](paper_tables/preview.pdf) · [全部数值CSV](paper_tables/all_metrics.csv)

[测试清单与audio-key差异](TEST_MANIFEST.json) · [文献结果来源](LITERATURE_ROWS.json) · [固定协议](protocol.json) · [执行设计](DESIGN.md)

[官方权重](DOWNLOADS.json) · [编码器严格加载](ENCODER.json) · [真实前向对照](ENCODER_PARITY.json) · [解码与batch对照](MEDIA_PARITY.json) · [官方指标全43条核验](EMOCA_METRIC_PARITY.json) · [GT自比较](GT_SELF_CHECK.json) · [GT统计锚点](GT_ANCHOR.json) · [最终验收](FINAL_AUDIT.json)

复算：项目根目录执行 `env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' compare/.envs/vico_more/bin/python compare/deployment/vico_literature_report.py`。正式封存后先验证COMPLETE.json，不覆盖旧结果。

代码：[清单与来源](../../compare/deployment/vico_literature_protocol.py) · [EMOCA提取](../../compare/deployment/vico_literature_emoca.py) · [原指标适配](../../compare/deployment/vico_literature_metrics.py) · [评分与导出](../../compare/deployment/vico_literature_report.py) · [接续与验收](../../compare/deployment/vico_literature_finish.py)。

新发现的ReactFace用户资源见[来源记录](REACTFACE_USER_RESOURCES.json)，不替代EMOCA，不声称已取得其生成器。
