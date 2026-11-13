# 联合Whisper可视化更新

复用原双测试集HTML；两套各14组，7组消融、21项对照。第7组包括联合6000相对AV／Event／Feature／原组合／仅Whisper，同8000步Whisper对照，联合6000→8000，以及历史AV／Event／Feature8927终点参考。不同训练batch、展示数、调度与终点预算明确说明。

页面为每个测试集给出独立文字分析；所有原值／差值来自封存RESULTS.json，未重新推理或改分。Seamless原16条典型样本新增两套权重，共32份视频与精确评分PCM；支持直接切换AV、Event、原组合、仅Whisper和权重进展。ResponseNet仍使用独立随机200的整体结果，历史精选视频不冒充该随机样本。

构建：先用原生环境运行 scripts/publish_p2_combined_review.py --work experiments/p2_combined_whisper_eval_20261106，再运行 scripts/build_p2_metrics_dashboard.py。浏览器验证：/tmp/native-fps-review-qa/bin/python scripts/check_p2_metrics_dashboard.py。

验证覆盖两数据集全部588个指标差值、全部原值、第一／第二名、升降方向、pp、CSV、移动布局、新模型视频与音轨切换。逐个核对新增32份PCM哈希与评分一致。发布中修复新增相对媒体链接，固定使用绝对源路径；最终浏览器检查无错误。
