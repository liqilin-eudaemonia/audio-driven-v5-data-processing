# 同一SPECTRE系数上的DIM完整公开评分代码

43条，8项中0项与AF Table4两位小数一致。主AF文字重建结果仍单独保留，未按逐格接近程度改公式。

| 指标 | 分组 | 实测 | AF Table4 | 一致 |
|---|---|---:|---:|---|
| FD | exp | 22.979871 | 16.64 | False |
| FD | pose | 0.089712 | 0.05 | False |
| rPCC | exp | 0.057621 | 0.01 | False |
| rPCC | pose | 0.019046 | 0.01 | False |
| SID | exp | 4.227788 | 3.12 | False |
| SID | pose | 3.706947 | 3.00 | False |
| Var | exp | 1.720771 | 2.80 | False |
| Var | pose | 0.021519 | 0.03 | False |

这是同一43条原始视野生成视频／SPECTRE系数上的DIM完整数学实现。K40／20及拉平方差与AF附录K15／9、逐系数方差描述存在冲突，不能把它称作已取得AF私有评分代码；也不等于完成DIM的EMOCA十二项复现。
