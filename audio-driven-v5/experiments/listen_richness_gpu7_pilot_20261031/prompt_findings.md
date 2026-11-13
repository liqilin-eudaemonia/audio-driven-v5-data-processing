# Prompt试验结论：不能把格式通过当成准确

本轮只对照原始caption与提取结果，没有重新观看全部视频；以下是**文本证据一致性检查**，不是新的人工动作真值，也没有据此改写模型输出。

## 已证明有效的改动

1. **模型选原句ID，代码取原文。** 避免自由复制时改大小写、漏主语或把两个片段拼成非原文。保留完整caption，quote仍是原文子串。
2. **取消逐对时序关系图。** 原caption已经保留顺序。v2出现关系重复／截断；v3/v4将relations固定为空，避免这部分无效生成。
3. **JSON schema约束。** 限制字段、合法句子ID、次数字段结构；不限制开放行为名称。格式错误显著减少。
4. **明确普通静止状态与动作的区别。** v4在稳态注视等案例改善，但仍非完全可靠。

## 具体改善与退化

| 样本简称 | 原文依据 | 对照发现 |
|---|---|---|
| improvised S1979 `2038013807d9` | 固定注视、头部不动、无发声 | v1会把若干“no …”写成positive；v4大体正确保留baseline／negative，不把安静片段当丰富动作 |
| naturalistic S0476 `15fa8c6b5c9a` | 手碰鼻子附近，没有调整眼镜 | v3虚构adjusting glasses；v4消除该错误，但“碰鼻子”的动作信息又不够完整 |
| naturalistic S1003 `f2ca565ac69c` | wearing glasses，没有调整动作 | v3出现adjusting glasses；v4移除这项推断 |
| naturalistic S1788 `c5a3bf7e6ab6` | raises their right hand to adjust their glasses before lowering it | v3保留“抬手／调整眼镜／放手”；v4只剩抬手／放手，漏掉中间实际动作 |
| naturalistic S3373 `e5a4ed2fdefb` | 闭眼后一直保持到结尾 | v3标成blinking；v4用eyes closed/state，更贴近原文 |
| naturalistic S0877 `4d2ff70606d7` | 明确说出Exactly | v3保留speech/audio；v4漏掉音频回应，只保留可见嘴部运动 |
| naturalistic S5232 `c9767b16b310` | two distinct head nods | v3记录2；v4退为多次但不确定 |
| improvised S2055 `729e50c3c598` | remains silent，净化波形全零 | v4仍把silent作为positive audio；导出时被波形一致性检查标记，不能自动入选 |

完整sample ID及各版提取见 [50段对照](review50.md)。

## 为什么不能直接选通过率最高的v4

- v4的全部行为记录中，**explicit次数为0**；原文实际有single、once、twice、two等明确数量。它通过检查的一部分原因是过于保守地全部不填写，而不是次数识别正确。
- v3在47条通过项中有46种正向非baseline名称；v4在50条中增至85种。范围略有不同，不能直接当成多样性增长；检查发现head nodding/nodding/head nod等同义表述更分散，**会虚增稀有类型与丰富度**。
- baseline有所改善，但仍有普通闭唇、静止注视被归入state；也有真实眨眼、抬眉被压入baseline。
- v4的50条导出中，程序自动标记1条音频一致性问题；这不表示其余49条语义全部正确。

## 下一步建议（未在本轮继续执行）

保留“原句编号取证＋结构约束”，不再把动作发现、类型归并、baseline判别和次数理解全部塞进一次回答：

1. 第一遍只列原文明确描述的动作／表情证据，优先覆盖完整；暂不计算总分。
2. 第二遍针对已有证据做同义归并与baseline判别，逐项检查类型是否由原文支持；尤其检查穿戴物品是否被误当作操作动作。
3. 明确次数单独读取，无法确定就保留原文证据和未知；不能为了通过检查把全部次数设为未知。
4. 复用本轮S1788、S0877、S3373、S5232与静音案例作为回归，不扩大样本或全量运行。

这是一项后续候选方案，尚未验证能解决所有问题。本轮到50段试验和结果交付为止，没有全量筛选。
