# 原文对照例子

这里只检查模型写出了什么，不把原文当作视频真值。S1103的早段露齿微笑另在上轮相同输入的抽帧中检查过。

## 持续微笑与手部动作保留

`naturalistic_V03_S1103_I00000150_local_f031c9f810fd`

新版整段保留0.0125–3.516秒的露齿微笑和后续抬手／捂嘴。相较上轮简单追加禁令的整段输出，这是正面变化；末条仍含appears to be speaking or reacting及amused等解释。

[完整原始响应](results/naturalistic_V03_S1103_I00000150_local_f031c9f810fd.json)

### baseline · full · valid · 3条

| 起止秒 | 原文 |
|---|---|
| 0.012500–3.516000 | The listener is laughing heartily, with a wide smile and crinkled eyes, indicating amusement or enjoyment of what is being said. |
| 3.516000–4.250000 | The listener brings their hand to their mouth, a gesture often associated with surprise, contemplation, or trying to suppress laughter or a reaction. |
| 4.250000–7.152967 | The listener's expression shifts to one of exaggerated surprise or disbelief, with wide eyes and a slightly open mouth, suggesting a reaction to something unexpected or humorous. |

### observation_v1 · full · valid · 3条

| 起止秒 | 原文 |
|---|---|
| 0.012500–3.516000 | The man is smiling broadly with his mouth open, showing his teeth. His head is tilted slightly to his right. |
| 3.516000–4.250000 | The man brings his right hand up to his mouth, covering it partially. His eyes widen and his eyebrows raise. |
| 4.250000–7.226000 | The man lowers his hand from his mouth. His mouth is slightly open, and he appears to be speaking or reacting with a surprised or amused expression. His eyes remain wide. |

## 仍有心理推演

`naturalistic_V03_S2147_I00000023_local_e3c42e2beb75`

新版整段仍有suggesting a moment of reflection or listening、indicating continued listening or thought等表述，直接违反物理观察约束。

[完整原始响应](results/naturalistic_V03_S2147_I00000023_local_e3c42e2beb75.json)

### observation_v1 · full · valid · 7条

| 起止秒 | 原文 |
|---|---|
| 0.018067–0.251633 | The person's mouth is slightly open, and their eyes are partially closed, suggesting they are speaking or pausing mid-sentence. |
| 0.251633–0.652033 | The person's eyes open wider, and their lips are slightly parted, indicating a more attentive or listening state. |
| 0.652033–1.152533 | The person's eyes close briefly, then reopen, and their lips remain slightly parted, suggesting a moment of reflection or listening. |
| 1.152533–1.753133 | The person's eyes remain mostly open, and their lips are slightly parted, indicating continued listening or thought. |
| 1.753133–2.520567 | The person's mouth opens wider, and their eyes are open, suggesting they are speaking or about to speak. |
| 2.520567–3.521567 | The person's mouth remains open, and their eyes are open, indicating continued speaking or talking. |
| 3.521567–4.756133 | The person's mouth opens and closes, and their eyes open and close, indicating ongoing speech with natural pauses and emphasis. |

## 相邻短区间重复

`naturalistic_V03_S1032_I00000143_local_92e6424e9032`

6.572秒片段被新版整段输出分成52条，交替出现两句嘴部描述，每句26次。这是碎片化／重复待核对的明确案例；仅凭同文不能断言真实动作次数。

[完整原始响应](results/naturalistic_V03_S1032_I00000143_local_92e6424e9032.json)

### baseline · full · valid · 27条

| 起止秒 | 原文 |
|---|---|
| 0.025900–0.126000 | Nodding head slightly while speaking, indicating active listening and engagement. |
| 0.259467–0.392933 | Maintaining steady eye contact and a slight smile, signaling attentiveness and positive engagement. |
| 0.526400–0.626500 | Leaning slightly forward, a gesture that conveys interest and focus on the speaker. |
| 0.759967–0.893433 | Nodding head in agreement or understanding while the speaker is talking. |
| 1.026900–1.127000 | Maintaining eye contact and a neutral expression, indicating attentive listening. |
| 1.260467–1.393933 | Slight head tilt, suggesting curiosity or thoughtful consideration of what is being said. |
| 1.527400–1.627500 | Maintaining steady eye contact and a neutral expression, indicating continued attentiveness. |
| 1.760967–1.894433 | Nodding head slightly, reinforcing understanding or agreement. |
| 2.027900–2.128000 | Maintaining eye contact and a neutral expression, indicating focused listening. |
| 2.261467–2.394933 | Slight head tilt, suggesting thoughtful consideration of the speaker's words. |

仅展示前10条，其余17条见完整响应。

### observation_v1 · full · valid · 52条

| 起止秒 | 原文 |
|---|---|
| 0.025900–0.126000 | Man's mouth opens and closes as he speaks. |
| 0.126000–0.259467 | Man's mouth closes and remains still. |
| 0.259467–0.392933 | Man's mouth opens and closes as he speaks. |
| 0.392933–0.526400 | Man's mouth closes and remains still. |
| 0.526400–0.626500 | Man's mouth opens and closes as he speaks. |
| 0.626500–0.759967 | Man's mouth closes and remains still. |
| 0.759967–0.893433 | Man's mouth opens and closes as he speaks. |
| 0.893433–1.026900 | Man's mouth closes and remains still. |
| 1.026900–1.127000 | Man's mouth opens and closes as he speaks. |
| 1.127000–1.260467 | Man's mouth closes and remains still. |

仅展示前10条，其余42条见完整响应。
