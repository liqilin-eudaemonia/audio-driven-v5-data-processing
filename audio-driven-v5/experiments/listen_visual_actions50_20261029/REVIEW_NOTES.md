# 新版动作命名prompt：原文对照

这里只核对模型输出，未把文字描述或旧模型结果当作动作真值。

## 动作与表情保留，但多了外观细节

`naturalistic_V03_S1103_I00000150_local_f031c9f810fd`

两组均保留持续微笑、抬手／捂嘴、睁大眼睛和抬眉。新版新增耳饰、衬衫等外观描述，末条仍写appears to be speaking or reacting。

[完整原始响应](results/naturalistic_V03_S1103_I00000150_local_f031c9f810fd.json)

### observation_v1 · full · valid · 3条

| 起止秒 | 原文 |
|---|---|
| 0.012500–3.516000 | The man is smiling broadly with his mouth open, showing his teeth. His head is tilted slightly to his right. |
| 3.516000–4.250000 | The man brings his right hand up to his mouth, covering it partially. His eyes widen and his eyebrows raise. |
| 4.250000–7.226000 | The man lowers his hand from his mouth. His mouth is slightly open, and he appears to be speaking or reacting with a surprised or amused expression. His eyes remain wide. |

### observed_actions_v2 · full · valid · 3条

| 起止秒 | 原文 |
|---|---|
| 0.012500–3.516000 | A bald man with small black ear piercings is smiling broadly with his mouth open, showing his teeth. His head is slightly tilted, and his eyes are crinkled at the corners. He is wearing a dark-colored shirt with a small white button visible at the collar. |
| 3.516000–4.250067 | The man brings his right hand up to his mouth, with his fingers partially covering his lips. His eyes widen, and his eyebrows raise. His head tilts slightly forward. |
| 4.250067–7.226000 | The man lowers his hand from his mouth. His mouth is slightly open, and he appears to be speaking or reacting. His eyes remain wide, and his eyebrows are raised. His head remains tilted slightly forward. |

## 过度合并：多次动作藏在整段描述中

`naturalistic_V03_S0404_I00000037_local_ce8c48a9e093`

新版整段只有一条0.0066–9.822秒的记录，正文分别提到6.379633秒、9.282533秒闭眼，没有为它们单列起止区间。短窗仍可能补充候选；不能据此宣称整条片段漏检。

[完整原始响应](results/naturalistic_V03_S0404_I00000037_local_ce8c48a9e093.json)

### observation_v1 · full · valid · 5条

| 起止秒 | 原文 |
|---|---|
| 1.875133–2.008600 | Eyes briefly close, then reopen. |
| 4.511100–4.644567 | Mouth slightly opens and closes. |
| 6.379633–6.513100 | Eyes briefly close, then reopen. |
| 8.281533–8.381633 | Eyes briefly close, then reopen. |
| 9.282533–9.382633 | Eyes briefly close, then reopen. |

### observed_actions_v2 · full · valid · 1条

| 起止秒 | 原文 |
|---|---|
| 0.006600–9.822000 | A man with dark, curly hair, wearing a white collared shirt, a dark tie, and a dark V-neck sweater with white stripes, is shown in a static medium close-up. His gaze is directed upward and to his right for the majority of the clip. His facial expression is neutral, with his mouth closed and lips slightly parted at times. He blinks occasionally. At 6.379633 seconds, he briefly closes his eyes. At 9.282533 seconds, he briefly closes his eyes again. |

## 仍有重复短区间

`naturalistic_V03_S1032_I00000143_local_92e6424e9032`

两组整段均输出52条，新版仍将同句放在许多相邻短区间，没有解决该样本的碎片化。

[完整原始响应](results/naturalistic_V03_S1032_I00000143_local_92e6424e9032.json)

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

仅列前8条，其余44条见完整响应。

### observed_actions_v2 · full · valid · 52条

| 起止秒 | 原文 |
|---|---|
| 0.025900–0.126000 | Man with glasses and plaid shirt is speaking, mouth open, head slightly tilted up. |
| 0.126000–0.259467 | Man closes his eyes briefly while speaking, then opens them again. |
| 0.259467–0.392933 | Man continues speaking with eyes closed, then opens them again. |
| 0.392933–0.526400 | Man continues speaking with eyes closed, then opens them again. |
| 0.526400–0.626500 | Man continues speaking with eyes closed, then opens them again. |
| 0.626500–0.759967 | Man continues speaking with eyes closed, then opens them again. |
| 0.759967–0.893433 | Man continues speaking with eyes closed, then opens them again. |
| 0.893433–1.026900 | Man continues speaking with eyes closed, then opens them again. |

仅列前8条，其余44条见完整响应。

## 对点头真值的限制

另对S0343前3秒做了原始8fps输入抽帧检查，可见头部角度、转向和目光变化。该检查不足以确定每个完整点头周期及边界，不作为人工GT，不能把早期模型写过nod直接视为正确。抽帧及来源保存在internal_review/；没有改写任何标注。
