# 前10／后10排序效果检查

## 结论

按原始描述复核，前后两组的总体区分方向成立：前组多为多种动作、表情和短回应组合，后组主要是单一持续状态、单次点头或静止倾听。但类型拆分粒度和提取遗漏会影响精确名次，因此适合粗筛，不宜直接把前10当作最终最优10段。

这次逐条核对的是已有完整caption与提取统计，没有重新观看视频或听音。类型数差异本身由排序规则决定，不能当作准确率；以下原文复核用于检查排序是否符合文本内容。

## 排序口径

1. 采用类型数降序。
2. 同类型只对有依据的额外重复给奖励，最多2；眨眼、闭眼、微笑不靠次数获奖。
3. 原始人脸短边p10降序；最后用ID稳定排序。

未知次数保持未知，局部待核不扣分。奖励缺失不等于事件没有发生；另做完全不使用次数的敏感性对照。本次未执行人脸门槛或训练准入，全部50段留在排序表。

## 两组概况

| 指标 | 前10 | 后10 |
|---|---:|---:|
| 采用类型数中位数 | 5.000 | 1.000 |
| 时长中位数／秒 | 6.468 | 6.542 |
| 人脸p10中位数／像素 | 175.984 | 150.932 |
| 有官方发声支持的片段 | 7.000 | 0.000 |
| 人脸p10≥160的片段 | 6.000 | 2.000 |
| 含局部待核项的片段 | 2.000 | 3.000 |

去掉次数奖励后，前10重合10/10，后10重合10/10。只说明本批次名单对次数奖励不敏感，不代表全量未知次数不会受影响。

两组时长中位数接近，前组也含短片段、后组也含长片段；未见本批排序被时长单独决定。但这不是经过匹配的时长因果检验。

## 前10逐条检查

| 排名 | 样本 | 类型数 | 秒 | 人脸p10 | 原文复核 |
|---:|---|---:|---:|---:|---|
| 1 | naturalistic_V03_S1485_I00000146_local_0f7d71819484 | 7 | 6.705 | 198.4 | 高组合理：持续微笑、两次点头、笑声和Yeah均有原文支持；嘴部/闭眼可属同一笑的组成，7种不等于7个独立回应。 |
| 2 | improvised_V00_S1249_I00001114_local_126b3bd30e76 | 6 | 4.982 | 144.6 | 高组基本合理：视线转移、点头、皱眉、开口和right。普通眨眼贡献有限；人脸p10不足160，不应因丰富就自动入选。 |
| 3 | naturalistic_V00_S0760_I00000309_local_6edce73cfc56 | 5 | 9.858 | 139.4 | 两次点头和轻笑支持丰富回应；眼部运动与视线转移可能存在粒度重叠。人脸p10不足160。 |
| 4 | naturalistic_V03_S1348_I00000393_local_52f27cf92f82 | 5 | 7.022 | 257.0 | 高组合理：微笑后抬头、连续轻笑、举手放下和点头，原文有清楚的行为序列。 |
| 5 | naturalistic_V03_S1788_I00000293_local_c5a3bf7e6ab6 | 5 | 6.621 | 251.7 | 动作确实多，但抬手、调整眼镜、放手是同一复合行为的多个阶段，5种可能高估独立回应丰富度。 |
| 6 | naturalistic_V02_S3631_I00000411_local_4c5694a3adc3 | 5 | 5.664 | 188.9 | 高组合理：点头、皱眉、嘴唇变化、短yeah；眨眼不应成为主要加分来源。 |
| 7 | naturalistic_V00_S0552_I00000371_local_6346662ba47c | 5 | 8.143 | 165.9 | 高组合理：微笑、闭眼点头、抿嘴和hmm；原文支持不同模态的回应组合。 |
| 8 | improvised_V00_S0606_I00000797_local_af4f5a40d6b5 | 5 | 6.314 | 156.1 | 有闭眼、视线和头部倾斜变化；抬眉与额头起皱可能是同一表情的组成，类型数偏细，人脸p10不足160。 |
| 9 | improvised_V00_S1506_I00001013_local_de761b5fc733 | 5 | 2.571 | 140.8 | 高排名需谨慎：皱眉加一次倾头/开口发声及后续口动；嘴部开合与口部运动可能重复描述同一发声过程，人脸p10不足160。 |
| 10 | naturalistic_V02_S5232_I00000027_local_c9767b16b310 | 4 | 2.985 | 186.1 | 高组合理：两次点头后转头、微笑和张嘴，有明确重复及状态变化；无声音不影响视觉丰富度。 |

## 后10逐条检查

| 排名 | 样本 | 类型数 | 秒 | 人脸p10 | 原文复核 |
|---:|---|---:|---:|---:|---|
| 41 | naturalistic_V02_S3407_I00000728_local_508b67af7f08 | 2 | 5.925 | 149.8 | 相对单一：一次点头后持续微笑，低组可理解，但不是无回应，不应删除。 |
| 42 | naturalistic_V02_S3535_I00000399_local_d95bc0f39990 | 2 | 7.243 | 149.3 | 相对单一：一次点头后开始并维持微笑，仍是有效的倾听回应。 |
| 43 | improvised_V00_S0290_I00000479_local_addfd726e77f | 1 | 9.120 | 192.6 | 种类少但有两次点头；排低符合种类优先，却不代表回应次数少，应由重复回应补充保留。 |
| 44 | naturalistic_V00_S0312_I00000520_local_0ac15af8dcd6 | 1 | 3.323 | 163.8 | 整体较单一：微笑减弱至中性，伴细微头角调整；待核的头动未计入，当前数字是保守统计。 |
| 45 | naturalistic_V02_S3373_I00000414_local_e5a4ed2fdefb | 1 | 6.920 | 158.5 | 存在低估：原文除一次点头还有闭眼后持续保持；提取错写眨眼被留待核，因此采用类型仅1。 |
| 46 | naturalistic_V02_S3535_I00000399_local_2dedff83681f | 1 | 6.163 | 156.3 | 低组合理：主要静态姿势，只有一次轻微点头。 |
| 47 | naturalistic_V00_S0155_I00000126_local_2d4cac53ee97 | 1 | 10.808 | 152.1 | 低组合理：长片段里主要中性表情和静止注视，末尾一次点头；细微姿态变化未确定。 |
| 48 | improvised_V01_S0434_I00001240_local_ed364987637f | 1 | 10.723 | 141.0 | 低组合理：主要维持皱眉和嘴角向下的状态，无明确发声或多次回应变化。 |
| 49 | naturalistic_V02_S3399_I00000729_local_6144bcc8c1bc | 0 | 2.321 | 147.6 | 低组合理：中性表情、闭唇、固定注视和无声，原文未描述明显回应变化。 |
| 50 | improvised_V00_S1979_I00001011_local_2038013807d9 | 0 | 2.441 | 134.1 | 按原文可见描述属于静止低变化；文中另称补充提示有点头但画面不可见，不能把0种当作确定没有点头。 |

## 额外检查的中间位置反例

- 第13名 `naturalistic_V00_S0552_I00000371_local_e9f5a66941a5`：第13名值得保留：两次点头、两次Yeah、持续微笑，只算3种，次数丰富却排在5种细动作之后，说明种类优先不等于综合最优。
- 第14名 `naturalistic_V03_S0466_I00000042_local_846c91e5c7dc`：第14名有两次轻哼；次数与原文相符，不应因类型少于细拆动作就被硬截掉。

## 使用建议

保留类型和数量统计，用于粗分丰富度层级；真正选训练集时先应用画质、人脸和划分门槛，再补充同类多次回应及稀有行为。无需因本次检查重跑视频打标。不要因低排名或待核条目删除片段，也不要把当前类型数当成独立事件总数。此次只做检查，没有改排序字段、原始标注、统计表或训练清单。

## 原始描述与采用项

### 第1名：naturalistic_V03_S1485_I00000146_local_0f7d71819484

The listener maintains a consistent smile throughout the clip, her head tilted slightly back at first before settling into a more neutral position. She nods twice, the first time between approximately one and two seconds, and again between three and four seconds. Her mouth opens several times during the clip, producing audible laughter and a single spoken word, "Yeah," which she says clearly around the two-second mark. The rest of the time, she remains visually engaged with a smiling expression, occasionally closing her eyes briefly while laughing.

采用类型：audio:audible laughter、audio:speech、visual:eye closing、visual:head nodding、visual:head tilting、visual:mouth opening、visual:smiling

明确各类次数：{"audio:speech": 1, "visual:head nodding": 2}

次数未知的类型：audio:audible laughter、visual:eye closing、visual:head tilting、visual:mouth opening

### 第2名：improvised_V00_S1249_I00001114_local_126b3bd30e76

The listener maintains a steady forward gaze throughout the clip, her dark hair pulled back into a high bun with a few loose strands framing her face. She wears large hoop earrings and a pink fleece jacket with a purple collar. Initially, she looks slightly downward before lifting her eyes forward again. Around the 1.6-second mark, she performs a single, subtle head nod. Her expression remains largely neutral, with slight furrowing of the brow and occasional blinking. Towards the end of the clip, she briefly opens her mouth as if to speak, but no words are audible. The audio contains a single, clear utterance of the word "right" at approximately the 3.5-second mark.

采用类型：audio:vocalization、visual:blinking、visual:brow furrowing、visual:gaze shifting、visual:head nodding、visual:mouth opening

明确各类次数：{"audio:vocalization": 1, "visual:head nodding": 1}

次数未知的类型：visual:blinking、visual:gaze shifting、visual:mouth opening

### 第3名：naturalistic_V00_S0760_I00000309_local_6edce73cfc56

The listener maintains a steady forward-facing posture throughout the clip, with her dark hair parted down the middle and falling around her shoulders. Initially, she looks slightly downward before lifting her gaze forward, where it remains for most of the duration. Her facial expression is generally neutral, with subtle shifts in her mouth and eye position that suggest ongoing listening or thought. At approximately seven seconds, she performs a single, distinct head nod, tilting her head downward and then back up. This nod is followed by another similar motion toward the end of the clip. Between these nods, her head remains relatively still. She does not speak, but a soft, breathy chuckle is audible near the beginning of the clip, after which there is no further vocalization.

采用类型：audio:vocalization、visual:eye movement、visual:gaze shifting、visual:head nodding、visual:mouth movement

明确各类次数：{"visual:head nodding": 2}

次数未知的类型：audio:vocalization、visual:eye movement、visual:gaze shifting、visual:mouth movement

### 第4名：naturalistic_V03_S1348_I00000393_local_52f27cf92f82

The listener begins by looking down and smiling, then lifts their head to look forward with a broad smile that persists for several seconds. While maintaining this expression, they laugh audibly, producing a series of short, breathy chuckles. Toward the end of the clip, they briefly raise both hands in front of their chest, palms facing each other, before lowering them again. A single, distinct head nod occurs during this final moment.

采用类型：audio:audible laughter、visual:gaze shifting、visual:hand raising、visual:head nodding、visual:smiling

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：audio:audible laughter、visual:gaze shifting、visual:hand raising

### 第5名：naturalistic_V03_S1788_I00000293_local_c5a3bf7e6ab6

The listener, a person with long blonde hair and glasses wearing a blue hooded sweatshirt, is seated against a blue quilted background. Initially, they raise their right hand to adjust their glasses before lowering it and maintaining a steady forward gaze for the remainder of the clip. Their head nods downward once between approximately 0.5 and 1.3 seconds. The listener blinks several times throughout the clip but otherwise remains still, with no discernible changes in facial expression. There are no audible vocalizations, speech, or other sounds produced by the listener during the entire duration.

采用类型：visual:adjusting glasses、visual:blinking、visual:hand lowering、visual:hand raising、visual:head nodding

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：visual:adjusting glasses、visual:blinking、visual:hand lowering、visual:hand raising

### 第6名：naturalistic_V02_S3631_I00000411_local_4c5694a3adc3

The listener, wearing a red hooded sweatshirt and silver hoop earrings, maintains a generally still posture throughout the clip. Initially, her gaze is directed slightly off-camera to her right, and she blinks naturally. Around the 1.3-second mark, she performs a single, subtle head nod forward and back. Her facial expression shifts subtly during the clip, with her lips parting slightly at times and her brow furrowing briefly around the two-second mark before relaxing again. She continues to look mostly forward for the remainder of the clip, blinking occasionally. Towards the end, she produces a soft, low-pitched vocalization that sounds like a breathy "yeah."

采用类型：audio:vocalization、visual:blinking、visual:brow furrowing、visual:head nodding、visual:mouth opening

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：audio:vocalization、visual:blinking

### 第7名：naturalistic_V00_S0552_I00000371_local_6346662ba47c

The listener maintains a consistent forward-facing posture throughout the clip. Initially, she looks slightly down and to her right with a faint smile. Around two seconds in, she briefly closes her eyes and nods her head downward once. She then opens her eyes and continues looking forward with a neutral expression. Towards the end of the clip, she makes a soft, audible "hmm" sound while pursing her lips slightly.

采用类型：audio:vocalization、visual:blinking、visual:head nodding、visual:lip pursing、visual:smiling

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：audio:vocalization、visual:blinking

### 第8名：improvised_V00_S0606_I00000797_local_af4f5a40d6b5

The listener maintains a steady forward-facing posture throughout the clip. Initially, his eyebrows are raised and his forehead is wrinkled, creating a furrowed expression. His eyes remain open and directed forward for most of the duration. Around the midpoint, he briefly closes his eyes and lowers his gaze slightly before reopening them and looking forward again. Toward the end of the clip, he tilts his head slightly to his right while keeping his eyes open and focused forward. There are no audible vocalizations during the entire clip.

采用类型：visual:blinking、visual:eyebrow raising、visual:forehead wrinkling、visual:gaze shifting、visual:head tilting

明确各类次数：{}

次数未知的类型：visual:blinking、visual:gaze shifting、visual:head tilting

### 第9名：improvised_V00_S1506_I00001013_local_de761b5fc733

The listener maintains a steady forward-facing posture throughout the clip, her head remaining mostly still with only slight shifts in orientation. Her facial expression is consistent, showing furrowed brows and a downturned mouth that persists for the entire duration. She wears large silver hoop earrings and a black crew-neck shirt, with a small microphone clipped near the collar. While no distinct hand or body movements are observed, she begins to speak around the two-second mark, producing a single, short utterance that sounds like "It." This vocalization occurs simultaneously with a subtle downward tilt of her head and a slight opening of her mouth. Following this, she continues speaking with her mouth moving visibly, though no further intelligible words are captured. The clip ends with her head tilted slightly downward and her gaze directed away from the camera.

采用类型：audio:speech、visual:frowning、visual:head tilting、visual:mouth movement、visual:mouth opening

明确各类次数：{}

次数未知的类型：audio:speech、visual:head tilting、visual:mouth movement、visual:mouth opening

### 第10名：naturalistic_V02_S5232_I00000027_local_c9767b16b310

The listener maintains a forward-facing posture throughout the clip. Initially, her head is tilted slightly back and to her right, with her eyes looking upward and to her left. She then performs two distinct head nods: first, a brief downward tilt around the 1.1-second mark, followed by another similar motion near the 2.5-second mark. After the second nod, she turns her head forward and breaks into a smile that persists until the end of the clip. Her mouth opens during the smile, but no speech or other vocalizations are audible.

采用类型：visual:head nodding、visual:head turning、visual:mouth opening、visual:smiling

明确各类次数：{"visual:head nodding": 2}

次数未知的类型：visual:head turning

### 第13名：naturalistic_V00_S0552_I00000371_local_e9f5a66941a5

The listener maintains a consistent forward-facing posture throughout the clip, with long dark hair framing her face and small hoop earrings visible. Initially, she has a neutral expression with her lips closed, then begins to smile slightly while looking forward. Her head nods downward twice during the clip, first between approximately two and three seconds, and again near six seconds, each time accompanied by a subtle shift in facial expression. As the clip progresses, she continues to smile more broadly, showing her teeth, and her mouth opens slightly as she speaks the word "Yeah" at around four seconds. She repeats the word "Yeah" again shortly after, maintaining eye contact and a steady upper body position.

采用类型：audio:speech、visual:head nodding、visual:smiling

明确各类次数：{"audio:speech": 2, "visual:head nodding": 2}

次数未知的类型：

### 第14名：naturalistic_V03_S0466_I00000042_local_846c91e5c7dc

The listener maintains a mostly still posture throughout the clip, with her hair pulled back into a bun and her gaze directed forward. Initially, she blinks slowly and keeps her head steady. Then, at approximately one second, she produces a soft, low-pitched hum, followed by a brief pause before emitting another similar sound around seven seconds. During these vocalizations, her lips move slightly, but there is no significant change in facial expression or head position. She continues to look forward for the remainder of the clip, blinking occasionally and making very subtle shifts in her head angle, though no distinct nodding motions are observed beyond the provided supplementary data.

采用类型：audio:humming、visual:blinking、visual:mouth movement

明确各类次数：{"audio:humming": 2}

次数未知的类型：visual:blinking、visual:mouth movement

### 第41名：naturalistic_V02_S3407_I00000728_local_508b67af7f08

The listener maintains a consistent upward gaze throughout the clip. Initially, he performs a single, brief nod of his head. Following this, he holds a steady posture with his head tilted slightly back and his mouth closed, displaying a subtle smile that persists for the remainder of the recording. There are no other discernible facial movements, head shifts, or hand gestures observed. The listener remains visually still and silent for the entire duration, with no audible vocalizations or speech detected.

采用类型：visual:head nodding、visual:smiling

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：

### 第42名：naturalistic_V02_S3535_I00000399_local_d95bc0f39990

The listener maintains a steady forward gaze throughout the clip. Initially, her expression is neutral with her lips closed. Then, around the three-second mark, she performs a single, brief head nod. Following this, her facial expression changes as she begins to smile, showing her upper teeth, and continues smiling for the remainder of the clip. There are no audible vocalizations at any point during the recording.

采用类型：visual:head nodding、visual:smiling

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：

### 第43名：improvised_V00_S0290_I00000479_local_addfd726e77f

The listener maintains a consistent downward gaze throughout the clip, her head tilted slightly forward and remaining mostly still. Her facial expression is neutral, with her lips closed and eyebrows relaxed. At approximately two seconds, she performs a single, brief head nod, tilting her head down and up quickly before returning to her initial posture. She repeats this nod around seven seconds, again moving her head down and up in a short, distinct motion. Aside from these two instances, there is no other discernible head or body movement. The listener does not speak or make any audible vocalizations during the entire duration.

采用类型：visual:head nodding

明确各类次数：{"visual:head nodding": 2}

次数未知的类型：

### 第44名：naturalistic_V00_S0312_I00000520_local_0ac15af8dcd6

The listener maintains a steady forward gaze throughout the clip, her expression shifting subtly from a slight smile to a more neutral, attentive look. Her head remains mostly still, with only minor adjustments in angle. She wears a green top, a gold chain necklace, and small hoop earrings, and a microphone is clipped to her collar. There is no audible speech or vocalization during the entire duration.

采用类型：visual:smiling

明确各类次数：{}

次数未知的类型：

### 第45名：naturalistic_V02_S3373_I00000414_local_e5a4ed2fdefb

The listener maintains a steady upward gaze throughout the clip. Initially, he performs a single, brief head nod. His eyes remain open for several seconds before closing completely around the three-second mark. While his eyes are closed, his facial expression remains unchanged, showing no visible signs of smiling or frowning. He holds this still posture with closed eyes until the end of the clip. There are no audible vocalizations at any point during the recording.

采用类型：visual:head nodding

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：

### 第46名：naturalistic_V02_S3535_I00000399_local_2dedff83681f

The listener maintains a steady forward gaze throughout the clip, his head held upright with minimal movement. His facial expression remains largely unchanged, featuring a neutral mouth position and relaxed brow. At approximately four seconds into the clip, he performs a single, brief downward nod of the head before returning to his initial posture. This nod occurs while his eyes remain open and directed forward. No other discernible movements of the head, eyes, or body take place during the remainder of the clip. The listener does not produce any audible vocalizations, including speech or laughter, for the entire duration.

采用类型：visual:head nodding

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：

### 第47名：naturalistic_V00_S0155_I00000126_local_2d4cac53ee97

The listener maintains a steady forward gaze throughout the clip, his head remaining mostly still with only slight shifts in posture. His facial expression is consistent, showing a neutral look with a closed mouth and relaxed features. Toward the end of the clip, he performs a single, subtle head nod. There are no audible vocalizations during the entire duration.

采用类型：visual:head nodding

明确各类次数：{"visual:head nodding": 1}

次数未知的类型：

### 第48名：improvised_V01_S0434_I00001240_local_ed364987637f

The listener maintains a fixed forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position. Her facial expression is tense, featuring furrowed brows and a downturned mouth, which persists for the entire duration. There are no discernible hand, arm, or body movements observed. The listener does not produce any audible vocalizations, including speech or nonverbal sounds like laughter, and remains silent for the full length of the clip.

采用类型：visual:frowning

明确各类次数：{}

次数未知的类型：

### 第49名：naturalistic_V02_S3399_I00000729_local_6144bcc8c1bc

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His dark eyes remain fixed on a point ahead, showing no discernible shift in focus or direction. The thick, dark beard framing his face remains motionless, and there is no visible movement in his shoulders or upper body. He wears a light-colored knit beanie that sits snugly on his head, and his expression stays neutral with lips closed. There are no audible vocalizations, laughter, or speech detected during the entire duration.

采用类型：

明确各类次数：{}

次数未知的类型：

### 第50名：improvised_V00_S1979_I00001011_local_2038013807d9

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His facial expression remains largely neutral, with his lips closed and eyebrows relaxed. There is no discernible movement of his hands or body. The provided supplementary information indicates that he nods at certain points during the clip, but these specific instances are not visually apparent in the available frames. No audible vocalizations, such as speech or laughter, are present.

采用类型：

明确各类次数：{}

次数未知的类型：

