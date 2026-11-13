# 50段整体丰富度评分复核

LLM直接阅读已有描述和时间证据打0–5分；不是事件次数。全部片段保留，以下按评分、原始人脸p10降序排列。尚未执行训练集筛选。

| 样本 | 分数 | 人脸p10 | 发声支持秒 | 评分理由 |
|---|---:|---:|---:|---|
| naturalistic_V03_S1348_I00000393_local_52f27cf92f82 | 4 | 257.0 | 0.885 | 多种明显动作变化：低头笑→抬头大笑→举手→点头，且笑声与举手为持续性行为，体现丰富响应序列。 |
| naturalistic_V03_S0877_I00000411_local_ed7cb61d3086 | 4 | 205.0 | 0.412 | 两次清晰的点头动作，每次均有明确的下倾-上抬运动，且中间有非重复的微小眼动变化，体现多样且重复的响应行为。 |
| naturalistic_V03_S1485_I00000146_local_0f7d71819484 | 4 | 198.4 | 2.324 | 包含微笑、两次点头、多次张嘴笑及发声，行为多样且有重复特征 |
| improvised_V00_S0290_I00000479_local_addfd726e77f | 4 | 192.6 | 0.000 | 两次清晰的点头动作，动作模式一致且有间隔，体现重复行为；头部姿态稳定，无其他明显动作，行为变化明确且具重复性。 |
| naturalistic_V02_S5232_I00000027_local_c9767b16b310 | 4 | 186.1 | 0.000 | 两次明确的点头动作，头部前倾后转向前，持续微笑，表情变化丰富且有重复特征。 |
| naturalistic_V00_S0552_I00000371_local_e9f5a66941a5 | 4 | 165.8 | 0.664 | 多次明显表情变化与重复回应，包含点头、微笑扩展及两次'Yeah'回应，行为多样且有重复 |
| improvised_V01_S1827_I00000704_local_89d8db1b77e0 | 4 | 141.5 | 0.152 | 多个明显不同的反应行为，包括微笑、点头、凝视转移和持续张口，且点头与张口均有明确持续性表现。 |
| naturalistic_V03_S1211_I00000158_local_907fd33efcbf | 3 | 350.1 | 0.000 | 包含眼神、头部姿态的多次变化及持续状态，体现明显的行为序列 |
| naturalistic_V03_S1003_I00000059_local_f2ca565ac69c | 3 | 243.3 | 0.000 | 包含初始大笑、头部微动、一次点头及持续柔和微笑，视觉反应多样且有明确变化 |
| naturalistic_V03_S0466_I00000042_local_846c91e5c7dc | 3 | 210.0 | 1.473 | 包含两次明显发声及唇部微动，有状态变化和重复行为，但无显著头部动作或表情变化。 |
| naturalistic_V03_S0360_I00000384_local_7223dfefce8f | 3 | 198.4 | 0.000 | 包含多次点头、面部表情变化及间歇性嘴部开合，行为多样且有明显变化 |
| naturalistic_V02_S3631_I00000411_local_4c5694a3adc3 | 3 | 188.9 | 0.387 | 包含点头、面部微表情变化及发声反应，行为多样且有明显变化 |
| naturalistic_V00_S0552_I00000371_local_6346662ba47c | 3 | 165.9 | 0.856 | 包含姿态、眼神、点头和唇部动作的多种变化，且有明确的重复行为特征 |
| naturalistic_V00_S0725_I00000125_local_ffbb8d3f7806 | 3 | 156.9 | 0.380 | 面部表情变化与微笑持续，伴随眼神上移，有明确的视觉响应变化和一次发声反应。 |
| improvised_V00_S0606_I00000797_local_af4f5a40d6b5 | 3 | 156.1 | 0.000 | 包含面部表情、眼神变化和头部动作的多种明显反应，且有清晰的阶段变化 |
| naturalistic_V02_S5043_I00000069_local_26d534af4d65 | 3 | 155.7 | 0.627 | 包含头部点头、闭眼后仰、发声反应及状态回归，行为变化明显且有重复特征 |
| naturalistic_V02_S3648_I00000032_local_fb8abbfe8ad2 | 3 | 151.2 | 0.412 | 包含眼神闭合、点头和发声三种不同反应，且有明确变化 |
| naturalistic_V00_S0864_I00000382_local_6dceb9e9b771 | 3 | 144.0 | 0.292 | 包含手势、闭眼抿嘴、持续凝视等多个明显动作变化，且有明确重复行为特征 |
| improvised_V00_S1506_I00001013_local_de761b5fc733 | 3 | 140.8 | 0.316 | 头部微倾与口部开合为一次明确反应，后续持续口动构成另一反应，两者有明显变化，且反应具持续性。 |
| naturalistic_V00_S0760_I00000309_local_6edce73cfc56 | 3 | 139.4 | 0.634 | 两次明确的点头动作，配合眼神和面部细微变化，体现多样且有意义的反应行为。 |
| improvised_V01_S0296_I00001107_local_38a57f0ff7f9 | 2 | 292.0 | 2.340 | 一个持续状态加一次明显反应变化，微笑持续存在，眼睑开合为一次明确变化。 |
| naturalistic_V03_S1788_I00000293_local_c5a3bf7e6ab6 | 2 | 251.7 | 0.000 | 一次明确的点头动作，配合手部调整眼镜的单一行为，其余为静止状态。 |
| naturalistic_V03_S0877_I00000411_local_4d2ff70606d7 | 2 | 251.0 | 0.604 | 一个持续状态加一次明确的点头反应，无其他明显变化。 |
| naturalistic_V03_S0268_I00000486_local_85eedccbc632 | 2 | 246.4 | 0.000 | 一个持续状态加一次明确反应变化，反应重复性不明显 |
| naturalistic_V03_S0877_I00000411_local_7e467a7be48c | 2 | 202.1 | 0.412 | 一个持续状态加一次明确反应，反应后回归静态，无其他变化或重复行为。 |
| improvised_V00_S0769_I00000377_local_d9a6e037ecfa | 2 | 175.3 | 0.000 | 一个明确的反应变化，包含闭眼和低头动作，且重复发生一次，但无其他明显变化。 |
| naturalistic_V00_S0498_I00000495_local_794a0ccac190 | 2 | 165.4 | 0.000 | 中性表情持续，一次眼神下移后恢复，末尾单次轻微点头，行为变化有限但有明确重复特征 |
| naturalistic_V02_S3631_I00000411_local_bb07a099ca84 | 2 | 164.5 | 0.942 | 头部轻微倾斜后归中，一次明确点头，面部表情持续不变，变化有限。 |
| improvised_V00_S0769_I00000377_local_bbe4a4d914f6 | 2 | 164.4 | 0.000 | 单一点头动作后持续说话，面部有轻微变化，但无重复行为或多重反应变化。 |
| naturalistic_V00_S0312_I00000520_local_4d9976487646 | 2 | 160.3 | 0.000 | 一个持续注视状态加一次明确点头，随后有轻微视线转移，但无其他明显反应变化。 |
| improvised_V00_S0606_I00000797_local_7b1dc729764c | 2 | 156.7 | 0.023 | 持续的嘴部运动与单一音调的持续发声构成一种重复行为，面部表情略有变化但整体稳定，无其他明显反应行为。 |
| naturalistic_V02_S3156_I00000386_local_0e5b708d3954 | 2 | 151.0 | 0.000 | 一个持续状态加一次明确反应变化，后续保持单一手势不变。 |
| naturalistic_V02_S3535_I00000399_local_d95bc0f39990 | 2 | 149.3 | 0.000 | 一个持续状态加一次明确反应变化，后续持续微笑，无其他明显行为变化。 |
| improvised_V00_S1249_I00001114_local_126b3bd30e76 | 2 | 144.6 | 0.240 | 一次明显点头动作，伴随眼神微调，其余为静态或单一重复行为 |
| naturalistic_V03_S0476_I00000315_local_15fa8c6b5c9a | 1 | 296.4 | 0.977 | 仅描述一次手部动作和持续张口，无其他明显反应变化或重复行为。 |
| naturalistic_V03_S1393_I00000111_local_76b117b06695 | 1 | 295.8 | 0.000 | 仅描述了持续的面部表情和一次简短点头，无其他明显反应变化。 |
| naturalistic_V03_S1532_I00000036_local_0a47cb3ab2a1 | 1 | 261.1 | 0.000 | 仅描述一次短暂点头及持续静态表情，无其他明显反应行为变化。 |
| naturalistic_V03_S1421_I00000117_local_674f523d111d | 1 | 220.8 | 0.000 | 仅描述持续微笑和一次眼神变化，无其他明显行为变化或重复 |
| naturalistic_V03_S0782_I00000059_local_7f5d8964d753 | 1 | 219.6 | 0.000 | 仅描述一次短暂点头，后续为静态凝视与轻微眨眼，无其他明显反应行为。 |
| naturalistic_V02_S3601_I00000156_local_719d6c438834 | 1 | 170.4 | 0.429 | 仅描述一次开口后闭合的面部变化，后续全程静止，无其他明显反应行为。 |
| naturalistic_V00_S0312_I00000520_local_0ac15af8dcd6 | 1 | 163.8 | 0.000 | 仅描述了持续凝视和表情轻微变化，无其他明显反应行为。 |
| naturalistic_V02_S3373_I00000414_local_e5a4ed2fdefb | 1 | 158.5 | 0.000 | 仅描述一次短暂点头和持续凝视，后续为静态闭眼状态，无其他明显反应变化。 |
| improvised_V00_S2055_I00001096_local_729e50c3c598 | 1 | 156.6 | 0.000 | 仅描述一次短暂点头，其余为静态下视姿态，无其他明显反应行为。 |
| naturalistic_V02_S3535_I00000399_local_2dedff83681f | 1 | 156.3 | 0.000 | 仅描述一次短暂点头，其余为静态姿态，无其他明显反应行为。 |
| improvised_V00_S0606_I00000797_local_fa36d6a09bb2 | 1 | 152.4 | 2.562 | 仅描述持续的静止姿态和单一面部表情，无明显变化或重复行为。 |
| naturalistic_V00_S0155_I00000126_local_2d4cac53ee97 | 1 | 152.1 | 0.000 | 仅描述了持续的中性表情和一次轻微点头，无其他明显变化或重复行为。 |
| naturalistic_V02_S3407_I00000728_local_508b67af7f08 | 1 | 149.8 | 0.000 | 仅描述一次短暂点头和持续静止状态，无其他明显反应变化。 |
| improvised_V01_S0434_I00001240_local_ed364987637f | 1 | 141.0 | 0.000 | 仅描述持续的固定凝视与紧张面部表情，无其他变化或重复行为。 |
| improvised_V00_S1979_I00001011_local_2038013807d9 | 1 | 134.1 | 0.000 | 仅描述持续凝视与静态姿态，虽提及点头但未见视觉表现，无其他明显反应行为。 |
| naturalistic_V02_S3399_I00000729_local_6144bcc8c1bc | 0 | 147.6 | 0.000 | 全程无明显表情或动作变化，仅描述静态姿态与无声状态 |

## naturalistic_V03_S1348_I00000393_local_52f27cf92f82

**评分 4/5**：多种明显动作变化：低头笑→抬头大笑→举手→点头，且笑声与举手为持续性行为，体现丰富响应序列。

不确定项：模型未报告（不代表没有错误）

时长 7.022s；原始人脸p10/p50 257.0/279.5px；官方发声支持 0.885s；点头候选支持 0.440s。

The listener begins by looking down and smiling, then lifts their head to look forward with a broad smile that persists for several seconds. While maintaining this expression, they laugh audibly, producing a series of short, breathy chuckles. Toward the end of the clip, they briefly raise both hands in front of their chest, palms facing each other, before lowering them again. A single, distinct head nod occurs during this final moment.

评分引用的原句：

> s001: The listener begins by looking down and smiling, then lifts their head to look forward with a broad smile that persists for several seconds.

> s002: While maintaining this expression, they laugh audibly, producing a series of short, breathy chuckles.

> s003: Toward the end of the clip, they briefly raise both hands in front of their chest, palms facing each other, before lowering them again.

> s004: A single, distinct head nod occurs during this final moment.


## naturalistic_V03_S0877_I00000411_local_ed7cb61d3086

**评分 4/5**：两次清晰的点头动作，每次均有明确的下倾-上抬运动，且中间有非重复的微小眼动变化，体现多样且重复的响应行为。

不确定项：模型未报告（不代表没有错误）

时长 9.549s；原始人脸p10/p50 205.0/207.0px；官方发声支持 0.412s；点头候选支持 1.400s。

Assistant: The listener maintains a steady forward gaze throughout the clip, her head held upright and still. She wears large silver hoop earrings and a gray high-necked sweater, with a small black microphone clipped near her collar. Her dark hair frames her face as she blinks occasionally and shifts her eyes slightly upward and to the side. At approximately five seconds, she performs a single, distinct head nod, tilting her head down and up quickly before returning to her initial position. This nod repeats around seven seconds, again showing a brief downward tilt followed by a return to the original posture. No other significant movements of the head, hands, or body are observed. The listener does not speak, laugh, or make any other audible vocalizations during the clip.

评分引用的原句：

> s001: Assistant: The listener maintains a steady forward gaze throughout the clip, her head held upright and still.

> s004: At approximately five seconds, she performs a single, distinct head nod, tilting her head down and up quickly before returning to her initial position.

> s005: This nod repeats around seven seconds, again showing a brief downward tilt followed by a return to the original posture.

> s003: Her dark hair frames her face as she blinks occasionally and shifts her eyes slightly upward and to the side.


## naturalistic_V03_S1485_I00000146_local_0f7d71819484

**评分 4/5**：包含微笑、两次点头、多次张嘴笑及发声，行为多样且有重复特征

不确定项：模型未报告（不代表没有错误）

时长 6.705s；原始人脸p10/p50 198.4/200.9px；官方发声支持 2.324s；点头候选支持 0.800s。

The listener maintains a consistent smile throughout the clip, her head tilted slightly back at first before settling into a more neutral position. She nods twice, the first time between approximately one and two seconds, and again between three and four seconds. Her mouth opens several times during the clip, producing audible laughter and a single spoken word, "Yeah," which she says clearly around the two-second mark. The rest of the time, she remains visually engaged with a smiling expression, occasionally closing her eyes briefly while laughing.

评分引用的原句：

> s001: The listener maintains a consistent smile throughout the clip, her head tilted slightly back at first before settling into a more neutral position.

> s002: She nods twice, the first time between approximately one and two seconds, and again between three and four seconds.

> s003: Her mouth opens several times during the clip, producing audible laughter and a single spoken word, "Yeah," which she says clearly around the two-second mark.

> s004: The rest of the time, she remains visually engaged with a smiling expression, occasionally closing her eyes briefly while laughing.


## improvised_V00_S0290_I00000479_local_addfd726e77f

**评分 4/5**：两次清晰的点头动作，动作模式一致且有间隔，体现重复行为；头部姿态稳定，无其他明显动作，行为变化明确且具重复性。

不确定项：模型未报告（不代表没有错误）

时长 9.120s；原始人脸p10/p50 192.6/194.7px；官方发声支持 0.000s；点头候选支持 1.320s。

The listener maintains a consistent downward gaze throughout the clip, her head tilted slightly forward and remaining mostly still. Her facial expression is neutral, with her lips closed and eyebrows relaxed. At approximately two seconds, she performs a single, brief head nod, tilting her head down and up quickly before returning to her initial posture. She repeats this nod around seven seconds, again moving her head down and up in a short, distinct motion. Aside from these two instances, there is no other discernible head or body movement. The listener does not speak or make any audible vocalizations during the entire duration.

评分引用的原句：

> s001: The listener maintains a consistent downward gaze throughout the clip, her head tilted slightly forward and remaining mostly still.

> s003: At approximately two seconds, she performs a single, brief head nod, tilting her head down and up quickly before returning to her initial posture.

> s004: She repeats this nod around seven seconds, again moving her head down and up in a short, distinct motion.

> s005: Aside from these two instances, there is no other discernible head or body movement.


## naturalistic_V02_S5232_I00000027_local_c9767b16b310

**评分 4/5**：两次明确的点头动作，头部前倾后转向前，持续微笑，表情变化丰富且有重复特征。

不确定项：模型未报告（不代表没有错误）

时长 2.985s；原始人脸p10/p50 186.1/187.6px；官方发声支持 0.000s；点头候选支持 0.680s。

The listener maintains a forward-facing posture throughout the clip. Initially, her head is tilted slightly back and to her right, with her eyes looking upward and to her left. She then performs two distinct head nods: first, a brief downward tilt around the 1.1-second mark, followed by another similar motion near the 2.5-second mark. After the second nod, she turns her head forward and breaks into a smile that persists until the end of the clip. Her mouth opens during the smile, but no speech or other vocalizations are audible.

评分引用的原句：

> s003: She then performs two distinct head nods: first, a brief downward tilt around the 1.1-second mark, followed by another similar motion near the 2.5-second mark.

> s004: After the second nod, she turns her head forward and breaks into a smile that persists until the end of the clip.

> s005: Her mouth opens during the smile, but no speech or other vocalizations are audible.

> s002: Initially, her head is tilted slightly back and to her right, with her eyes looking upward and to her left.


## naturalistic_V00_S0552_I00000371_local_e9f5a66941a5

**评分 4/5**：多次明显表情变化与重复回应，包含点头、微笑扩展及两次'Yeah'回应，行为多样且有重复

不确定项：模型未报告（不代表没有错误）

时长 7.226s；原始人脸p10/p50 165.8/167.1px；官方发声支持 0.664s；点头候选支持 0.720s。

The listener maintains a consistent forward-facing posture throughout the clip, with long dark hair framing her face and small hoop earrings visible. Initially, she has a neutral expression with her lips closed, then begins to smile slightly while looking forward. Her head nods downward twice during the clip, first between approximately two and three seconds, and again near six seconds, each time accompanied by a subtle shift in facial expression. As the clip progresses, she continues to smile more broadly, showing her teeth, and her mouth opens slightly as she speaks the word "Yeah" at around four seconds. She repeats the word "Yeah" again shortly after, maintaining eye contact and a steady upper body position.

评分引用的原句：

> s002: Initially, she has a neutral expression with her lips closed, then begins to smile slightly while looking forward.

> s003: Her head nods downward twice during the clip, first between approximately two and three seconds, and again near six seconds, each time accompanied by a subtle shift in facial expression.

> s004: As the clip progresses, she continues to smile more broadly, showing her teeth, and her mouth opens slightly as she speaks the word "Yeah" at around four seconds.

> s005: She repeats the word "Yeah" again shortly after, maintaining eye contact and a steady upper body position.


## improvised_V01_S1827_I00000704_local_89d8db1b77e0

**评分 4/5**：多个明显不同的反应行为，包括微笑、点头、凝视转移和持续张口，且点头与张口均有明确持续性表现。

不确定项：模型未报告（不代表没有错误）

时长 5.291s；原始人脸p10/p50 141.5/143.4px；官方发声支持 0.152s；点头候选支持 0.480s。

The listener maintains a steady forward-facing posture throughout the clip. Initially, he displays a broad smile with his teeth showing, and his head remains still. Around the 0.9-second mark, he performs a single, brief nod by tilting his head downward and back up. Following this, he holds a neutral expression with his lips closed for several seconds before shifting his gaze upward and to his right. His mouth opens wide in a sustained expression resembling a laugh or exclamation, continuing until the end of the clip.

评分引用的原句：

> s002: Initially, he displays a broad smile with his teeth showing, and his head remains still.

> s003: Around the 0.9-second mark, he performs a single, brief nod by tilting his head downward and back up.

> s004: Following this, he holds a neutral expression with his lips closed for several seconds before shifting his gaze upward and to his right.

> s005: His mouth opens wide in a sustained expression resembling a laugh or exclamation, continuing until the end of the clip.


## naturalistic_V03_S1211_I00000158_local_907fd33efcbf

**评分 3/5**：包含眼神、头部姿态的多次变化及持续状态，体现明显的行为序列

不确定项：模型未报告（不代表没有错误）

时长 7.044s；原始人脸p10/p50 350.1/360.4px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a generally still posture throughout the clip, wearing a blue hooded sweatshirt and facing forward. Initially, he looks slightly upward and to his right with a neutral expression. He then briefly closes his eyes before reopening them and shifting his gaze toward the camera. His head tilts slightly to his left, and he holds this position for several seconds. Toward the end of the clip, he turns his head further to his left, looking off-camera in that direction. There are no audible vocalizations during the entire duration.

评分引用的原句：

> s002: Initially, he looks slightly upward and to his right with a neutral expression.

> s003: He then briefly closes his eyes before reopening them and shifting his gaze toward the camera.

> s004: His head tilts slightly to his left, and he holds this position for several seconds.

> s005: Toward the end of the clip, he turns his head further to his left, looking off-camera in that direction.


## naturalistic_V03_S1003_I00000059_local_f2ca565ac69c

**评分 3/5**：包含初始大笑、头部微动、一次点头及持续柔和微笑，视觉反应多样且有明确变化

不确定项：模型未报告（不代表没有错误）

时长 6.908s；原始人脸p10/p50 243.3/250.4px；官方发声支持 0.000s；点头候选支持 0.760s。

The listener maintains a consistent seated posture throughout the clip, facing forward with long dark hair framing her face and wearing glasses with thin frames. Initially, she smiles broadly, showing her teeth, and her head moves slightly upward and to the right before returning to center. She then nods once between approximately 0.2 and 0.96 seconds, after which her expression settles into a gentle smile that persists for the remainder of the clip. Her eyes remain open and directed forward, blinking occasionally, while her mouth stays closed except for the initial smile. There are no audible vocalizations, and the background remains static behind her.

评分引用的原句：

> s002: Initially, she smiles broadly, showing her teeth, and her head moves slightly upward and to the right before returning to center.

> s003: She then nods once between approximately 0.2 and 0.96 seconds, after which her expression settles into a gentle smile that persists for the remainder of the clip.

> s004: Her eyes remain open and directed forward, blinking occasionally, while her mouth stays closed except for the initial smile.

> s001: The listener maintains a consistent seated posture throughout the clip, facing forward with long dark hair framing her face and wearing glasses with thin frames.


## naturalistic_V03_S0466_I00000042_local_846c91e5c7dc

**评分 3/5**：包含两次明显发声及唇部微动，有状态变化和重复行为，但无显著头部动作或表情变化。

不确定项：模型未报告（不代表没有错误）

时长 8.630s；原始人脸p10/p50 210.0/213.5px；官方发声支持 1.473s；点头候选支持 0.840s。

The listener maintains a mostly still posture throughout the clip, with her hair pulled back into a bun and her gaze directed forward. Initially, she blinks slowly and keeps her head steady. Then, at approximately one second, she produces a soft, low-pitched hum, followed by a brief pause before emitting another similar sound around seven seconds. During these vocalizations, her lips move slightly, but there is no significant change in facial expression or head position. She continues to look forward for the remainder of the clip, blinking occasionally and making very subtle shifts in her head angle, though no distinct nodding motions are observed beyond the provided supplementary data.

评分引用的原句：

> s003: Then, at approximately one second, she produces a soft, low-pitched hum, followed by a brief pause before emitting another similar sound around seven seconds.

> s004: During these vocalizations, her lips move slightly, but there is no significant change in facial expression or head position.

> s002: Initially, she blinks slowly and keeps her head steady.

> s005: She continues to look forward for the remainder of the clip, blinking occasionally and making very subtle shifts in her head angle, though no distinct nodding motions are observed beyond the provided supplementary data.


## naturalistic_V03_S0360_I00000384_local_7223dfefce8f

**评分 3/5**：包含多次点头、面部表情变化及间歇性嘴部开合，行为多样且有明显变化

不确定项：caption称有说话暗示，但full_silence=true，存在矛盾

时长 7.162s；原始人脸p10/p50 198.4/201.5px；官方发声支持 0.000s；点头候选支持 1.160s。

The listener maintains a consistent forward-facing posture throughout the clip. Initially, they exhibit a series of rapid head nods. Following this, their head remains relatively still for the remainder of the duration. Their facial expression shifts from a broad smile to a more neutral look, with their mouth opening and closing intermittently, suggesting speech. The listener does not produce any audible vocalizations during the clip.

评分引用的原句：

> s002: Initially, they exhibit a series of rapid head nods.

> s003: Following this, their head remains relatively still for the remainder of the duration.

> s004: Their facial expression shifts from a broad smile to a more neutral look, with their mouth opening and closing intermittently, suggesting speech.

> s005: The listener does not produce any audible vocalizations during the clip.


## naturalistic_V02_S3631_I00000411_local_4c5694a3adc3

**评分 3/5**：包含点头、面部微表情变化及发声反应，行为多样且有明显变化

不确定项：模型未报告（不代表没有错误）

时长 5.664s；原始人脸p10/p50 188.9/193.5px；官方发声支持 0.387s；点头候选支持 0.960s。

The listener, wearing a red hooded sweatshirt and silver hoop earrings, maintains a generally still posture throughout the clip. Initially, her gaze is directed slightly off-camera to her right, and she blinks naturally. Around the 1.3-second mark, she performs a single, subtle head nod forward and back. Her facial expression shifts subtly during the clip, with her lips parting slightly at times and her brow furrowing briefly around the two-second mark before relaxing again. She continues to look mostly forward for the remainder of the clip, blinking occasionally. Towards the end, she produces a soft, low-pitched vocalization that sounds like a breathy "yeah."

评分引用的原句：

> s003: Around the 1.3-second mark, she performs a single, subtle head nod forward and back.

> s004: Her facial expression shifts subtly during the clip, with her lips parting slightly at times and her brow furrowing briefly around the two-second mark before relaxing again.

> s006: Towards the end, she produces a soft, low-pitched vocalization that sounds like a breathy "yeah."

> s002: Initially, her gaze is directed slightly off-camera to her right, and she blinks naturally.


## naturalistic_V00_S0552_I00000371_local_6346662ba47c

**评分 3/5**：包含姿态、眼神、点头和唇部动作的多种变化，且有明确的重复行为特征

不确定项：模型未报告（不代表没有错误）

时长 8.143s；原始人脸p10/p50 165.9/167.6px；官方发声支持 0.856s；点头候选支持 1.080s。

The listener maintains a consistent forward-facing posture throughout the clip. Initially, she looks slightly down and to her right with a faint smile. Around two seconds in, she briefly closes her eyes and nods her head downward once. She then opens her eyes and continues looking forward with a neutral expression. Towards the end of the clip, she makes a soft, audible "hmm" sound while pursing her lips slightly.

评分引用的原句：

> s001: The listener maintains a consistent forward-facing posture throughout the clip.

> s002: Initially, she looks slightly down and to her right with a faint smile.

> s003: Around two seconds in, she briefly closes her eyes and nods her head downward once.

> s005: Towards the end of the clip, she makes a soft, audible "hmm" sound while pursing her lips slightly.


## naturalistic_V00_S0725_I00000125_local_ffbb8d3f7806

**评分 3/5**：面部表情变化与微笑持续，伴随眼神上移，有明确的视觉响应变化和一次发声反应。

不确定项：模型未报告（不代表没有错误）

时长 4.342s；原始人脸p10/p50 156.9/160.8px；官方发声支持 0.380s；点头候选支持 0.000s。

The listener maintains a steady forward-facing posture throughout the clip, his head remaining mostly still with only slight shifts in position. Initially, he looks slightly downward before lifting his gaze forward, where it remains for most of the duration. His facial expression transitions from neutral to a subtle smile that persists until the end of the clip. The corners of his mouth turn upward, and his lips part slightly, revealing his upper teeth in a gentle grin. There is no discernible nodding or hand movement observed. Audibly, he says "Yeah" at the beginning of the clip, followed by a period of silence.

评分引用的原句：

> s002: Initially, he looks slightly downward before lifting his gaze forward, where it remains for most of the duration.

> s003: His facial expression transitions from neutral to a subtle smile that persists until the end of the clip.

> s004: The corners of his mouth turn upward, and his lips part slightly, revealing his upper teeth in a gentle grin.

> s006: Audibly, he says "Yeah" at the beginning of the clip, followed by a period of silence.


## improvised_V00_S0606_I00000797_local_af4f5a40d6b5

**评分 3/5**：包含面部表情、眼神变化和头部动作的多种明显反应，且有清晰的阶段变化

不确定项：模型未报告（不代表没有错误）

时长 6.314s；原始人脸p10/p50 156.1/156.7px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a steady forward-facing posture throughout the clip. Initially, his eyebrows are raised and his forehead is wrinkled, creating a furrowed expression. His eyes remain open and directed forward for most of the duration. Around the midpoint, he briefly closes his eyes and lowers his gaze slightly before reopening them and looking forward again. Toward the end of the clip, he tilts his head slightly to his right while keeping his eyes open and focused forward. There are no audible vocalizations during the entire clip.

评分引用的原句：

> s002: Initially, his eyebrows are raised and his forehead is wrinkled, creating a furrowed expression.

> s004: Around the midpoint, he briefly closes his eyes and lowers his gaze slightly before reopening them and looking forward again.

> s005: Toward the end of the clip, he tilts his head slightly to his right while keeping his eyes open and focused forward.

> s003: His eyes remain open and directed forward for most of the duration.


## naturalistic_V02_S5043_I00000069_local_26d534af4d65

**评分 3/5**：包含头部点头、闭眼后仰、发声反应及状态回归，行为变化明显且有重复特征

不确定项：模型未报告（不代表没有错误）

时长 4.135s；原始人脸p10/p50 155.7/162.8px；官方发声支持 0.627s；点头候选支持 0.720s。

The listener maintains a steady forward-facing posture throughout the clip. Initially, her head is upright and she looks directly ahead. Then, around the 1.5-second mark, she performs a single, distinct head nod, tilting her head downward and slightly forward before returning it to its original position. Following this, she briefly closes her eyes and tilts her head back slightly, holding that posture for a moment. She then opens her eyes and begins to speak, producing a short, low-pitched vocalization that sounds like "Mmm." After speaking, she returns to a neutral expression, keeping her head mostly still until the end of the clip.

评分引用的原句：

> s003: Then, around the 1.5-second mark, she performs a single, distinct head nod, tilting her head downward and slightly forward before returning it to its original position.

> s004: Following this, she briefly closes her eyes and tilts her head back slightly, holding that posture for a moment.

> s005: She then opens her eyes and begins to speak, producing a short, low-pitched vocalization that sounds like "Mmm." After speaking, she returns to a neutral expression, keeping her head mostly still until the end of the clip.

> s001: The listener maintains a steady forward-facing posture throughout the clip.


## naturalistic_V02_S3648_I00000032_local_fb8abbfe8ad2

**评分 3/5**：包含眼神闭合、点头和发声三种不同反应，且有明确变化

不确定项：模型未报告（不代表没有错误）

时长 7.392s；原始人脸p10/p50 151.2/152.4px；官方发声支持 0.412s；点头候选支持 1.200s。

The listener maintains a steady forward-facing posture throughout the clip, wearing a light blue baseball cap backward and a light-colored t-shirt against a plain gray background. Initially, their eyes are open and fixed on the camera. Around three seconds in, they briefly close their eyes before reopening them. At approximately four seconds, the listener performs a single, distinct head nod. Their eyes remain open for the rest of the clip, occasionally blinking. Towards the end, at about six seconds, they produce a short, audible vocalization that sounds like "Yeah."

评分引用的原句：

> s003: Around three seconds in, they briefly close their eyes before reopening them.

> s004: At approximately four seconds, the listener performs a single, distinct head nod.

> s006: Towards the end, at about six seconds, they produce a short, audible vocalization that sounds like "Yeah."

> s002: Initially, their eyes are open and fixed on the camera.


## naturalistic_V00_S0864_I00000382_local_6dceb9e9b771

**评分 3/5**：包含手势、闭眼抿嘴、持续凝视等多个明显动作变化，且有明确重复行为特征

不确定项：模型未报告（不代表没有错误）

时长 5.725s；原始人脸p10/p50 144.0/149.0px；官方发声支持 0.292s；点头候选支持 0.000s。

The listener begins by looking forward with a neutral expression. She then raises her right hand and places her index finger against her lips in a shushing gesture, holding it there for several seconds before lowering her hand. After a moment, she closes her eyes briefly and purses her lips, creating a slight pout. She opens her eyes again and maintains a steady gaze forward for the remainder of the clip.

评分引用的原句：

> s001: The listener begins by looking forward with a neutral expression.

> s002: She then raises her right hand and places her index finger against her lips in a shushing gesture, holding it there for several seconds before lowering her hand.

> s003: After a moment, she closes her eyes briefly and purses her lips, creating a slight pout.

> s004: She opens her eyes again and maintains a steady gaze forward for the remainder of the clip.


## improvised_V00_S1506_I00001013_local_de761b5fc733

**评分 3/5**：头部微倾与口部开合为一次明确反应，后续持续口动构成另一反应，两者有明显变化，且反应具持续性。

不确定项：模型未报告（不代表没有错误）

时长 2.571s；原始人脸p10/p50 140.8/141.5px；官方发声支持 0.316s；点头候选支持 0.651s。

The listener maintains a steady forward-facing posture throughout the clip, her head remaining mostly still with only slight shifts in orientation. Her facial expression is consistent, showing furrowed brows and a downturned mouth that persists for the entire duration. She wears large silver hoop earrings and a black crew-neck shirt, with a small microphone clipped near the collar. While no distinct hand or body movements are observed, she begins to speak around the two-second mark, producing a single, short utterance that sounds like "It." This vocalization occurs simultaneously with a subtle downward tilt of her head and a slight opening of her mouth. Following this, she continues speaking with her mouth moving visibly, though no further intelligible words are captured. The clip ends with her head tilted slightly downward and her gaze directed away from the camera.

评分引用的原句：

> s004: While no distinct hand or body movements are observed, she begins to speak around the two-second mark, producing a single, short utterance that sounds like "It." This vocalization occurs simultaneously with a subtle downward tilt of her head and a slight opening of her mouth.

> s005: Following this, she continues speaking with her mouth moving visibly, though no further intelligible words are captured.

> s002: Her facial expression is consistent, showing furrowed brows and a downturned mouth that persists for the entire duration.

> s006: The clip ends with her head tilted slightly downward and her gaze directed away from the camera.


## naturalistic_V00_S0760_I00000309_local_6edce73cfc56

**评分 3/5**：两次明确的点头动作，配合眼神和面部细微变化，体现多样且有意义的反应行为。

不确定项：模型未报告（不代表没有错误）

时长 9.858s；原始人脸p10/p50 139.4/140.4px；官方发声支持 0.634s；点头候选支持 1.218s。

The listener maintains a steady forward-facing posture throughout the clip, with her dark hair parted down the middle and falling around her shoulders. Initially, she looks slightly downward before lifting her gaze forward, where it remains for most of the duration. Her facial expression is generally neutral, with subtle shifts in her mouth and eye position that suggest ongoing listening or thought. At approximately seven seconds, she performs a single, distinct head nod, tilting her head downward and then back up. This nod is followed by another similar motion toward the end of the clip. Between these nods, her head remains relatively still. She does not speak, but a soft, breathy chuckle is audible near the beginning of the clip, after which there is no further vocalization.

评分引用的原句：

> s002: Initially, she looks slightly downward before lifting her gaze forward, where it remains for most of the duration.

> s004: At approximately seven seconds, she performs a single, distinct head nod, tilting her head downward and then back up.

> s005: This nod is followed by another similar motion toward the end of the clip.

> s006: Between these nods, her head remains relatively still.


## improvised_V01_S0296_I00001107_local_38a57f0ff7f9

**评分 2/5**：一个持续状态加一次明显反应变化，微笑持续存在，眼睑开合为一次明确变化。

不确定项：模型未报告（不代表没有错误）

时长 2.340s；原始人脸p10/p50 292.0/295.3px；官方发声支持 2.340s；点头候选支持 0.000s。

The listener maintains a consistent posture throughout the clip, facing forward with their head slightly tilted. Initially, they have a neutral expression with their eyes open and looking forward. Then, their eyes close briefly before reopening, and they display a subtle smile that persists for the remainder of the clip. Their head remains mostly still, with only minor shifts in position. The listener does not produce any audible vocalizations during the entire duration.

评分引用的原句：

> s002: Initially, they have a neutral expression with their eyes open and looking forward.

> s003: Then, their eyes close briefly before reopening, and they display a subtle smile that persists for the remainder of the clip.

> s004: Their head remains mostly still, with only minor shifts in position.

> s005: The listener does not produce any audible vocalizations during the entire duration.


## naturalistic_V03_S1788_I00000293_local_c5a3bf7e6ab6

**评分 2/5**：一次明确的点头动作，配合手部调整眼镜的单一行为，其余为静止状态。

不确定项：模型未报告（不代表没有错误）

时长 6.621s；原始人脸p10/p50 251.7/255.0px；官方发声支持 0.000s；点头候选支持 0.760s。

The listener, a person with long blonde hair and glasses wearing a blue hooded sweatshirt, is seated against a blue quilted background. Initially, they raise their right hand to adjust their glasses before lowering it and maintaining a steady forward gaze for the remainder of the clip. Their head nods downward once between approximately 0.5 and 1.3 seconds. The listener blinks several times throughout the clip but otherwise remains still, with no discernible changes in facial expression. There are no audible vocalizations, speech, or other sounds produced by the listener during the entire duration.

评分引用的原句：

> s002: Initially, they raise their right hand to adjust their glasses before lowering it and maintaining a steady forward gaze for the remainder of the clip.

> s003: Their head nods downward once between approximately 0.5 and 1.3 seconds.

> s004: The listener blinks several times throughout the clip but otherwise remains still, with no discernible changes in facial expression.

> s005: There are no audible vocalizations, speech, or other sounds produced by the listener during the entire duration.


## naturalistic_V03_S0877_I00000411_local_4d2ff70606d7

**评分 2/5**：一个持续状态加一次明确的点头反应，无其他明显变化。

不确定项：模型未报告（不代表没有错误）

时长 5.746s；原始人脸p10/p50 251.0/255.9px；官方发声支持 0.604s；点头候选支持 1.040s。

The listener, wearing a red hooded sweatshirt, black baseball cap, and glasses, maintains a steady forward-facing posture throughout the clip. Initially, his head is still, and he looks slightly upward before beginning to speak. Around two seconds in, he performs a single, distinct head nod, tilting his head down and then back up. He continues speaking for the remainder of the clip, his mouth moving visibly as he says, "Exactly." His gaze remains fixed forward during the utterance, and no other significant facial or bodily movements are observed.

评分引用的原句：

> s001: The listener, wearing a red hooded sweatshirt, black baseball cap, and glasses, maintains a steady forward-facing posture throughout the clip.

> s003: Around two seconds in, he performs a single, distinct head nod, tilting his head down and then back up.

> s004: He continues speaking for the remainder of the clip, his mouth moving visibly as he says, "Exactly." His gaze remains fixed forward during the utterance, and no other significant facial or bodily movements are observed.


## naturalistic_V03_S0268_I00000486_local_85eedccbc632

**评分 2/5**：一个持续状态加一次明确反应变化，反应重复性不明显

不确定项：模型未报告（不代表没有错误）

时长 9.566s；原始人脸p10/p50 246.4/247.5px；官方发声支持 0.000s；点头候选支持 0.840s。

The listener remains in a consistent seated position throughout the clip, facing forward with their head slightly tilted back against a blue quilted background. Initially, they maintain a neutral expression with their eyes open and fixed forward. Around the 0.56-second mark, they perform a single, brief head nod, lowering their chin slightly before returning to the original posture. For the remainder of the clip, their head stays relatively still, though subtle shifts in gaze occur, including a moment where their eyes briefly close and then reopen. Their mouth remains mostly closed, with only minor adjustments visible around the lips. No audible vocalizations or speech are detected at any point during the recording.

评分引用的原句：

> s001: The listener remains in a consistent seated position throughout the clip, facing forward with their head slightly tilted back against a blue quilted background.

> s002: Initially, they maintain a neutral expression with their eyes open and fixed forward.

> s003: Around the 0.56-second mark, they perform a single, brief head nod, lowering their chin slightly before returning to the original posture.

> s004: For the remainder of the clip, their head stays relatively still, though subtle shifts in gaze occur, including a moment where their eyes briefly close and then reopen.


## naturalistic_V03_S0877_I00000411_local_7e467a7be48c

**评分 2/5**：一个持续状态加一次明确反应，反应后回归静态，无其他变化或重复行为。

不确定项：模型未报告（不代表没有错误）

时长 6.684s；原始人脸p10/p50 202.1/204.5px；官方发声支持 0.412s；点头候选支持 0.400s。

The listener maintains a steady forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position. Her facial expression is consistent, showing a neutral look with slightly furrowed brows and a closed mouth. At approximately two and a half seconds into the clip, she performs a single, brief head nod. She then emits a short, low-pitched vocalization, "Mhm," before returning to her previous posture. The rest of the clip consists of her holding this same static pose and expression until the end.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position.

> s002: Her facial expression is consistent, showing a neutral look with slightly furrowed brows and a closed mouth.

> s003: At approximately two and a half seconds into the clip, she performs a single, brief head nod.


## improvised_V00_S0769_I00000377_local_d9a6e037ecfa

**评分 2/5**：一个明确的反应变化，包含闭眼和低头动作，且重复发生一次，但无其他明显变化。

不确定项：模型未报告（不代表没有错误）

时长 5.002s；原始人脸p10/p50 175.3/178.2px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a steady, forward-facing posture throughout the clip. His head remains mostly still, with only minor shifts in orientation. Initially, his eyes are open and directed slightly off-camera, and his expression is neutral. Then, he briefly closes his eyes and tilts his head downward before reopening his eyes and returning his gaze forward. This sequence of closing his eyes and lowering his head occurs once during the clip. He does not produce any audible vocalizations at any point.

评分引用的原句：

> s003: Initially, his eyes are open and directed slightly off-camera, and his expression is neutral.

> s004: Then, he briefly closes his eyes and tilts his head downward before reopening his eyes and returning his gaze forward.

> s005: This sequence of closing his eyes and lowering his head occurs once during the clip.

> s002: His head remains mostly still, with only minor shifts in orientation.


## naturalistic_V00_S0498_I00000495_local_794a0ccac190

**评分 2/5**：中性表情持续，一次眼神下移后恢复，末尾单次轻微点头，行为变化有限但有明确重复特征

不确定项：模型未报告（不代表没有错误）

时长 3.581s；原始人脸p10/p50 165.4/165.6px；官方发声支持 0.000s；点头候选支持 0.360s。

The listener maintains a steady forward gaze throughout the clip, his expression remaining neutral with his lips closed. His head remains mostly still, though he briefly lowers his eyes around the two-second mark before returning his focus forward. Toward the end of the clip, he performs a single, subtle downward nod of his head. There are no audible vocalizations during the entire duration.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, his expression remaining neutral with his lips closed.

> s002: His head remains mostly still, though he briefly lowers his eyes around the two-second mark before returning his focus forward.

> s003: Toward the end of the clip, he performs a single, subtle downward nod of his head.

> s004: There are no audible vocalizations during the entire duration.


## naturalistic_V02_S3631_I00000411_local_bb07a099ca84

**评分 2/5**：头部轻微倾斜后归中，一次明确点头，面部表情持续不变，变化有限。

不确定项：模型未报告（不代表没有错误）

时长 2.744s；原始人脸p10/p50 164.5/166.6px；官方发声支持 0.942s；点头候选支持 0.760s。

The listener maintains a steady gaze forward throughout the clip. Initially, her eyes look slightly upward and to her right before shifting to center. Her head remains mostly still, with a slight tilt back at first, then returning to a neutral position. Around the 1.9-second mark, she performs a single, distinct head nod. She continues to hold a consistent facial expression, characterized by a subtle smile and raised cheeks, until the end of the clip. The audio contains no discernible speech or vocalizations.

评分引用的原句：

> s002: Initially, her eyes look slightly upward and to her right before shifting to center.

> s003: Her head remains mostly still, with a slight tilt back at first, then returning to a neutral position.

> s004: Around the 1.9-second mark, she performs a single, distinct head nod.

> s005: She continues to hold a consistent facial expression, characterized by a subtle smile and raised cheeks, until the end of the clip.


## improvised_V00_S0769_I00000377_local_bbe4a4d914f6

**评分 2/5**：单一点头动作后持续说话，面部有轻微变化，但无重复行为或多重反应变化。

不确定项：caption称说话时有嘴部和眉部动作，但full_silence表明无音频，可能存在描述与实际不符。

时长 4.643s；原始人脸p10/p50 164.4/171.1px；官方发声支持 0.000s；点头候选支持 0.760s。

The listener maintains a consistent posture throughout the clip, facing forward with his head slightly tilted. Initially, he has a neutral expression with his eyes open and looking forward. Then, around the 1.4-second mark, he performs a single, brief nod by tilting his head downward and forward before returning to the initial position. Following the nod, he begins speaking, his mouth opening and closing rhythmically as he articulates words. His facial expression changes during speech, showing slight lip movements and subtle shifts in his eyebrows. The clip ends with him still in the process of speaking.

评分引用的原句：

> s003: Then, around the 1.4-second mark, he performs a single, brief nod by tilting his head downward and forward before returning to the initial position.

> s004: Following the nod, he begins speaking, his mouth opening and closing rhythmically as he articulates words.

> s005: His facial expression changes during speech, showing slight lip movements and subtle shifts in his eyebrows.

> s001: The listener maintains a consistent posture throughout the clip, facing forward with his head slightly tilted.


## naturalistic_V00_S0312_I00000520_local_4d9976487646

**评分 2/5**：一个持续注视状态加一次明确点头，随后有轻微视线转移，但无其他明显反应变化。

不确定项：模型未报告（不代表没有错误）

时长 3.803s；原始人脸p10/p50 160.3/168.7px；官方发声支持 0.000s；点头候选支持 0.400s。

The listener maintains a steady forward gaze throughout the clip. Initially, her head is still, and she blinks normally. Then, at approximately 1.5 seconds, she performs a single, distinct head nod, moving her head downward and slightly forward before returning to its original position. After the nod, she continues to look forward, blinking occasionally. Toward the end of the clip, she briefly shifts her gaze to her right before looking forward again. There are no other discernible movements of her hands, arms, or body, and there is no audible speech or vocalization.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip.

> s003: Then, at approximately 1.5 seconds, she performs a single, distinct head nod, moving her head downward and slightly forward before returning to its original position.

> s005: Toward the end of the clip, she briefly shifts her gaze to her right before looking forward again.

> s002: Initially, her head is still, and she blinks normally.


## improvised_V00_S0606_I00000797_local_7b1dc729764c

**评分 2/5**：持续的嘴部运动与单一音调的持续发声构成一种重复行为，面部表情略有变化但整体稳定，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 3.615s；原始人脸p10/p50 156.7/157.4px；官方发声支持 0.023s；点头候选支持 0.000s。

The listener maintains a steady forward-facing posture throughout the clip, with his head held upright and his gaze directed toward the camera. His facial expression remains relatively consistent, featuring slightly raised eyebrows and a neutral mouth position that shifts subtly between open and closed states. There is no discernible head movement such as nodding or shaking, nor any visible hand or body gestures. The listener's mouth moves continuously, producing a low-pitched, breathy vocalization that persists for the entire duration of the clip without interruption. This sound is sustained and does not vary significantly in pitch or intensity, resembling a continuous hum or soft exhalation. No other audible sounds, such as speech or laughter, are present.

评分引用的原句：

> s001: The listener maintains a steady forward-facing posture throughout the clip, with his head held upright and his gaze directed toward the camera.

> s002: His facial expression remains relatively consistent, featuring slightly raised eyebrows and a neutral mouth position that shifts subtly between open and closed states.

> s004: The listener's mouth moves continuously, producing a low-pitched, breathy vocalization that persists for the entire duration of the clip without interruption.

> s005: This sound is sustained and does not vary significantly in pitch or intensity, resembling a continuous hum or soft exhalation.


## naturalistic_V02_S3156_I00000386_local_0e5b708d3954

**评分 2/5**：一个持续状态加一次明确反应变化，后续保持单一手势不变。

不确定项：模型未报告（不代表没有错误）

时长 5.664s；原始人脸p10/p50 151.0/152.1px；官方发声支持 0.000s；点头候选支持 0.400s。

The listener maintains a steady upward gaze throughout the clip. Initially, their head is still, but then they perform a single, brief nod forward and down before returning to the original position. After the nod, the listener raises their right hand and covers their right eye and part of their nose with their palm, holding this gesture for the remainder of the clip. There are no audible vocalizations during the entire duration.

评分引用的原句：

> s001: The listener maintains a steady upward gaze throughout the clip.

> s002: Initially, their head is still, but then they perform a single, brief nod forward and down before returning to the original position.

> s003: After the nod, the listener raises their right hand and covers their right eye and part of their nose with their palm, holding this gesture for the remainder of the clip.

> s004: There are no audible vocalizations during the entire duration.


## naturalistic_V02_S3535_I00000399_local_d95bc0f39990

**评分 2/5**：一个持续状态加一次明确反应变化，后续持续微笑，无其他明显行为变化。

不确定项：模型未报告（不代表没有错误）

时长 7.243s；原始人脸p10/p50 149.3/150.6px；官方发声支持 0.000s；点头候选支持 0.720s。

The listener maintains a steady forward gaze throughout the clip. Initially, her expression is neutral with her lips closed. Then, around the three-second mark, she performs a single, brief head nod. Following this, her facial expression changes as she begins to smile, showing her upper teeth, and continues smiling for the remainder of the clip. There are no audible vocalizations at any point during the recording.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip.

> s003: Then, around the three-second mark, she performs a single, brief head nod.

> s004: Following this, her facial expression changes as she begins to smile, showing her upper teeth, and continues smiling for the remainder of the clip.

> s005: There are no audible vocalizations at any point during the recording.


## improvised_V00_S1249_I00001114_local_126b3bd30e76

**评分 2/5**：一次明显点头动作，伴随眼神微调，其余为静态或单一重复行为

不确定项：模型未报告（不代表没有错误）

时长 4.982s；原始人脸p10/p50 144.6/146.3px；官方发声支持 0.240s；点头候选支持 0.440s。

The listener maintains a steady forward gaze throughout the clip, her dark hair pulled back into a high bun with a few loose strands framing her face. She wears large hoop earrings and a pink fleece jacket with a purple collar. Initially, she looks slightly downward before lifting her eyes forward again. Around the 1.6-second mark, she performs a single, subtle head nod. Her expression remains largely neutral, with slight furrowing of the brow and occasional blinking. Towards the end of the clip, she briefly opens her mouth as if to speak, but no words are audible. The audio contains a single, clear utterance of the word "right" at approximately the 3.5-second mark.

评分引用的原句：

> s003: Initially, she looks slightly downward before lifting her eyes forward again.

> s004: Around the 1.6-second mark, she performs a single, subtle head nod.

> s005: Her expression remains largely neutral, with slight furrowing of the brow and occasional blinking.


## naturalistic_V03_S0476_I00000315_local_15fa8c6b5c9a

**评分 1/5**：仅描述一次手部动作和持续张口，无其他明显反应变化或重复行为。

不确定项：模型未报告（不代表没有错误）

时长 2.961s；原始人脸p10/p50 296.4/299.1px；官方发声支持 0.977s；点头候选支持 0.000s。

The listener maintains a consistent forward-facing posture throughout the clip. Initially, their right hand moves up toward their face, making contact near the nose before lowering again. The person's mouth remains open for the duration of the clip, showing continuous visible movement consistent with speaking. Their eyes stay directed forward, and no distinct head or body movements occur beyond the initial hand motion. There is no audible laughter or other nonverbal vocalization detected during the clip.

评分引用的原句：

> s002: Initially, their right hand moves up toward their face, making contact near the nose before lowering again.

> s003: The person's mouth remains open for the duration of the clip, showing continuous visible movement consistent with speaking.

> s001: The listener maintains a consistent forward-facing posture throughout the clip.

> s004: Their eyes stay directed forward, and no distinct head or body movements occur beyond the initial hand motion.


## naturalistic_V03_S1393_I00000111_local_76b117b06695

**评分 1/5**：仅描述了持续的面部表情和一次简短点头，无其他明显反应变化。

不确定项：模型未报告（不代表没有错误）

时长 7.919s；原始人脸p10/p50 295.8/300.1px；官方发声支持 0.000s；点头候选支持 0.840s。

The listener maintains a consistent posture throughout the clip, facing forward with their head held steady. Their eyes remain open and directed slightly upward and to the left for the entire duration. A subtle smile is present on their face from the beginning, persisting without noticeable change. Around the three-second mark, the listener performs a single, brief head nod, moving their head downward and then returning to its original position. No other distinct facial movements, hand gestures, or body shifts occur during the clip. The listener remains silent for the full duration, with no audible speech or vocalizations detected.

评分引用的原句：

> s001: The listener maintains a consistent posture throughout the clip, facing forward with their head held steady.

> s003: A subtle smile is present on their face from the beginning, persisting without noticeable change.

> s004: Around the three-second mark, the listener performs a single, brief head nod, moving their head downward and then returning to its original position.

> s005: No other distinct facial movements, hand gestures, or body shifts occur during the clip.


## naturalistic_V03_S1532_I00000036_local_0a47cb3ab2a1

**评分 1/5**：仅描述一次短暂点头及持续静态表情，无其他明显反应行为变化。

不确定项：模型未报告（不代表没有错误）

时长 5.322s；原始人脸p10/p50 261.1/262.9px；官方发声支持 0.000s；点头候选支持 0.600s。

The listener begins by looking slightly upward and to their right, with a faint smile on their face. They then perform a single, brief nod of the head around the first second of the clip. After the nod, they maintain a steady gaze directed toward their right, holding a consistent facial expression that includes a slight smile and relaxed brow for the remainder of the clip. There are no other discernible head movements, hand gestures, or body shifts throughout the duration. The person remains visually still and silent, with no audible vocalizations or speech detected.

评分引用的原句：

> s001: The listener begins by looking slightly upward and to their right, with a faint smile on their face.

> s002: They then perform a single, brief nod of the head around the first second of the clip.

> s003: After the nod, they maintain a steady gaze directed toward their right, holding a consistent facial expression that includes a slight smile and relaxed brow for the remainder of the clip.

> s004: There are no other discernible head movements, hand gestures, or body shifts throughout the duration.


## naturalistic_V03_S1421_I00000117_local_674f523d111d

**评分 1/5**：仅描述持续微笑和一次眼神变化，无其他明显行为变化或重复

不确定项：模型未报告（不代表没有错误）

时长 3.701s；原始人脸p10/p50 220.8/229.1px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a consistent smile throughout the clip, her gaze directed downward for the first two seconds before lifting to look forward. Her head remains mostly still, with only slight shifts in position, and she blinks naturally at intervals. There is no discernible nodding or hand movement, and her posture appears relaxed. The audio contains no speech or vocalizations, only silence.

评分引用的原句：

> s001: The listener maintains a consistent smile throughout the clip, her gaze directed downward for the first two seconds before lifting to look forward.

> s002: Her head remains mostly still, with only slight shifts in position, and she blinks naturally at intervals.

> s003: There is no discernible nodding or hand movement, and her posture appears relaxed.

> s004: The audio contains no speech or vocalizations, only silence.


## naturalistic_V03_S0782_I00000059_local_7f5d8964d753

**评分 1/5**：仅描述一次短暂点头，后续为静态凝视与轻微眨眼，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 10.746s；原始人脸p10/p50 219.6/221.5px；官方发声支持 0.000s；点头候选支持 0.800s。

The listener remains seated and facing forward throughout the clip. Initially, his head is still and he looks directly at the camera. Around three seconds in, he performs a single, brief nod by lowering his head slightly and then raising it back to its original position. After the nod, he continues to look forward, blinking occasionally and making small, subtle shifts in his gaze. His facial expression stays relatively neutral for the duration of the clip. There are no audible vocalizations or speech from the listener during the entire segment.

评分引用的原句：

> s001: The listener remains seated and facing forward throughout the clip.

> s003: Around three seconds in, he performs a single, brief nod by lowering his head slightly and then raising it back to its original position.

> s004: After the nod, he continues to look forward, blinking occasionally and making small, subtle shifts in his gaze.

> s005: His facial expression stays relatively neutral for the duration of the clip.


## naturalistic_V02_S3601_I00000156_local_719d6c438834

**评分 1/5**：仅描述一次开口后闭合的面部变化，后续全程静止，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 3.002s；原始人脸p10/p50 170.4/172.0px；官方发声支持 0.429s；点头候选支持 0.000s。

The listener maintains a steady forward gaze throughout the clip, their head held upright and still against a plain gray background. Their dark hair frames their face, and they wear large silver hoop earrings and a denim shirt. Initially, their mouth is slightly open, then closes into a neutral expression that persists for most of the duration. While there is no visible nodding, the listener produces a single, clear utterance of "Yeah" at the beginning of the clip. The rest of the time, they remain visually still, with no discernible changes in posture or facial expression.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, their head held upright and still against a plain gray background.

> s003: Initially, their mouth is slightly open, then closes into a neutral expression that persists for most of the duration.

> s004: While there is no visible nodding, the listener produces a single, clear utterance of "Yeah" at the beginning of the clip.

> s005: The rest of the time, they remain visually still, with no discernible changes in posture or facial expression.


## naturalistic_V00_S0312_I00000520_local_0ac15af8dcd6

**评分 1/5**：仅描述了持续凝视和表情轻微变化，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 3.323s；原始人脸p10/p50 163.8/166.5px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a steady forward gaze throughout the clip, her expression shifting subtly from a slight smile to a more neutral, attentive look. Her head remains mostly still, with only minor adjustments in angle. She wears a green top, a gold chain necklace, and small hoop earrings, and a microphone is clipped to her collar. There is no audible speech or vocalization during the entire duration.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, her expression shifting subtly from a slight smile to a more neutral, attentive look.

> s002: Her head remains mostly still, with only minor adjustments in angle.

> s003: She wears a green top, a gold chain necklace, and small hoop earrings, and a microphone is clipped to her collar.

> s004: There is no audible speech or vocalization during the entire duration.


## naturalistic_V02_S3373_I00000414_local_e5a4ed2fdefb

**评分 1/5**：仅描述一次短暂点头和持续凝视，后续为静态闭眼状态，无其他明显反应变化。

不确定项：模型未报告（不代表没有错误）

时长 6.920s；原始人脸p10/p50 158.5/161.5px；官方发声支持 0.000s；点头候选支持 0.480s。

The listener maintains a steady upward gaze throughout the clip. Initially, he performs a single, brief head nod. His eyes remain open for several seconds before closing completely around the three-second mark. While his eyes are closed, his facial expression remains unchanged, showing no visible signs of smiling or frowning. He holds this still posture with closed eyes until the end of the clip. There are no audible vocalizations at any point during the recording.

评分引用的原句：

> s001: The listener maintains a steady upward gaze throughout the clip.

> s002: Initially, he performs a single, brief head nod.

> s003: His eyes remain open for several seconds before closing completely around the three-second mark.

> s005: He holds this still posture with closed eyes until the end of the clip.


## improvised_V00_S2055_I00001096_local_729e50c3c598

**评分 1/5**：仅描述一次短暂点头，其余为静态下视姿态，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 2.716s；原始人脸p10/p50 156.6/165.5px；官方发声支持 0.000s；点头候选支持 0.560s。

The listener maintains a downward gaze for most of the clip, with their head tilted slightly forward. Around the 1.6-second mark, they perform a single, brief nod by moving their head down and up. After this, they return to looking down, holding that posture until the end of the clip. There are no other discernible head movements, facial expressions, or body motions throughout the sequence. The listener remains silent for the entire duration.

评分引用的原句：

> s001: The listener maintains a downward gaze for most of the clip, with their head tilted slightly forward.

> s002: Around the 1.6-second mark, they perform a single, brief nod by moving their head down and up.

> s003: After this, they return to looking down, holding that posture until the end of the clip.

> s004: There are no other discernible head movements, facial expressions, or body motions throughout the sequence.


## naturalistic_V02_S3535_I00000399_local_2dedff83681f

**评分 1/5**：仅描述一次短暂点头，其余为静态姿态，无其他明显反应行为。

不确定项：模型未报告（不代表没有错误）

时长 6.163s；原始人脸p10/p50 156.3/157.5px；官方发声支持 0.000s；点头候选支持 0.280s。

The listener maintains a steady forward gaze throughout the clip, his head held upright with minimal movement. His facial expression remains largely unchanged, featuring a neutral mouth position and relaxed brow. At approximately four seconds into the clip, he performs a single, brief downward nod of the head before returning to his initial posture. This nod occurs while his eyes remain open and directed forward. No other discernible movements of the head, eyes, or body take place during the remainder of the clip. The listener does not produce any audible vocalizations, including speech or laughter, for the entire duration.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, his head held upright with minimal movement.

> s003: At approximately four seconds into the clip, he performs a single, brief downward nod of the head before returning to his initial posture.

> s005: No other discernible movements of the head, eyes, or body take place during the remainder of the clip.

> s006: The listener does not produce any audible vocalizations, including speech or laughter, for the entire duration.


## improvised_V00_S0606_I00000797_local_fa36d6a09bb2

**评分 1/5**：仅描述持续的静止姿态和单一面部表情，无明显变化或重复行为。

不确定项：模型未报告（不代表没有错误）

时长 2.562s；原始人脸p10/p50 152.4/152.7px；官方发声支持 2.562s；点头候选支持 0.000s。

The listener maintains a steady forward-facing posture throughout the clip, his head remaining mostly still with only slight shifts in position. His facial expression is consistent, featuring raised eyebrows and a slightly parted mouth that reveals his upper teeth, creating a look of focused attention or mild surprise. There is no discernible nodding or other repetitive head movement observed. The listener does not produce any audible vocalizations during the entire duration of the clip.

评分引用的原句：

> s001: The listener maintains a steady forward-facing posture throughout the clip, his head remaining mostly still with only slight shifts in position.

> s002: His facial expression is consistent, featuring raised eyebrows and a slightly parted mouth that reveals his upper teeth, creating a look of focused attention or mild surprise.

> s003: There is no discernible nodding or other repetitive head movement observed.

> s004: The listener does not produce any audible vocalizations during the entire duration of the clip.


## naturalistic_V00_S0155_I00000126_local_2d4cac53ee97

**评分 1/5**：仅描述了持续的中性表情和一次轻微点头，无其他明显变化或重复行为。

不确定项：模型未报告（不代表没有错误）

时长 10.808s；原始人脸p10/p50 152.1/154.1px；官方发声支持 0.000s；点头候选支持 0.920s。

The listener maintains a steady forward gaze throughout the clip, his head remaining mostly still with only slight shifts in posture. His facial expression is consistent, showing a neutral look with a closed mouth and relaxed features. Toward the end of the clip, he performs a single, subtle head nod. There are no audible vocalizations during the entire duration.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, his head remaining mostly still with only slight shifts in posture.

> s002: His facial expression is consistent, showing a neutral look with a closed mouth and relaxed features.

> s003: Toward the end of the clip, he performs a single, subtle head nod.

> s004: There are no audible vocalizations during the entire duration.


## naturalistic_V02_S3407_I00000728_local_508b67af7f08

**评分 1/5**：仅描述一次短暂点头和持续静止状态，无其他明显反应变化。

不确定项：模型未报告（不代表没有错误）

时长 5.925s；原始人脸p10/p50 149.8/153.7px；官方发声支持 0.000s；点头候选支持 0.480s。

The listener maintains a consistent upward gaze throughout the clip. Initially, he performs a single, brief nod of his head. Following this, he holds a steady posture with his head tilted slightly back and his mouth closed, displaying a subtle smile that persists for the remainder of the recording. There are no other discernible facial movements, head shifts, or hand gestures observed. The listener remains visually still and silent for the entire duration, with no audible vocalizations or speech detected.

评分引用的原句：

> s001: The listener maintains a consistent upward gaze throughout the clip.

> s002: Initially, he performs a single, brief nod of his head.

> s003: Following this, he holds a steady posture with his head tilted slightly back and his mouth closed, displaying a subtle smile that persists for the remainder of the recording.

> s004: There are no other discernible facial movements, head shifts, or hand gestures observed.


## improvised_V01_S0434_I00001240_local_ed364987637f

**评分 1/5**：仅描述持续的固定凝视与紧张面部表情，无其他变化或重复行为。

不确定项：模型未报告（不代表没有错误）

时长 10.723s；原始人脸p10/p50 141.0/142.4px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a fixed forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position. Her facial expression is tense, featuring furrowed brows and a downturned mouth, which persists for the entire duration. There are no discernible hand, arm, or body movements observed. The listener does not produce any audible vocalizations, including speech or nonverbal sounds like laughter, and remains silent for the full length of the clip.

评分引用的原句：

> s001: The listener maintains a fixed forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position.

> s002: Her facial expression is tense, featuring furrowed brows and a downturned mouth, which persists for the entire duration.

> s003: There are no discernible hand, arm, or body movements observed.

> s004: The listener does not produce any audible vocalizations, including speech or nonverbal sounds like laughter, and remains silent for the full length of the clip.


## improvised_V00_S1979_I00001011_local_2038013807d9

**评分 1/5**：仅描述持续凝视与静态姿态，虽提及点头但未见视觉表现，无其他明显反应行为。

不确定项：补充信息称有点头但视觉不可见，存在矛盾。

时长 2.441s；原始人脸p10/p50 134.1/135.3px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His facial expression remains largely neutral, with his lips closed and eyebrows relaxed. There is no discernible movement of his hands or body. The provided supplementary information indicates that he nods at certain points during the clip, but these specific instances are not visually apparent in the available frames. No audible vocalizations, such as speech or laughter, are present.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, his head held upright and still.

> s002: His facial expression remains largely neutral, with his lips closed and eyebrows relaxed.

> s003: There is no discernible movement of his hands or body.

> s004: The provided supplementary information indicates that he nods at certain points during the clip, but these specific instances are not visually apparent in the available frames.


## naturalistic_V02_S3399_I00000729_local_6144bcc8c1bc

**评分 0/5**：全程无明显表情或动作变化，仅描述静态姿态与无声状态

不确定项：模型未报告（不代表没有错误）

时长 2.321s；原始人脸p10/p50 147.6/149.1px；官方发声支持 0.000s；点头候选支持 0.000s。

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His dark eyes remain fixed on a point ahead, showing no discernible shift in focus or direction. The thick, dark beard framing his face remains motionless, and there is no visible movement in his shoulders or upper body. He wears a light-colored knit beanie that sits snugly on his head, and his expression stays neutral with lips closed. There are no audible vocalizations, laughter, or speech detected during the entire duration.

评分引用的原句：

> s001: The listener maintains a steady forward gaze throughout the clip, his head held upright and still.

> s002: His dark eyes remain fixed on a point ahead, showing no discernible shift in focus or direction.

> s003: The thick, dark beard framing his face remains motionless, and there is no visible movement in his shoulders or upper body.

> s004: He wears a light-colored knit beanie that sits snugly on his head, and his expression stays neutral with lips closed.
