# 同一50段：原文与四版行为提取对照

本页仅比较文本提取，不是视频动作真值。v4通过结构校验不代表语义完整；所有原文和模型回复保留。

## improvised_V00_S0290_I00000479_local_addfd726e77f

The listener maintains a consistent downward gaze throughout the clip, her head tilted slightly forward and remaining mostly still. Her facial expression is neutral, with her lips closed and eyebrows relaxed. At approximately two seconds, she performs a single, brief head nod, tilting her head down and up quickly before returning to her initial posture. She repeats this nod around seven seconds, again moving her head down and up in a short, distinct motion. Aside from these two instances, there is no other discernible head or body movement. The listener does not speak or make any audible vocalizations during the entire duration.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'repeats this nod'}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze direction downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lip closing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyebrow relaxation · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalization · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Invalid relation**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s003', 's004']}
- gaze downward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows relaxed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s003', 's004']}
- facial expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003', 's004']}
- lips closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows relaxed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S0606_I00000797_local_7b1dc729764c

The listener maintains a steady forward-facing posture throughout the clip, with his head held upright and his gaze directed toward the camera. His facial expression remains relatively consistent, featuring slightly raised eyebrows and a neutral mouth position that shifts subtly between open and closed states. There is no discernible head movement such as nodding or shaking, nor any visible hand or body gestures. The listener's mouth moves continuously, producing a low-pitched, breathy vocalization that persists for the entire duration of the clip without interruption. This sound is sustained and does not vary significantly in pitch or intensity, resembling a continuous hum or soft exhalation. No other audible sounds, such as speech or laughter, are present.

**v1：结构通过**

- upright head · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze directed toward camera · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slightly raised eyebrows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral mouth position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'subtly between open and closed states'}
- mouth closing · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'subtly between open and closed states'}
- continuous mouth movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- low-pitched breathy vocalization · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- sustained vocalization · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- vocalization · audio · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head upright · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows slightly raised · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth moving · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- humming · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- shaking head · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand gesture · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body gesture · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughing · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S0606_I00000797_local_af4f5a40d6b5

The listener maintains a steady forward-facing posture throughout the clip. Initially, his eyebrows are raised and his forehead is wrinkled, creating a furrowed expression. His eyes remain open and directed forward for most of the duration. Around the midpoint, he briefly closes his eyes and lowers his gaze slightly before reopening them and looking forward again. Toward the end of the clip, he tilts his head slightly to his right while keeping his eyes open and focused forward. There are no audible vocalizations during the entire clip.

**v1：Evidence must be an exact source substring**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyebrow raising · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- forehead wrinkling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye reopening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- audible vocalization · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- forehead wrinkling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- forehead wrinkling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- maintaining forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows raised · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- forehead wrinkled · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze directed forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze lowered · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted to right · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S0606_I00000797_local_fa36d6a09bb2

The listener maintains a steady forward-facing posture throughout the clip, his head remaining mostly still with only slight shifts in position. His facial expression is consistent, featuring raised eyebrows and a slightly parted mouth that reveals his upper teeth, creating a look of focused attention or mild surprise. There is no discernible nodding or other repetitive head movement observed. The listener does not produce any audible vocalizations during the entire duration of the clip.

**v1：结构通过**

- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- raised eyebrows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slightly parted mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no nodding · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raised eyebrows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- slightly parted mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raised eyebrows · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- slightly parted mouth · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S0769_I00000377_local_bbe4a4d914f6

The listener maintains a consistent posture throughout the clip, facing forward with his head slightly tilted. Initially, he has a neutral expression with his eyes open and looking forward. Then, around the 1.4-second mark, he performs a single, brief nod by tilting his head downward and forward before returning to the initial position. Following the nod, he begins speaking, his mouth opening and closing rhythmically as he articulates words. His facial expression changes during speech, showing slight lip movements and subtle shifts in his eyebrows. The clip ends with him still in the process of speaking.

**v1：Explicit count lacks literal numerical evidence**

- head tilting · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- mouth opening · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'rhythmically'}
- eyebrow movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'subtle shifts'}
- lip movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'slight lip movements'}

**v2：结构通过**

- head tilting · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip movements · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow shifts · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speaking · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S0769_I00000377_local_d9a6e037ecfa

The listener maintains a steady, forward-facing posture throughout the clip. His head remains mostly still, with only minor shifts in orientation. Initially, his eyes are open and directed slightly off-camera, and his expression is neutral. Then, he briefly closes his eyes and tilts his head downward before reopening his eyes and returning his gaze forward. This sequence of closing his eyes and lowering his head occurs once during the clip. He does not produce any audible vocalizations at any point.

**v1：Evidence must be an exact source substring**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head orientation · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye openness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze direction · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closure · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilt · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze return · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- vocalization · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Count evidence must also support the behavior**

- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s005']}
- head tilting · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s005']}

**v3：Count evidence must also support the behavior**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s005']}
- eye closing · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s005']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s005']}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze returning forward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S1249_I00001114_local_126b3bd30e76

The listener maintains a steady forward gaze throughout the clip, her dark hair pulled back into a high bun with a few loose strands framing her face. She wears large hoop earrings and a pink fleece jacket with a purple collar. Initially, she looks slightly downward before lifting her eyes forward again. Around the 1.6-second mark, she performs a single, subtle head nod. Her expression remains largely neutral, with slight furrowing of the brow and occasional blinking. Towards the end of the clip, she briefly opens her mouth as if to speak, but no words are audible. The audio contains a single, clear utterance of the word "right" at approximately the 3.5-second mark.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single'}
- brow furrowing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasional'}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single'}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s007']}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- brow furrowing · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s007']}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow furrowing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S1506_I00001013_local_de761b5fc733

The listener maintains a steady forward-facing posture throughout the clip, her head remaining mostly still with only slight shifts in orientation. Her facial expression is consistent, showing furrowed brows and a downturned mouth that persists for the entire duration. She wears large silver hoop earrings and a black crew-neck shirt, with a small microphone clipped near the collar. While no distinct hand or body movements are observed, she begins to speak around the two-second mark, producing a single, short utterance that sounds like "It." This vocalization occurs simultaneously with a subtle downward tilt of her head and a slight opening of her mouth. Following this, she continues speaking with her mouth moving visibly, though no further intelligible words are captured. The clip ends with her head tilted slightly downward and her gaze directed away from the camera.

**v1：Explicit count lacks literal numerical evidence**

- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- furrowed brows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- downturned mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- subtle downward tilt of head · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slight opening of mouth · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth moving visibly · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- speaking · audio · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single, short utterance'}
- head tilted slightly downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze directed away from camera · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- steady forward-facing posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- slight head shifts · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- furrowed brows · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- downturned mouth · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- downward head tilt · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze directed away · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- furrowed brows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- downturned mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S1979_I00001011_local_2038013807d9

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His facial expression remains largely neutral, with his lips closed and eyebrows relaxed. There is no discernible movement of his hands or body. The provided supplementary information indicates that he nods at certain points during the clip, but these specific instances are not visually apparent in the available frames. No audible vocalizations, such as speech or laughter, are present.

**v1：结构通过**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head upright · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- facial expression neutral · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lips closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyebrows relaxed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no hand movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no body movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- humming · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head upright · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrows relaxed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hands still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V00_S2055_I00001096_local_729e50c3c598

The listener maintains a downward gaze for most of the clip, with their head tilted slightly forward. Around the 1.6-second mark, they perform a single, brief nod by moving their head down and up. After this, they return to looking down, holding that posture until the end of the clip. There are no other discernible head movements, facial expressions, or body motions throughout the sequence. The listener remains silent for the entire duration.

**v1：Explicit count lacks literal numerical evidence**

- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze direction downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- gaze direction downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- silence · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head tilting · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- looking down · visual · positive · baseline · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s001', 's003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- downward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- silent · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V01_S0296_I00001107_local_38a57f0ff7f9

The listener maintains a consistent posture throughout the clip, facing forward with their head slightly tilted. Initially, they have a neutral expression with their eyes open and looking forward. Then, their eyes close briefly before reopening, and they display a subtle smile that persists for the remainder of the clip. Their head remains mostly still, with only minor shifts in position. The listener does not produce any audible vocalizations during the entire duration.

**v1：结构通过**

- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Invalid source sentence IDs**

- head tilting · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes open · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes open · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V01_S0434_I00001240_local_ed364987637f

The listener maintains a fixed forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position. Her facial expression is tense, featuring furrowed brows and a downturned mouth, which persists for the entire duration. There are no discernible hand, arm, or body movements observed. The listener does not produce any audible vocalizations, including speech or nonverbal sounds like laughter, and remains silent for the full length of the clip.

**v1：结构通过**

- fixed forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- furrowed brows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- downturned mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no hand movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- gaze · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- brow furrowing · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- speech · audio · negative · baseline · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- nonverbal vocalization · audio · negative · baseline · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- arm movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- furrowed brows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- downturned mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no hand movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no arm movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no body movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## improvised_V01_S1827_I00000704_local_89d8db1b77e0

The listener maintains a steady forward-facing posture throughout the clip. Initially, he displays a broad smile with his teeth showing, and his head remains still. Around the 0.9-second mark, he performs a single, brief nod by tilting his head downward and back up. Following this, he holds a neutral expression with his lips closed for several seconds before shifting his gaze upward and to his right. His mouth opens wide in a sustained expression resembling a laugh or exclamation, continuing until the end of the clip.

**v1：Explicit count lacks literal numerical evidence**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze upward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0155_I00000126_local_2d4cac53ee97

The listener maintains a steady forward gaze throughout the clip, his head remaining mostly still with only slight shifts in posture. His facial expression is consistent, showing a neutral look with a closed mouth and relaxed features. Toward the end of the clip, he performs a single, subtle head nod. There are no audible vocalizations during the entire duration.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slight posture shift · visual · uncertain · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'slight shifts'}
- neutral facial expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- closed mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- relaxed facial features · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- subtle head nod · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single'}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gazing · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0312_I00000520_local_0ac15af8dcd6

The listener maintains a steady forward gaze throughout the clip, her expression shifting subtly from a slight smile to a more neutral, attentive look. Her head remains mostly still, with only minor adjustments in angle. She wears a green top, a gold chain necklace, and small hoop earrings, and a microphone is clipped to her collar. There is no audible speech or vocalization during the entire duration.

**v1：结构通过**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slight smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- minor head angle adjustment · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'minor adjustments in angle'}

**v2：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0312_I00000520_local_4d9976487646

The listener maintains a steady forward gaze throughout the clip. Initially, her head is still, and she blinks normally. Then, at approximately 1.5 seconds, she performs a single, distinct head nod, moving her head downward and slightly forward before returning to its original position. After the nod, she continues to look forward, blinking occasionally. Toward the end of the clip, she briefly shifts her gaze to her right before looking forward again. There are no other discernible movements of her hands, arms, or body, and there is no audible speech or vocalization.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- normal blinking · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- continued forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- occasional blinking · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}
- gaze shift to right · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002', 's004']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · baseline · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s001', 's004']}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002', 's004']}
- gaze holding · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0498_I00000495_local_794a0ccac190

The listener maintains a steady forward gaze throughout the clip, his expression remaining neutral with his lips closed. His head remains mostly still, though he briefly lowers his eyes around the two-second mark before returning his focus forward. Toward the end of the clip, he performs a single, subtle downward nod of his head. There are no audible vocalizations during the entire duration.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lips closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head still · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes lowered · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- downward nod · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- steady forward gaze · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze holding · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes lowered · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0552_I00000371_local_6346662ba47c

The listener maintains a consistent forward-facing posture throughout the clip. Initially, she looks slightly down and to her right with a faint smile. Around two seconds in, she briefly closes her eyes and nods her head downward once. She then opens her eyes and continues looking forward with a neutral expression. Towards the end of the clip, she makes a soft, audible "hmm" sound while pursing her lips slightly.

**v1：结构通过**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking down · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- closing eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'once'}
- opening eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- pursing lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- audible hmm · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip pursing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip pursing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking down · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking to the right · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- closing eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding head downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- opening eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- pursing lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- making a sound · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0552_I00000371_local_e9f5a66941a5

The listener maintains a consistent forward-facing posture throughout the clip, with long dark hair framing her face and small hoop earrings visible. Initially, she has a neutral expression with her lips closed, then begins to smile slightly while looking forward. Her head nods downward twice during the clip, first between approximately two and three seconds, and again near six seconds, each time accompanied by a subtle shift in facial expression. As the clip progresses, she continues to smile more broadly, showing her teeth, and her mouth opens slightly as she speaks the word "Yeah" at around four seconds. She repeats the word "Yeah" again shortly after, maintaining eye contact and a steady upper body position.

**v1：Evidence must be an exact source substring**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'quote': 'twice'}
- broad smile · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- speaking · audio · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'quote': 'again'}
- eye contact · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- steady upper body position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Invalid source sentence IDs**

- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s004', 's005']}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- speech · audio · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speaking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0725_I00000125_local_ffbb8d3f7806

The listener maintains a steady forward-facing posture throughout the clip, his head remaining mostly still with only slight shifts in position. Initially, he looks slightly downward before lifting his gaze forward, where it remains for most of the duration. His facial expression transitions from neutral to a subtle smile that persists until the end of the clip. The corners of his mouth turn upward, and his lips part slightly, revealing his upper teeth in a gentle grin. There is no discernible nodding or hand movement observed. Audibly, he says "Yeah" at the beginning of the clip, followed by a period of silence.

**v1：Evidence must be an exact source substring**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slight head shift · visual · uncertain · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'slight shifts in position'}
- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral facial expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- subtle smile · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth corner upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lips parting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- upper teeth visible · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no nodding · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no hand movement · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- saying yeah · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- silence · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth parting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0760_I00000309_local_6edce73cfc56

The listener maintains a steady forward-facing posture throughout the clip, with her dark hair parted down the middle and falling around her shoulders. Initially, she looks slightly downward before lifting her gaze forward, where it remains for most of the duration. Her facial expression is generally neutral, with subtle shifts in her mouth and eye position that suggest ongoing listening or thought. At approximately seven seconds, she performs a single, distinct head nod, tilting her head downward and then back up. This nod is followed by another similar motion toward the end of the clip. Between these nods, her head remains relatively still. She does not speak, but a soft, breathy chuckle is audible near the beginning of the clip, after which there is no further vocalization.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'another'}
- looking downward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- breathy chuckling · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Invalid source sentence IDs**

- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughing · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004', 's005']}
- smiling · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V00_S0864_I00000382_local_6dceb9e9b771

The listener begins by looking forward with a neutral expression. She then raises her right hand and places her index finger against her lips in a shushing gesture, holding it there for several seconds before lowering her hand. After a moment, she closes her eyes briefly and purses her lips, creating a slight pout. She opens her eyes again and maintains a steady gaze forward for the remainder of the clip.

**v1：结构通过**

- looking forward · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- finger touching lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lip pursing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze maintaining · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raising hand · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- placing finger against lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- closing eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- pursing lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- opening eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting fingers · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip pursing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze holding · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral expression · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raising right hand · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- placing index finger against lips · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- holding hand in shushing gesture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lowering hand · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- closing eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- pursing lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- opening eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3156_I00000386_local_0e5b708d3954

The listener maintains a steady upward gaze throughout the clip. Initially, their head is still, but then they perform a single, brief nod forward and down before returning to the original position. After the nod, the listener raises their right hand and covers their right eye and part of their nose with their palm, holding this gesture for the remainder of the clip. There are no audible vocalizations during the entire duration.

**v1：Evidence must be an exact source substring**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- gaze upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand covering eye and nose · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- gazing upward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand covering eye and nose · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand covering · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- gaze upward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand covering eye and nose · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3373_I00000414_local_e5a4ed2fdefb

The listener maintains a steady upward gaze throughout the clip. Initially, he performs a single, brief head nod. His eyes remain open for several seconds before closing completely around the three-second mark. While his eyes are closed, his facial expression remains unchanged, showing no visible signs of smiling or frowning. He holds this still posture with closed eyes until the end of the clip. There are no audible vocalizations at any point during the recording.

**v1：Explicit count lacks literal numerical evidence**

- steady upward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- eye opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- facial expression stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no smiling · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no frowning · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- posture stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- gazing upward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression unchanged · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking upward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression unchanged · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3399_I00000729_local_6144bcc8c1bc

The listener maintains a steady forward gaze throughout the clip, his head held upright and still. His dark eyes remain fixed on a point ahead, showing no discernible shift in focus or direction. The thick, dark beard framing his face remains motionless, and there is no visible movement in his shoulders or upper body. He wears a light-colored knit beanie that sits snugly on his head, and his expression stays neutral with lips closed. There are no audible vocalizations, laughter, or speech detected during the entire duration.

**v1：结构通过**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head upright · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes fixed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no shift in focus · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- beard motionless · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no shoulder movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- knit beanie in place · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lips closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- gaze · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- beard movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- shoulder movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- beanie movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- beard movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- shoulder movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- beard motionless · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- shoulders still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- upper body still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- wearing beanie · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- expression neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lips closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3407_I00000728_local_508b67af7f08

The listener maintains a consistent upward gaze throughout the clip. Initially, he performs a single, brief nod of his head. Following this, he holds a steady posture with his head tilted slightly back and his mouth closed, displaying a subtle smile that persists for the remainder of the recording. There are no other discernible facial movements, head shifts, or hand gestures observed. The listener remains visually still and silent for the entire duration, with no audible vocalizations or speech detected.

**v1：Explicit count lacks literal numerical evidence**

- upward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single'}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth closing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- visual stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- silence · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand gesture · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head shift · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand gesture · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking upward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted back · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- subtle smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3535_I00000399_local_2dedff83681f

The listener maintains a steady forward gaze throughout the clip, his head held upright with minimal movement. His facial expression remains largely unchanged, featuring a neutral mouth position and relaxed brow. At approximately four seconds into the clip, he performs a single, brief downward nod of the head before returning to his initial posture. This nod occurs while his eyes remain open and directed forward. No other discernible movements of the head, eyes, or body take place during the remainder of the clip. The listener does not produce any audible vocalizations, including speech or laughter, for the entire duration.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head upright · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral mouth position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- relaxed brow · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- downward head nod · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- forward eye gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- eyes open · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral mouth · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- relaxed brow · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting glasses · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- humming · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze holding · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye openness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow relaxed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3535_I00000399_local_d95bc0f39990

The listener maintains a steady forward gaze throughout the clip. Initially, her expression is neutral with her lips closed. Then, around the three-second mark, she performs a single, brief head nod. Following this, her facial expression changes as she begins to smile, showing her upper teeth, and continues smiling for the remainder of the clip. There are no audible vocalizations at any point during the recording.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- lips · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral expression · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3601_I00000156_local_719d6c438834

The listener maintains a steady forward gaze throughout the clip, their head held upright and still against a plain gray background. Their dark hair frames their face, and they wear large silver hoop earrings and a denim shirt. Initially, their mouth is slightly open, then closes into a neutral expression that persists for most of the duration. While there is no visible nodding, the listener produces a single, clear utterance of "Yeah" at the beginning of the clip. The rest of the time, they remain visually still, with no discernible changes in posture or facial expression.

**v1：Explicit count lacks literal numerical evidence**

- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head upright · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth slightly open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no nodding · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- utterance · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}

**v2：Incomplete model generation**

输出截断，原始回复见对应attempts文件。
**v3：结构通过**

- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyebrow raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting glasses · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- mouth opening · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- posture change · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression change · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3631_I00000411_local_4c5694a3adc3

The listener, wearing a red hooded sweatshirt and silver hoop earrings, maintains a generally still posture throughout the clip. Initially, her gaze is directed slightly off-camera to her right, and she blinks naturally. Around the 1.3-second mark, she performs a single, subtle head nod forward and back. Her facial expression shifts subtly during the clip, with her lips parting slightly at times and her brow furrowing briefly around the two-second mark before relaxing again. She continues to look mostly forward for the remainder of the clip, blinking occasionally. Towards the end, she produces a soft, low-pitched vocalization that sounds like a breathy "yeah."

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- blinking · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'naturally'}
- lip parting · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'at times'}
- brow furrowing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- blinking · visual · positive · baseline · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002', 's005']}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- lips parting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow furrowing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002', 's005']}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- lip parting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow furrowing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip parting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- brow furrowing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3631_I00000411_local_bb07a099ca84

The listener maintains a steady gaze forward throughout the clip. Initially, her eyes look slightly upward and to her right before shifting to center. Her head remains mostly still, with a slight tilt back at first, then returning to a neutral position. Around the 1.9-second mark, she performs a single, distinct head nod. She continues to hold a consistent facial expression, characterized by a subtle smile and raised cheeks, until the end of the clip. The audio contains no discernible speech or vocalizations.

**v1：Explicit count lacks literal numerical evidence**

- gaze forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes looking upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes looking to the right · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilt back · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- cheeks raised · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- gaze · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilt · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S3648_I00000032_local_fb8abbfe8ad2

The listener maintains a steady forward-facing posture throughout the clip, wearing a light blue baseball cap backward and a light-colored t-shirt against a plain gray background. Initially, their eyes are open and fixed on the camera. Around three seconds in, they briefly close their eyes before reopening them. At approximately four seconds, the listener performs a single, distinct head nod. Their eyes remain open for the rest of the clip, occasionally blinking. Towards the end, at about six seconds, they produce a short, audible vocalization that sounds like "Yeah."

**v1：Explicit count lacks literal numerical evidence**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'a single'}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}
- audible vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003', 's005']}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s006']}
- eyes open · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S5043_I00000069_local_26d534af4d65

The listener maintains a steady forward-facing posture throughout the clip. Initially, her head is upright and she looks directly ahead. Then, around the 1.5-second mark, she performs a single, distinct head nod, tilting her head downward and slightly forward before returning it to its original position. Following this, she briefly closes her eyes and tilts her head back slightly, holding that posture for a moment. She then opens her eyes and begins to speak, producing a short, low-pitched vocalization that sounds like "Mmm." After speaking, she returns to a neutral expression, keeping her head mostly still until the end of the clip.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilting back · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- baseline · visual · positive · baseline · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s001', 's002', 's005']}

**v3：结构通过**

- posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting back · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V02_S5232_I00000027_local_c9767b16b310

The listener maintains a forward-facing posture throughout the clip. Initially, her head is tilted slightly back and to her right, with her eyes looking upward and to her left. She then performs two distinct head nods: first, a brief downward tilt around the 1.1-second mark, followed by another similar motion near the 2.5-second mark. After the second nod, she turns her head forward and breaks into a smile that persists until the end of the clip. Her mouth opens during the smile, but no speech or other vocalizations are audible.

**v1：结构通过**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes looking upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes looking to the left · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'quote': 'two distinct head nods'}
- head turning forward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s003']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s003']}
- head turning · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- head turned forward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0268_I00000486_local_85eedccbc632

The listener remains in a consistent seated position throughout the clip, facing forward with their head slightly tilted back against a blue quilted background. Initially, they maintain a neutral expression with their eyes open and fixed forward. Around the 0.56-second mark, they perform a single, brief head nod, lowering their chin slightly before returning to the original posture. For the remainder of the clip, their head stays relatively still, though subtle shifts in gaze occur, including a moment where their eyes briefly close and then reopen. Their mouth remains mostly closed, with only minor adjustments visible around the lips. No audible vocalizations or speech are detected at any point during the recording.

**v1：Explicit count lacks literal numerical evidence**

- seated position · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilted back · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes fixed forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- head still · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze shift · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'subtle shifts in gaze occur'}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lip adjustment · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'minor adjustments'}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- neutral expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head tilted · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0360_I00000384_local_7223dfefce8f

The listener maintains a consistent forward-facing posture throughout the clip. Initially, they exhibit a series of rapid head nods. Following this, their head remains relatively still for the remainder of the duration. Their facial expression shifts from a broad smile to a more neutral look, with their mouth opening and closing intermittently, suggesting speech. The listener does not produce any audible vocalizations during the clip.

**v1：结构通过**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'a series of'}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'intermittently'}
- mouth closing · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'intermittently'}
- no audible vocalizations · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening and closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- forward-facing posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0466_I00000042_local_846c91e5c7dc

The listener maintains a mostly still posture throughout the clip, with her hair pulled back into a bun and her gaze directed forward. Initially, she blinks slowly and keeps her head steady. Then, at approximately one second, she produces a soft, low-pitched hum, followed by a brief pause before emitting another similar sound around seven seconds. During these vocalizations, her lips move slightly, but there is no significant change in facial expression or head position. She continues to look forward for the remainder of the clip, blinking occasionally and making very subtle shifts in her head angle, though no distinct nodding motions are observed beyond the provided supplementary data.

**v1：Explicit count lacks literal numerical evidence**

- blinking · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'blinks slowly'}
- head steady · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- humming · audio · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'quote': 'another similar sound'}
- lip movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head angle shift · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'shifts'}

**v2：Incomplete model generation**

输出截断，原始回复见对应attempts文件。
**v3：Explicit count lacks literal numerical evidence**

- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- humming · audio · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s003']}
- lip movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s005']}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- humming · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lip movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0476_I00000315_local_15fa8c6b5c9a

The listener maintains a consistent forward-facing posture throughout the clip. Initially, their right hand moves up toward their face, making contact near the nose before lowering again. The person's mouth remains open for the duration of the clip, showing continuous visible movement consistent with speaking. Their eyes stay directed forward, and no distinct head or body movements occur beyond the initial hand motion. There is no audible laughter or other nonverbal vocalization detected during the clip.

**v1：Evidence must be an exact source substring**

- forward-facing posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand moving up · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand making contact · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- mouth open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- visible mouth movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes directed forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no head movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible laughter · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no nonverbal vocalization · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- baseline · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth open · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting glasses · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth opening · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nonverbal vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- hand moving up · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0782_I00000059_local_7f5d8964d753

The listener remains seated and facing forward throughout the clip. Initially, his head is still and he looks directly at the camera. Around three seconds in, he performs a single, brief nod by lowering his head slightly and then raising it back to its original position. After the nod, he continues to look forward, blinking occasionally and making small, subtle shifts in his gaze. His facial expression stays relatively neutral for the duration of the clip. There are no audible vocalizations or speech from the listener during the entire segment.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}
- gaze shifting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'small, subtle'}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- seated · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facing forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- facial expression neutral · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression neutral · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0877_I00000411_local_4d2ff70606d7

The listener, wearing a red hooded sweatshirt, black baseball cap, and glasses, maintains a steady forward-facing posture throughout the clip. Initially, his head is still, and he looks slightly upward before beginning to speak. Around two seconds in, he performs a single, distinct head nod, tilting his head down and then back up. He continues speaking for the remainder of the clip, his mouth moving visibly as he says, "Exactly." His gaze remains fixed forward during the utterance, and no other significant facial or bodily movements are observed.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- mouth moving · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head still · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze fixation · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- baseline · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth moving · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze fixed forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0877_I00000411_local_7e467a7be48c

The listener maintains a steady forward gaze throughout the clip, her head remaining mostly still with only slight shifts in position. Her facial expression is consistent, showing a neutral look with slightly furrowed brows and a closed mouth. At approximately two and a half seconds into the clip, she performs a single, brief head nod. She then emits a short, low-pitched vocalization, "Mhm," before returning to her previous posture. The rest of the clip consists of her holding this same static pose and expression until the end.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slight head shifting · visual · uncertain · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- neutral facial expression · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- slightly furrowed brows · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- closed mouth · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gazing · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- facial expression · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- frowning · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- slight head shift · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- neutral facial expression · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- slightly furrowed brows · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- closed mouth · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nod · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S0877_I00000411_local_ed7cb61d3086

Assistant: The listener maintains a steady forward gaze throughout the clip, her head held upright and still. She wears large silver hoop earrings and a gray high-necked sweater, with a small black microphone clipped near her collar. Her dark hair frames her face as she blinks occasionally and shifts her eyes slightly upward and to the side. At approximately five seconds, she performs a single, distinct head nod, tilting her head down and up quickly before returning to her initial position. This nod repeats around seven seconds, again showing a brief downward tilt followed by a return to the original posture. No other significant movements of the head, hands, or body are observed. The listener does not speak, laugh, or make any other audible vocalizations during the clip.

**v1：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'repeats'}
- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}
- eye movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s004', 's005']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- eye movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：Explicit count lacks literal numerical evidence**

- gaze holding · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s004', 's005']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughing · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004', 's005']}
- eye shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head still · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1003_I00000059_local_f2ca565ac69c

The listener maintains a consistent seated posture throughout the clip, facing forward with long dark hair framing her face and wearing glasses with thin frames. Initially, she smiles broadly, showing her teeth, and her head moves slightly upward and to the right before returning to center. She then nods once between approximately 0.2 and 0.96 seconds, after which her expression settles into a gentle smile that persists for the remainder of the clip. Her eyes remain open and directed forward, blinking occasionally, while her mouth stays closed except for the initial smile. There are no audible vocalizations, and the background remains static behind her.

**v1：Evidence must be an exact source substring**

- seated posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- facing forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'once'}
- gentle smile · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}
- mouth closed · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- static background · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- smiling · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- head movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002', 's003']}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting glasses · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- smiling · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head moving · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes open · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- mouth closed · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1211_I00000158_local_907fd33efcbf

The listener maintains a generally still posture throughout the clip, wearing a blue hooded sweatshirt and facing forward. Initially, he looks slightly upward and to his right with a neutral expression. He then briefly closes his eyes before reopening them and shifting his gaze toward the camera. His head tilts slightly to his left, and he holds this position for several seconds. Toward the end of the clip, he turns his head further to his left, looking off-camera in that direction. There are no audible vocalizations during the entire duration.

**v1：结构通过**

- posture stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head turning · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no audible vocalizations · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Invalid source sentence IDs**

- head tilting · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eye closing · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- maintaining still posture · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking upward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking to the right · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- closing eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- opening eyes · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- shifting gaze toward camera · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting to the left · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- turning head to the left · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking off-camera · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1348_I00000393_local_52f27cf92f82

The listener begins by looking down and smiling, then lifts their head to look forward with a broad smile that persists for several seconds. While maintaining this expression, they laugh audibly, producing a series of short, breathy chuckles. Toward the end of the clip, they briefly raise both hands in front of their chest, palms facing each other, before lowering them again. A single, distinct head nod occurs during this final moment.

**v1：Evidence must be an exact source substring**

- looking down · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lifting head · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- broad smile · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- audible laughter · audio · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'a series of short, breathy chuckles'}
- raising hands · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- palms facing each other · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- lowering hands · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：结构通过**

- looking down · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughing · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raising hands · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- audible laughter · audio · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}

**v4：结构通过**

- looking down · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lifting head · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- laughing · audio · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- raising hands · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lowering hands · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1393_I00000111_local_76b117b06695

The listener maintains a consistent posture throughout the clip, facing forward with their head held steady. Their eyes remain open and directed slightly upward and to the left for the entire duration. A subtle smile is present on their face from the beginning, persisting without noticeable change. Around the three-second mark, the listener performs a single, brief head nod, moving their head downward and then returning to its original position. No other distinct facial movements, hand gestures, or body shifts occur during the clip. The listener remains silent for the full duration, with no audible speech or vocalizations detected.

**v1：Explicit count lacks literal numerical evidence**

- head holding steady · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eyes directed upward and left · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- subtle smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}

**v2：Invalid relation**

- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes open · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s004']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- head held steady · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes directed upward and to the left · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- subtle smile · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1421_I00000117_local_674f523d111d

The listener maintains a consistent smile throughout the clip, her gaze directed downward for the first two seconds before lifting to look forward. Her head remains mostly still, with only slight shifts in position, and she blinks naturally at intervals. There is no discernible nodding or hand movement, and her posture appears relaxed. The audio contains no speech or vocalizations, only silence.

**v1：Evidence must be an exact source substring**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze downward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- gaze forward · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head shift · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'slight shifts in position'}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'at intervals'}
- no nodding · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no hand movement · visual · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- relaxed posture · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- no speech · audio · negative · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- smiling · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s001']}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand raising · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head movement · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1485_I00000146_local_0f7d71819484

The listener maintains a consistent smile throughout the clip, her head tilted slightly back at first before settling into a more neutral position. She nods twice, the first time between approximately one and two seconds, and again between three and four seconds. Her mouth opens several times during the clip, producing audible laughter and a single spoken word, "Yeah," which she says clearly around the two-second mark. The rest of the time, she remains visually engaged with a smiling expression, occasionally closing her eyes briefly while laughing.

**v1：Evidence must be an exact source substring**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head tilting · visual · uncertain · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'quote': 'twice'}
- mouth opening · visual · uncertain · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'several times'}
- audible laughter · audio · positive · state · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'several times'}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- eye closing · visual · uncertain · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'occasionally'}

**v2：Invalid source sentence IDs**

- smiling · visual · positive · baseline · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s002']}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}
- speech · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': []}

**v3：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 2, 'source_ids': ['s002']}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- audible laughter · audio · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- speech · audio · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- eye closing · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}

**v4：结构通过**

- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head tilted · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s002']}
- mouth opening · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003']}
- laughing · audio · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s003', 's004']}
- speaking · audio · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- eyes closed · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}

## naturalistic_V03_S1532_I00000036_local_0a47cb3ab2a1

The listener begins by looking slightly upward and to their right, with a faint smile on their face. They then perform a single, brief nod of the head around the first second of the clip. After the nod, they maintain a steady gaze directed toward their right, holding a consistent facial expression that includes a slight smile and relaxed brow for the remainder of the clip. There are no other discernible head movements, hand gestures, or body shifts throughout the duration. The person remains visually still and silent, with no audible vocalizations or speech detected.

**v1：Explicit count lacks literal numerical evidence**

- looking upward · visual · uncertain · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'single'}
- gazing · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- relaxed eyebrows · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}

**v2：Explicit count lacks literal numerical evidence**

- looking upward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking to the right · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s001']}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- gaze steady · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- facial expression consistent · visual · positive · state · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- no head movement · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no hand gesture · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no body shift · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- visual stillness · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- no speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- gaze shifting · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s002']}
- gaze holding · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- looking upward · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- looking to the right · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- smiling · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand gesturing · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- body shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

## naturalistic_V03_S1788_I00000293_local_c5a3bf7e6ab6

The listener, a person with long blonde hair and glasses wearing a blue hooded sweatshirt, is seated against a blue quilted background. Initially, they raise their right hand to adjust their glasses before lowering it and maintaining a steady forward gaze for the remainder of the clip. Their head nods downward once between approximately 0.5 and 1.3 seconds. The listener blinks several times throughout the clip but otherwise remains still, with no discernible changes in facial expression. There are no audible vocalizations, speech, or other sounds produced by the listener during the entire duration.

**v1：结构通过**

- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- hand lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- steady forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'quote': ''}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'quote': 'once'}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'quote': 'several times'}

**v2：结构通过**

- hand movement · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- speech · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v3：结构通过**

- hand raising · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- adjusting glasses · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- hand lowering · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- gaze shifting · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'explicit', 'count': 1, 'source_ids': ['s003']}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- facial expression change · visual · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

**v4：结构通过**

- raising right hand · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- lowering right hand · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- head nodding · visual · positive · change · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- blinking · visual · positive · change · 重复：{'kind': 'multiple_unspecified', 'count': None, 'source_ids': ['s004']}
- maintaining forward gaze · visual · positive · state · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}
- vocalization · audio · negative · baseline · 重复：{'kind': 'not_stated', 'count': None, 'source_ids': []}

