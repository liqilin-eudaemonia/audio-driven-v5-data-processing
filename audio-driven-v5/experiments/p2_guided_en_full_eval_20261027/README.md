# 全英文标注指导重新推理与两套GLARE口径

按用户2026-10-27更正：Gemma实际指导文本必须全部英文；原中文发声词译成英文，同时保留原文和对应记录。

## 不变项与唯一推理干预

train24、test24，各4组权重×2seed，每组48个输出，共384个新指导输出。固定prompt的384个输出继续作为同输入、权重、seed、采样参数对照。只更换正向Gemma文本并重新编码；负向embedding逐张量校验不变。权重、LoRA、speaker音频、listener首帧、时长、采样步数与CFG不变。

新代码为`src/p2_guided_prompt_en/`及`scripts/p2_guided_prompt_en.py`；旧中英混杂版本的源码、缓存、输出和报告全部保留。原英文描述、发声文本和时间区间逐字保留；中文模板改为英文。

9条样本的10个音频事件含中文原始发声词；因description和text重复记录，共20个字段转换。明确映射：是→Yes，对→Right，哥→Brother，哼→Hmm，哎→Ah，对了→Right，嗯→Mm-hmm。它是用户授权的文字翻译，不是重新识别音频或修正ASR；无法据此保证原ASR本身正确。

- `english_prompt_audit.json`：48份文本审计、原记录不变、逐字段翻译。
- `../p2_guided_prompt_en_train24_20261027/prompts/`、`../p2_guided_prompt_en_test24_20261027/prompts/`：实际英文prompt与包含完整原文的JSON。
- `jobs.json`：384条任务、实际输入/权重/指导身份。
- `status.json`、`workers/`、`logs/`：生成及评测状态；`finish.py`负责缓存后自动接续生成、评测和页面发布。
- `generation_pairing_verified.json`：完成后核验全部384输出与原固定版输入、权重、seed、几何及配置一致。

使用已核实空闲的两机16卡。第三台原地址172.29.225.51关闭SSH握手，不更改其连接配置或占用未知任务。

## GLARE两版

两版共享同一套RGB修复后的原始检测，沿用原反应参数和指标公式。

1. 严格版：GT/生成每帧MediaPipe、唯一FER人脸、框IoU≥0.1全部通过，再取八条件及两个seed共同集合。
2. 去额外门槛版：取消上述整段准入筛选，全部24条/seed进入评分。多脸采用FER默认first_face_only语义；无FER人脸的情绪空值由原规则按0读取；MediaPipe漏检沿用detected mask及原姿态插值、平滑。漏检诊断保留，不宣称这些帧是可靠负例。

参考标注工具`generate_usable.py`本身也筛除MediaPipe detection_rate!=1和FER不完整片段；GLARE指标公式只接收反应CSV。因此去门槛版是独立诊断口径，不能称为官方标准全量评测。

反应规则必须使用与原检测缓存相同的Python/NumPy/SciPy/OpenCV环境。先前在不同NumPy环境重算时出现约1e-6末位差异，已在验收前拒绝并归档；当前锁定原检测环境并要求原严格有效视频的新旧反应CSV哈希完全一致。8个真实视频（4严格有效、4原无效）已验证通过。

`stream_detect.py`在生成期间并行建立检测与反应CSV；`reaction_eval.py`核验全部816个独立视频和768配对，生成严格/去门槛的评分任务。R-FID使用GPU提取特征，逐seed计算后平均；设备记录在评分结果中。没有匹配事件的R-ATD等仍为N/A。

FVD-32、UTMOSv2、SyncNet按各自有效支持集合统计，切换GLARE口径不改变其分数。

## 发布

生成与全部评测完成才写`publication_ready.json`，随后HTML整体切换到新英文输出，避免混用旧视频或旧指导分数。新评测目录：

- `../p2_guided_en_train24_fourway_eval_20261027/`
- `../p2_guided_en_test24_fourway_eval_20261027/`

每个目录的`comparison_rgb_v2.json`、`report_rgb_v2.md`是严格版；`comparison_unfiltered_v1.json`、`report_unfiltered_v1.md`是去门槛版。页面默认展示去门槛版，保留严格版与比较范围切换、检测漏检率、全英文实际prompt及原文翻译记录。

入口：http://127.0.0.1:8876/p2_av_review/ 。是否已发布以`status.json`和浏览器验收记录为准。

## 已完成并验收

384个全英文指导输出全部重新生成并完成评测；逐输出与固定版配对身份核验384/384通过。HTML已整体切换为384个新英文指导输出＋384个固定输出，共768个。生成、两套GLARE评分、发布和浏览器验证状态均为complete。

八条件、双seed的GLARE共同范围：严格版train为2/24、test为7/24；去额外门槛版两套数据均为24/24。FVD两套均为24/24；UTMOS/SyncNet为train24/24、test13/24，去掉GLARE门槛不改变音频指标适用范围。

`../p2_av_review_20261027/browser_english_validation.json`验证384个实际prompt全英文、两种GLARE口径与两种比较范围的全部表格数值及手机布局。`browser_full_guided_validation.json`验证有声GT、模式切换、互斥播放和连续浏览；`full_media_validation.json`验证768份混音来源、增益及HTTP Range。全部通过。
