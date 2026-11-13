# J 切分规则：新增随机五条验证与固定版本

本轮应用用户此前认可的同一份 J 代码，没有边跑边改切分规则。抽样先于结果：seed=20261031，对原始 catalog 打乱，排除此前 7 条；只要求已有原全量结果与音频缓存可比较，不按时长、题材或改善效果挑选。遇到 1 条缺少所需旧结果而跳过，过程见 selection.json。五条均为 naturalistic，是此次随机结果。

结论：固定当前切分规则为 `listen-pair-segmentation-j-v1`。H→J 初始候选均为 150 段，140 段（93.33%）说话人与起止时间完全相同；10 处变化均为句末保护，起点不变、收回下一句残尾。本次未触发短前缀软停顿调整，没有新切进对齐词内部。

旧全量已经保留的可靠对齐词，在 J 完整初始候选中，前四条没有减少；340 秒来源只有孤立残尾 I / I / He 不再纳入。该覆盖检查不等同于这些内容都已通过音质。

**最终通过不是基本不变：旧全量 131→J 113 段，92 段起止相同，共 36 组差异。H 自动通过为 133 段。J 有 37 个未自动通过的终端候选，其中 18 个因恢复子段不能确认完整而保留原候选及原 QC 结论。新增范围、重新分段、质量待复核与排除都分列，不把待复核当成已通过，也不把通过数量减少当成额外优化证据。**

固定的是本版切分与保守恢复行为；音质规则、2～11.95 秒、5 秒停顿阈值均未改变。保留完整候选供人工复核，不自动解除音质阻断。没有启动全量、替换生产结果或训练数据。

## 五条 HTML

| 来源 | 秒 | H→J 初始候选 | 初始不变 | 旧全量/H/J 自动通过 | 旧全量→J 不变 | 入口 |
|---|---:|---:|---:|---:|---:|---|
| 1 · naturalistic_V03_S0711_I00000041 | 276 | 29→29 | 25 | 28/28/27 | 25 | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S0711_I00000041&fixed=1&compare=random5_final1#fixedPairReview) |
| 2 · naturalistic_V03_S0257_I00000138 | 232 | 29→29 | 27 | 25/27/19 | 14 | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S0257_I00000138&fixed=1&compare=random5_final1#fixedPairReview) |
| 3 · naturalistic_V02_S3143_I00000313 | 184 | 21→21 | 21 | 20/21/20 | 17 | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3143_I00000313&fixed=1&compare=random5_final1#fixedPairReview) |
| 4 · naturalistic_V03_S1356_I00000068 | 212 | 22→22 | 21 | 16/15/7 | 4 | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1356_I00000068&fixed=1&compare=random5_initial1#fixedPairReview) |
| 5 · naturalistic_V03_S1634_I00000407 | 340 | 49→49 | 46 | 42/42/40 | 32 | [打开对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1634_I00000407&fixed=1&compare=random5_initial1#fixedPairReview) |

三列为旧全量 P／历史规则 H／固定 J，均使用实际 pair 视频及源时间。所有 H→J 初始变化、三版本最终通过差异、J 未自动通过候选均覆盖。时间表含不变片段，完整三版本明细为 [intervals.csv](intervals.csv)。视频按各列的源时间排列。

## 初始切分的全部 10 处变化

| 来源 | H 起止（秒） | J 起止（秒） | J 保留的句末 | 不再追加 |
|---|---|---|---|---|
| 1 | 74.666000–77.688000 | 74.666000–76.927000 | …what they're saying now for environmental purposes, right? | Let them |
| 1 | 107.892000–114.272500 | 107.892000–112.394000 | …And is this something that you do like every week? Is it like ongoing through fall? | Like how frequently are you |
| 1 | 153.397000–157.200000 | 153.397000–156.340000 | …I just don't want to do it anymore, right? That's the thing. | It's like I'm |
| 1 | 201.197000–210.239000 | 201.197000–208.899000 | …e, you're pulling it out in the forest. I have to bag it and then put it on the curb. | Um, I just don't |
| 2 | 16.313000–19.173000 | 16.313000–18.853000 | …younger. But then I guess I tuned into a little rap. | I |
| 2 | 127.030000–131.723000 | 127.030000–129.881000 | …They like only show like seconds, right? | And then they go and then they go to |
| 4 | 158.911000–161.372000 | 158.911000–161.092000 | …Did you ever think of going through CarMax? | I |
| 5 | 148.700500–151.601500 | 148.700500–151.394000 | …I lived on the planet. I stole it from the guy. | I |
| 5 | 156.836000–160.197000 | 156.836000–159.856000 | …don't have money. I give the money to the poor. | I |
| 5 | 252.516000–256.600000 | 252.516000–256.307000 | …pounds to the side next to it. Just white flesh. | He |

## 验证与复现

- 切分模块 SHA256：`8a7f326fafbda95382a986ce19a634270e42e915db050fd07bf89afaa6ec7f0a`，与前七条所用 J 相同。
- 原生产 38 份源码及五条旧结果指纹不变。原输入、规则、音质参数均留档。
- 5 项切分回归测试通过；全部初始变化、三版本最终差异、未自动通过候选均检查覆盖。
- 浏览器串行完成 116 次对照视频播放；时间、转写、三列标签、Range 请求、手机布局及历史两条 J 入口通过。
- H 计算复用 131 段、重算 41 段；J 复用 148 段、重算 4 段。重型任务均串行，未扫全量媒体。
- 固定版本：[manifest](../../releases/listen_pair_segmentation_j_v1_20261029/manifest.json)；源码与归档校验：`python releases/listen_pair_segmentation_j_v1_20261029/verify.py`。

已有结果可按 `prepare.py` → `run.py h` → `run.py j` → `finish.sh` → `validate.py` → `check_browser.py` 重现。浏览器依赖见已有运行环境，校验过程保存在 browser_validation.json。冻结脚本 freeze.py 只在数据与浏览器验收成功后运行。

用户已认可此前展示的 J 规则与案例；这五条的判断基于真实切分、对齐词、转写与可播放文件验证，尚未把用户未提交的试听判断记录为人工通过。
