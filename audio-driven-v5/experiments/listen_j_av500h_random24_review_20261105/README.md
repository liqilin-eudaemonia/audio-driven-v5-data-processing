**版本说明：此页24条来自旧train-only版。** 当前选集已按用户确认改为train＋dev，见[新版入口](../listen_j_upright_av500h_train_dev_v3_20261105/README.md)。原抽样和意见保留；每条在新版是否仍入选见[成员对应](MEMBERSHIP_IN_TRAIN_DEV_REVISION.json)。

# 最终500h训练集：随机24条实际素材复核

[打开可播放检查页（沿用8876端口）](http://localhost:8876/listen_benchmark_review/j_av500h_random24_20261105/)。

来自最终284,264条／500.00069849h训练清单，全部train。M01–M16为主集，V01–V08为有声补充。使用固定种子`j-av500h-selected-random24-20261105-v1`和group＋sample_id的SHA256优先级，在组内无放回随机抽样，不按caption内容或实际观看效果挑换。补充组超额抽取用于复核，比例不代表全量构成。见[sampling.json](sampling.json)、[样本清单](sample_list.csv)。

复用此前最终裁框复核HTML及float32双音轨播放器。默认播放listener音轨，可切换speaker或双方同时听，支持0.5倍速、拖动和参考首帧；右侧显示最新原caption和冻结v3行为／次数统计，未知次数仍为未知。可在页面记录并导出复核意见，也可直接在对话中按M／V编号反馈。

预览从选中行的真实最终来源解码，512×512、25fps；caption、原统计行、选集行及媒体来源均绑定SHA。listener按原官方词／句／VAD配方重放，float32波形哈希与原打标输入一致；speaker为配对参与者同时间区间音轨。下载MP4／M4A是AAC预览副本，网页播放器使用原float32 PCM。转正来源使用其已验收来源链；官方转写从原listener档案读取。

24条视频加载、48路PCM浏览器SHA、两组筛选、切换音轨／播放／拖动、意见保存导出和移动端布局验收通过。[构建验收](build_audit.json) · [实际浏览器验收](review_audit.json)。原标签、统计、500h清单和训练缓存均未改写，没有重新生成caption。

构建脚本：`scripts/build_j_av500h_random_review.py`；验收脚本：`scripts/check_j_av500h_random_review.py`。

实际抽样：非零listener音轨11条、全零13条；带普通caption复核标记3条；已转正0条。

## 助手语义抽查（24条）

[逐条判断和原响应／最终统计对照](quality_assessment/assessment.json)。本次对照24条caption、8fps抽帧图和记录的发声时间／波形支持，声音词句未作独立人工听辨；不外推全量准确率。发现原caption的局部表情遗漏、持续范围过度概括和约1秒发声时间偏差；结构化统计亦存在明确漏类和CPU计数汇总丢失，尤其M03两段点头变1、M04原模型2变null、V03原模型3变null。来源／索引／格式验收通过不等于语义准确率通过。原标注、统计和500h成员未改写，问题待后续定向修复。

## 用户确认本轮验收尺度

用户明确：“很细节的就算了，大致满足就可以了。我没时间继续打标了。”据此，本轮保留已选500h，以主要动作和段落大意符合为工作标准；不追加细节级打标，精确时间、微小表情和次数问题不作为训练准备阻塞项。助手发现的细节保留追溯，不等于全量逐条人工核验。见[确认记录](quality_assessment/user_acceptance.json)。
