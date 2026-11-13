# 音视频统一回看页

入口：http://127.0.0.1:8876/p2_av_review/ （沿用8876端口转发）

旧 `/p2_train24_fourway/comparison.html` 自动跳转；旧页面保存在原目录 `comparison_before_av_review_20261027.html`。原始生成媒体保留。

## 完整范围

训练集24条、独立test24条；每套数据4组权重 × 固定/原文标注指导2条件 × seed137/251，共384个输出。页面合计768个输出，每组每个条件每个数据集都是48个。指导版384个全部生成，逐输出与固定版核验权重、输入、seed、采样配置和几何信息一致，仅改变Gemma正向文本。

[完整配对评测说明](../p2_guided_full_eval_20261027/README.md)、[流程状态](../p2_guided_full_eval_20261027/status.json)。完整指标及RGB修正版GLARE评分均已完成，页面已更新；有效支持范围逐指标注明。

## 播放与来源

- 每套24条连续下滑浏览，无“下一条”。可按seed、生成条件、样本ID筛选。
- GT有声视频复制实际目标视频流，并封装清单中的独立listener音频。训练样本使用四组原训练音轨；test明确标注未用于训练，另展示实际评分时使用的后处理音轨。
- Speaker播放器读取实际observations的音频。48份Qwen3-ASR转写由同一文件产生，并绑定音频SHA；不是人工真值，也未作为训练输入。
- 生成视频可选“仅listener”或“listener＋原speaker”。768个输出均有混音版本，从片段0秒对齐、等增益相加；必要时共同降幅防削波，不独立放大listener。原生成及评分音轨不改。
- 切换模式保持播放位置和播放状态。一次只允许一个播放器发声，GT不受生成音轨切换影响。
- 每个指导版卡片可查看实际Gemma文本；两套数据、两个seed均已补齐，不再是先前9条工程样例。

## 指标口径

统一展示FVD-32、UTMOSv2、LSE-D/C及GLARE的R-F1、R-tIoU、R-ATD、R-FID。固定/指导共8条件、两个seed按每项指标取共同有效样本集合，逐seed计算后等权平均；表格注明覆盖范围。每个条件的完整48配对记录仍保留，失败、不适用、无匹配不能填0。FVD按样本池计算，不平均逐视频FVD。

卡片展示逐输出音频电平及已有UTMOS/SyncNet分数。UTMOS/SyncNet不证明回应语义和时机正确；指导版使用了目标标注，属于额外信息条件。原固定prompt报告保留在原评测目录，新配对表会依据8条件共同支持重算固定版数值。

GLARE修正版采用RGB做人脸框检测、原BGR做情绪分类，保留严格逐帧门槛，参见[修复与全量检测结果](../p2_glare_detector_fix_20261027/README.md)。发布后页面优先读取`comparison_rgb_v2.json`，原BGR报告保留。test八条件共同有效范围从0/24提高到10/24；train为1/24。

指标表可切换“固定／指导八条件同样本配对”与“各prompt内部四组权重比较”。后者test固定/指导范围分别15/24、14/24，train分别5/24、1/24。两种prompt集合可能不同，内部比较表不能跨prompt直接作差。每组仍有48个输出并全部执行评测；共同有效数表示评分支持集合，不是生成数。

构建入口：`scripts/build_p2_av_review.py`；前端：`scripts/p2_review_assets/`。`data.json`为页面清单，混音JSON保存输入哈希、对齐和增益。

## 验证

- `browser_full_guided_validation.json`：完整指导版24条×双seed浏览、GT有声播放、切换位置、单播放器互斥、桌面/手机布局，无脚本错误。
- `browser_rgb_validation.json`：train/test × 八条件/内部四组四种表格状态逐数值核对，每表8行、每行48输出，RGB版本标识与比较范围切换通过，无脚本错误。
- `full_media_validation.json`：48份GT、48份ASR、768份混音的实际来源和等增益核验，HTTP Range 206通过。
- 3项混音/重封装回归测试通过。
- 旧`browser_validation.json`和`media_validation.json`为先前393个输出版本的历史记录，不代表当前范围。
