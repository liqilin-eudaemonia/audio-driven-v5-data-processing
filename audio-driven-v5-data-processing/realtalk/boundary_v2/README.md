# RealTalk 切分边界 v2：已确认的小批规则与备份

版本 `realtalk-boundary-v2-pilot`。**2026-10-25 用户确认“现在的切分是对的”，并要求整理备份。** 确认范围是完整原视频 `-1wCGAxqT_4` 的本次切分和展示规则；未授权将本次结果视为全量角色真值、干净listener音轨或训练准入。

## 结果与规则入口

- [处理规则、参数和实现说明](snapshot/README.md)、[设计与局限](snapshot/DESIGN.md)、[用户确认原文](snapshot/user_acceptance.json)。
- [切分核心](snapshot/code/boundary_core.py)、[完整处理入口](snapshot/code/run_pilot.py)、[实际参数](snapshot/policy.json)。
- [完整切分清单](snapshot/manifest.json)、[统计](snapshot/summary.json)、[覆盖核验](snapshot/validation.json)、[浏览器验证](snapshot/browser_validation.json)。
- [新词级转写](snapshot/words.json)、[22个原始模型响应](snapshot/asr/)、[边界测试](snapshot/code/test_boundaries.py)。
- [旧版问题审计](audit_v1/findings.json)、[备份逐文件哈希](BACKUP_MANIFEST.json)、[备份检查结果](BACKUP_VALIDATION.json)。
- 原机器审核页：http://127.0.0.1:8798/boundary_v2_review/ 。localhost链接不是GitHub托管服务。

原片611.64秒、15,291帧；137个候选pair、401.56秒，最长10.44秒。旧版152条、400.84秒候选全部保留，新增0.72秒；单帧候选由4条降至1条。206.72秒角色未知、3.36秒黑屏完整单列，未当作废片隐去。137条原生裁剪＋原混音导出通过帧数/音频时长检查。

处理规则：正式接话前截断，前后继续提取；短回应候选保留在倾听片段中。缺失分数不当静音；持续几何变化须有图像支持，近邻几何切点与视觉切点对齐；逐帧黑屏包含EOF；切分不超出真实音频支持范围；词级句末/词间停顿优先，最长11秒、短尾段保留；代码/规则/输入变化使缓存失效。接话持续性采用0.8秒等明确试验参数，不能单靠时长证明语义，所有事件和未确定角色保留审核依据。

本批审核音频是**原混音**，方便判断说话人；不冒充已分离的listener音轨。音轨分离和SyncNet后续应在正确角色/边界上执行。

## 备份范围

`snapshot/`逐字节保存当前代码与规则、候选清单、ASR和验证元数据；`dependencies/`保存这条源视频的原清单、逐帧人物标注、页面字体及许可。`audit_v1/`保存问题定位证据。没有上传视频、WAV、模型权重、SSH配置或凭据。

源媒体、原生裁剪视频及权重位置见[本地资源清单](LOCAL_OUTPUTS.json)和[环境信息](environment.json)。原视频仍从TAR中的明确offset/size只读访问。ASR模型为本地Whisper large-v3；准确模型路径记录在环境信息与转写证据中。

旧版全量备份继续位于 `realtalk/snapshot/`，新版本没有覆盖它；原生产目录仍为 `/private/liqi_workspace/realtalk_processing/full_20261023`。本批工作目录为 `/private/liqi_workspace/realtalk_processing/boundary_v2_pilot_20261025`。

## 校验备份与恢复

仅运行边界回归，无需媒体或GPU：

```bash
python realtalk/boundary_v2/snapshot/code/test_boundaries.py
```

恢复脚本要求**不存在的新工作目录**，先核对备份哈希，再复制并记录路径替换；不覆盖旧运行目录或默默复用其缓存：

```bash
python realtalk/boundary_v2/restore.py \
  --workdir /path/to/realtalk_boundary_v2 \
  --data-root /public/0_DATA/2_Avatar/realtalk
```

同一数据集、相同源TAR的恢复复跑：

```bash
python /path/to/realtalk_boundary_v2/code/prepare_source.py
python /path/to/realtalk_boundary_v2/code/verify_audio.py
python /path/to/realtalk_boundary_v2/code/test_boundaries.py
python /path/to/realtalk_boundary_v2/code/run_pilot.py
python /path/to/realtalk_boundary_v2/code/export_review.py
python /path/to/realtalk_boundary_v2/code/publish_review.py
```

上述使用已归档的词时间，不需要重新跑GPU ASR；会校验原混音的采样率、样本数及float32波形哈希，不把另一条音频套上旧词时间。需要NumPy、SoundFile、FFmpeg/ffprobe，以及原始数据TAR。恢复后的输出位于新目录的`review/`；当前机器复用8798服务，不自动创建/占用新端口。

若明确需要重新转写，准备音频后，在含Torch/Transformers/SoundFile的GPU环境运行`code/transcribe_words.py`；模型迁移时可在restore命令指定`--model-path`。新的ASR输出不保证与归档逐字相同。恢复目录的路径改变也会改变运行指纹，这是正常行为。

本次归档执行了文件哈希、Python语法、12项边界回归、恢复目录与拒绝覆盖检查；没有因归档再跑一遍完整ASR或视频导出。历史137条媒体与浏览器验证原样保留，可追溯其原始范围。
