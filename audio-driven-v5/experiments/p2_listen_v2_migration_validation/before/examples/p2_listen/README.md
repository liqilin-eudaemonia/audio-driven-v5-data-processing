# P2 数据接口示例

这些 JSON 是格式示例，不是真实样本；占位文件与哈希不能用于训练。用户可以先提供现有数据路径/清单，后续再适配到此格式，不要求用户手工整理这些文件。

- `observations.example.json`：只含推理可用信息。路径相对于导入 JSON 所在目录解析；`grouping_ids` 同时记录双方参与者和 session，跨 split 重复会被拒绝。
- `targets.example.json`：独立目标视频/音轨。先完成质量与角色验收，再把 `admitted` 从示例中的 `false` 改为 `true`；代码不会代替人工决定数据是否合格。
- `labels.example.json`：片段内秒数、逐类已审阅 coverage/unknown、事件区间及媒体哈希。未知不是负例；dev/test 需人工核验。
- `evaluation_protocol.example.json`：固定的配对生成矩阵。只有一个样本的示例不支持科学结论。

基础素材为切好的等时长双路音轨、512×512 listener 视频和同裁框首帧。prepare 负责验证和导入，不会从长视频自动切分，也不会修改正在运行的 QC4 队列。

完整使用方式见 [实现与运行说明](../../related/P2_倾听事件监督实现与运行.md)。
