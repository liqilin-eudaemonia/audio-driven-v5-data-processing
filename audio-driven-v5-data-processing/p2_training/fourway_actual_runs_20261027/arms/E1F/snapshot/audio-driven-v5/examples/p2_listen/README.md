# P2 数据接口示例

当前训练协议为 `p2-listen-event-v2-visual6-audio1`，固定顺序：head_nod、head_shake、smile、laugh、frown、surprise、listener_vocalization。新配置为 `configs/p2_listen_event_v2.json`；旧二类标签示例保存在 `legacy_v1/`，新导入器拒绝直接使用。

这些 JSON 是格式示例，不是真实样本；占位文件与哈希不能用于训练。用户可以先提供现有数据路径/清单，后续再适配到此格式，不要求用户手工整理这些文件。

- `observations.example.json`：只含推理可用信息。路径相对于导入 JSON 所在目录解析；`grouping_ids` 同时记录双方参与者和 session，跨 split 重复会被拒绝。
- `targets.example.json`：独立目标视频/音轨。先完成质量与角色验收，再把 `admitted` 从示例中的 `false` 改为 `true`；代码不会代替人工决定数据是否合格。
- `labels.example.json`：片段内秒数、逐类已审阅 coverage/unknown、事件区间及媒体哈希。未知不是负例；dev/test 需人工核验。
- `decisions.example.json`：已有候选的独立接受/排除记录模板，默认未核验，不能直接导出为已准入标签。完整填写规则见[七类迁移实现记录](../../related/P2_七类事件监督迁移实现与验证.md)。
- `evaluation_protocol.example.json`：固定的配对生成矩阵。只有一个样本的示例不支持科学结论。

基础素材为切好的等时长双路音轨、512×512 listener 视频和同裁框首帧。prepare 负责验证和导入，不会从长视频自动切分，也不会修改正在运行的 QC4 队列。

完整使用方式见 [实现与运行说明](../../related/P2_倾听事件监督实现与运行.md)。

`labels.example.json` 的 coverage 全空且 `training_admitted=false`，用于展示未知状态。不能仅修改布尔值：每类已知覆盖需要 `coverage_sources`，准入需要 `admission_record.reference`；事件需要唯一 `event_id`、原文 `description`、来源 `source`、`verification` 和 `boundary_status`。弱标签政策与 mapping 版本必须和实验配置一致。

优先使用 `p2_listen.py export-labels` 从当前原文候选和独立 decisions 导出；不传 decisions 时仅输出全 unknown 的未准入草稿。它不会重跑模型、合并回应或改写候选内容。`import-labels` 只接收已经准入的七类 canonical 标签，并核对真实目标媒体哈希及 dev/test 人工核验状态。
