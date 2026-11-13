# 自然语言丰富度工具代码检查

2026-10-31，代码准备与小量只读验证；未调用真实30B模型、启动GPU任务、进行全量扫描或重写标签／缓存。

- 29项CPU回归：`python -m pytest -q tests/test_listen_richness.py`。
- 固定12条真实J结果：从单个已知caption分桶限定读取12个文件，**不是全量随机样本或语义评测**。包括旧框／修正框、静音／非静音输入。
- 首轮仅读caption，3条旧结果未内嵌画质验收，被正确拒绝默认放行；`stats.sqlite`保留该诊断，不是生产任务报错。
- 按同pair补读既有画质验收后，最终版本使用 `inputs_with_quality.jsonl → final_verified.sqlite`：12/12通过来源、配方、时间和原图人脸统计检查，0错误。`--work`会自动做这一回查。早期 `verified.sqlite` 仅留诊断，代码合同已过期，后续用final版本。
- 12条状态仍为pending，等待真实文本模型提取；没有伪造成功结果。
- 分层样本列表 `pilot_ids.json` 和12条模型请求 `requests.jsonl` 已导出，尚未提交模型。
- HTTP坏回复重试使用本地模拟服务；导入、断点、导出与筛选用构造数据验证，不能当作真实语义效果。

`verification.json`记录测试与输入哈希核验。`code_snapshot/MANIFEST.json`是新增代码、配置、测试和交接文档的本地冻结副本，不代表GitHub备份。

使用方式见[交接文档](../../related/P2_自然语言打标丰富度统计与子集筛选交接_20261031.md)。
