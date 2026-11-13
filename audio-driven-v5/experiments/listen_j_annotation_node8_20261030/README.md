# node8：J数据视觉打标＋点头补充

本节点承担502,140段、26,538条来源。配置、分片、启动记录和检查见[两机统一交接](../listen_j_annotation_two_nodes_20261030/README.md)。

视觉结果在 `vlm/results/`，点头结果在 `nod/results/`。来源规格 `jobs/` 为指向node17已提交规格的只读复用链接，队列、状态、完成凭证和结果独立保存。不得再次运行全量ingest覆盖分片。
