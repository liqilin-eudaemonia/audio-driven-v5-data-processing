# Avatar Forcing 是否公开 ViCo Table4 测评代码

2026-11-10重新核查的结论：**未找到完整公开的ViCo Table4评分程序。** 不能将本地重建的入口称为AF官方评分脚本。

官方仓库：<https://github.com/TaekyungKi/AvatarForcing>。

| 内容 | 公开情况 |
|---|---|
| 模型结构、预训练权重与下载脚本 | 已公开 |
| 单次推理、配置、参考图／speaker视频预处理 | 已公开，README明确定位为minimal PyTorch inference pipeline |
| 论文FD／rPCC／SID／Var定义、SPECTRE及K15／9说明 | 见论文附录D.5 |
| Table4实际ID／帧清单、整套特征提取和评分入口 | 未找到 |
| Table4的K-means中心／拟合语料、跨片段统计汇总配置 | 未找到 |
| Table4与公开checkpoint、采样种子／次数绑定的运行回执 | 未找到 |

检查范围是官方main和page两分支文件树、全部15个issue及已有评论、releases和tags。main提交为`507e3046ad539c938e0412ac7a74cf6e3ba2a61c`，page为`f9e9dcc658793f6dc317e4fc29590f389d9aac41`；releases和tags均为空。page含项目展示和人工评估图片，不是量化评分实现。[原始API响应](public_code_audit/inventory.json)保留查询地址和SHA，旁边保存各分支文件树及issue评论。

可直接核对的官方说明：[README推理范围](https://github.com/TaekyungKi/AvatarForcing#inference)、[inference.py](https://github.com/TaekyungKi/AvatarForcing/blob/main/inference.py)、[preprocess_user_video.py](https://github.com/TaekyungKi/AvatarForcing/blob/main/preprocess_user_video.py)、[附录D.5](https://arxiv.org/html/2601.00664v2#A4.SS5)。

目前可继续用公开推理代码和原始数据验证生成端，再依据论文和引用链重建评分。对未披露设置仍需保留推断标签；仅有官方权重并不证明Table4测评已经完全对齐。
