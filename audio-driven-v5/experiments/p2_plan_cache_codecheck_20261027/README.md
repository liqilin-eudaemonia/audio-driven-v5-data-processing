# 显式规划条件缓存代码验证

- [实现与运行](../../related/P2_显式规划条件缓存实现与运行.md)
- [最终验证摘要和源码哈希](validation_summary.json)：56项CPU测试通过；[JUnit](pytest.xml)。
- [真实数据最终检查](real_final_validation.json)：7条英文计划准备通过；1条含单字中文的记录被明确拒绝；1条后处理AV缓存实际CPU读取通过。
- [真实tokenizer检查](real_tokenizers.json)：7条文本覆盖1024/2048/4096档，Gemma原始token数955–2403；文本与最终准备版本已逐条核对。
- `smoke7_final/`是7条工程样本的准备协议，没有新GPUembedding/hidden；不能当作全量训练缓存。
- `smoke8_final/preparation_errors.json`保存单字中文冲突。早期smoke目录保留为迭代记录，最终结果以`real_final_validation.json`为准。

未启动全量缓存、预训练模型GPUforward或新训练；没有改动已验收的数据与旧缓存。
