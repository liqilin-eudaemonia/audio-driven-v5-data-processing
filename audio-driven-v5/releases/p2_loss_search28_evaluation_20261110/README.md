# 搜索专用统一生成测评接入

本目录独立于训练源码指纹，正在实现；当前不是已完成GPU验收或已上线的自动选参程序。

- `freeze_validation.py`核对最终500h源数据库，检查统一200与训练的sample／pair／component重叠，冻结分层64调参子集。另选最多128条不同component的最终保留候选；历史其他评测曝光及可用媒体仍待核验，候选不等于已验收最终测试集。
- `prepare_subset.py`复用统一200的原生GT，建立独立sealed64，保持源媒体身份不变；预建各候选／阶段的REQUEST，checkpoint不存在时不启动推理。
- `eval_runtime.py`严格核对搜索checkpoint、source／world28、模型和Adam有限值，导出逐张量一致的不含Adam推理文件；实际构建相同全LoRA模型并strict加载。
- `evaluate28.py`／`binder.py`接入现有并行生成、解码与原生指标流程，仅使用前三机8卡及node11的0–3卡。复用统一SFT2752规划／hidden、Gemma、Whisper和首帧；同seed137与Euler30／CFG3、7。搜索各权重训练使用原始Qwen H，推理使用统一SFT2752 H，与已有测评一致。
- subset64不能沿用200条聚合分母：使用自己的sealed references和28评分分片。每个checkpoint生成metadata保留实际step，不伪装6000步。

已有统一指标主要提供画面质量、表情和音频覆盖。独立整段事件presence的连接与跨候选固定GT、晋级规则、阶段自动驱动仍在实现；不会将缺失事件指标或N/A音质转换为0，也不会仅按FVD或训练loss宣布最优。大模型GPU验证需要训练checkpoint出现并释放28训练卡后进行。
