**七组500步全部双seed（用户确认，2026-11-10）：** 每组固定同64条，生成seed137／238，共896条；原seed137队列不中断，完成七组后由`dual_seed_queue.py`在node11 GPU7继续七组seed238。导出模型和SFT／Gemma／Whisper条件缓存直接复用；检测器seed保持137。比较分别报告每seed结果及均值，FID／FVD各自在64条上计算后平均，配对来源组bootstrap将两次生成一起保留，不把它们当128个独立样本。40项CPU检查通过；seed238实际GPU验收与完整评分待运行。旧自动晋级仍关闭。最终报告`DUAL_SEED_COMPARISON.json`；本队列停止标记`DUAL_SEED_STOP`，不改变训练。配置见[双seed冻结计划](DUAL_SEED_PLAN.json)。

**2026-11-10改为node11 GPU7边训练边评测：** 已完成候选无需等待七组全部结束，按baseline优先顺序评分已有检查点。直接复用历史`p2_all_lora_followups_gpu7_20261109.py`调度（4路交错解码、2路评分、显存准入、OOM降并发、就绪分片提前评分）；仅适配64／200样本数及搜索权重严格加载。SFT H／Gemma／Whisper、GT RGB／特征缓存、CPU预取2／像素4线程、GPU人脸／表情及batch64保持。34项回归和64条四分片原生绑定核验通过。实际GPU进度见`evaluations/baseline_500/status.json`，尚无评分结果时不得声称排名。

# 搜索专用统一生成测评接入

本目录独立于训练源码指纹。分阶段生成评分、保守筛选和恢复训练驱动已实现，31项CPU检查通过，两个接续控制器已启动待命；实际搜索checkpoint的GPU生成、检测与跨阶段恢复仍未验收，不能称为已完成搜索。

- `freeze_validation.py`核对最终500h源数据库，检查统一200与训练的sample／pair／component重叠，冻结分层64调参子集。另选最多128条不同component的最终保留候选；历史其他评测曝光及可用媒体仍待核验，候选不等于已验收最终测试集。
- `prepare_subset.py`复用统一200的原生GT，建立独立sealed64，保持源媒体身份不变；预建各候选／阶段的REQUEST，checkpoint不存在时不启动推理。
- `eval_runtime.py`严格核对搜索checkpoint、source／world28、模型和Adam有限值，导出逐张量一致的不含Adam推理文件；实际构建相同全LoRA模型并strict加载。
- `evaluate28.py`／`binder.py`接入现有并行生成、解码与原生指标流程，仅使用前三机8卡及node11的0–3卡。复用统一SFT2752规划／hidden、Gemma、Whisper和首帧；同seed137与Euler30／CFG3、7。搜索各权重训练使用原始Qwen H，推理使用统一SFT2752 H，与已有测评一致。
- subset64不能沿用200条聚合分母：使用自己的sealed references和28评分分片。每个checkpoint生成metadata保留实际step，不伪装6000步。

- `run_events.py`在生成评分全部结束后，使用已释放的node12 GPU7运行独立整段事件检测；输入包含生成的自身WAV。检测器提供4类视觉和3类声音presence，与训练Event头独立；它尚未经过人工可靠性标定，不能将结果解释为语境适当性或音画同步分数。
- `reference_events.py`从baseline首次结果逐ID冻结参考事件标签；后续候选只替换生成侧，64扩展到200时仅补充新ID。媒体或检测协议改变会拒绝，未知／错误标签不转换成负例。
- `reference_media_cache.py`严格复用相同参考媒体的候选／检测结果，避免每个权重重复做GT检测。复用前检查独立解码得到的像素／自身音频指纹、政策／源码、媒体和检测文件哈希；原生推理仍核验模型／版本／cache identity。生成侧保持自身媒体；只有独立准备后的内容指纹完全相同，才按原生content ID去重。参考标签不会因复用变成新标签或负例。
- `stage_driver.py --target 500`等待七组500步完整检查点和训练队列锁释放，再逐组在同一28卡范围生成并评分；后续阶段支持1500／3000／6000。占用卡只等待，不终止其他进程；node11 GPU4–7继续保留。阶段评分完成后写`generation_review_ready`，本驱动不凭训练loss自行晋级。

固定64条的音频分组为静音23、短时发声23、较长发声18；是覆盖导向分层抽样，并非200条原始比例的无偏缩影，最终筛选需要完整200复核。不会将缺失事件指标或N/A音质转换为0，也不会仅按FVD或训练loss宣布最优。

`PROMOTION_POLICY.json`已在查看任何候选生成结果前冻结：视觉／音频事件宏F1等权平均作为反应指标，同时检查逐类F1、LPIPS、PSNR、SSIM、错误发声与漏发声。跨候选使用相同的可评标签交集，逐类共同覆盖和GT正例保留均至少90%；未知／错误不充当负例。

按来源component进行2万次配对bootstrap，单阶段按有序候选对数调整区间尾概率。500步仅在反应增益区间下界达到0.05、其他指标均满足非劣约束时允许淘汰被压过的候选；1500／3000使用0.02，且要求同一候选对在前一阶段也形成明确优势。工程容许退化为：两模态F1各0.02、逐类F1 0.05、LPIPS 0.01、PSNR 0.25dB、SSIM 0.01、错误／漏发声率各0.02。这些是预声明筛选阈值，不是已验证的感知界限或跨阶段统计证明。

保留baseline对照；差异不清楚或覆盖不足时保留候选，不硬选top3／top2。超出预算目标的候选数和更新步数写入`DECISION_<step>.json`。后续`search_controller.py`依次执行评分→证据晋级→恢复训练→下一阶段评分，保留每组自己的模型、Adam、28份RNG；STOP及来源变化会阻止接续。

接续不会重写已完成评分或晋级证据：先校验checkpoint、执行源码、逐样本指标、参考标签及生成视频／自身WAV哈希，再直接复用完成记录。已冻结决定同时绑定规则和决策代码指纹；来源或证据改变则拒绝，保持不变则复用，避免重算或时间戳变化使后续不可变晋级文件失效。该路径已纳入回归检查，实际跨阶段GPU接续仍待运行验收。

6000步完成后输出`FINALISTS_READY.json`，不自动宣称最佳模型。独立最终保留集、增加生成种子、与已有最好模型比较和对尚未覆盖的感知／语境指标确认仍待完成；检测器尚未人工标定。大模型GPU验证需要训练checkpoint出现并释放28训练卡后进行。

补充实际检查：使用历史完整生成的64条视频＋自身WAV，重放本次28分片的发布／原生全量绑定，逐行一致；媒体和原模型7500步身份保持原样，不是本轮候选生成或分数。[CPU实际媒体绑定验收](../p2_loss_search28_eval_cpu_bind_audit_20261110/ACCEPTED.json)。

另选静音／短发声各1条历史完整媒体，实际执行事件准备和CPU候选检测，参考／生成共4份媒体均成功，face coverage均为1；真实vLLM环境的processor和结构化输出接口检查通过，未运行模型推理。[实际CPU链路](../p2_loss_search28_event_cpu_canary_20261110/ACCEPTED.json) · [运行环境接口](../p2_loss_search28_event_cpu_canary_20261110/INTERFACES_ACCEPTED.json)。最长几何96帧／512×512的CPU token化检查中，0／24条候选提示分别需要14213／14983 token（含1536输出重试预留），低于16384；这是所测两种提示长度的检查，不代表无限提示长度或GPU模型已验收。[上下文预算](../p2_loss_search28_event_cpu_canary_20261110/CONTEXT_BUDGET.json)。

除500h生成训练数据外，已核对推理规划器SFT2752的704368条train及5296条dev，当前200条在sample／pair／component层面也无交集。[规划器数据审计](PLANNER_MEMBERSHIP_ACCEPTED.json)。额外冻结的109条只是当前搜索之外的保留候选：其来源component存在历史生成记录，严格历史独立性尚不能确认，未投入最终评分。[候选与限制](../p2_loss_search28_holdout_20261110/README.md)。
