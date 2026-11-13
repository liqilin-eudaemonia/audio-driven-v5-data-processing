**第一组提前在node11 GPU7评测（2026-11-11）** 用户要求第一组500步先评测，已接入GPU7的64条×双seed生成、解码、常规指标及事件评分。第二组28卡训练继续到500步，原控制器在该处交接，随后由gpu7_first.py完成第二组28卡评测并合并四组结果。GPU7进度见gpu7_first/STATUS.json，生成进度见evaluations/event020_feature010_500_seed137/status.json。

# Event × Feature 组合验证

用户确认于2026-11-11新增两组500步实验，Event为0.20／1.00，视频与音频Feature均为0.10。两组采用原始预训练初始化，与既有500步搜索相同的seed、batch28、284218段／500.001259h数据、最新标签、缓存及逐样本noise计划。

已有Event=0.20／1.00、Feature两路各0.025的500步权重与完整双seed评测结果作为对照。新旧四个组合均使用Seamless固定64条、生成seed137／238，Euler30、视频CFG3／音频CFG7、512分辨率。新增生成256条，256条控制组输出复用。

使用node12／node8／node4各0–7及node11的0–3共28卡。按两组各500步依次训练和完整验收，随后自动进入跨组共享任务池的生成／解码／视觉与音频指标／事件评分，最后写COMPARISON.json。评测沿用CPU预取、GT媒体／特征／事件标签缓存、每卡就绪继续领取等已验证实现。

extension.py在独立进程中注册两项配置，worker.py和evaluate28.py复用原生训练及评测实现。实验源码另外登记进checkpoint身份，原实验文件及检查点保留。初始化验收将两组step0全部模型张量逐值对照原baseline，并检查空Adam状态。

入口为 `driver.py run`。训练进度见STATE.json和runs/<candidate>/status.json，评测进度见STATUS.json／ROLLING_VIDEO_STATUS.json。控制器见EXPERIMENT_CONTROLLER.json。停止时写本目录STOP及正在训练的runs/<candidate>/STOP，训练会保存完整当前状态。历史Feature=0.10、Event=0.05组保持暂停在step80。

七项专用回归通过。实际缓存预检及GPU最长样本两次真实更新、step0初始化比较、500步完整checkpoint验收均由原生入口执行并留档。最终结果仍需结合覆盖率及配对置信区间解读。


2026-11-11 用户要求将原 node11 GPU7 的第一组评测并入四机28卡。旧 gpu7_first 调度已通过 STOP 安全退出；现由 rolling_driver.py / rolling_video.py 跨两组、双 seed 共享28卡任务池，node12/node8/node4 GPU0–7、node11 GPU0–3。保留92条已生成 latent（第一组 seed137 28条、第二组 seed137 64条），第二组64条解码结果同时保留。采样、checkpoint、seed、条件身份不变，逻辑分片由4统一为28。迁移凭据 migration28_20261111/MIGRATION_ACCEPTED.json；当前状态 STATUS.json、ROLLING_VIDEO_STATUS.json，最终 COMPLETE.json / COMPARISON.json。旧 gpu7_first/FAILURE.json 为本次主动停止引发的取消记录，不代表新调度故障。


2026-11-11 用户选定 event020_feature010 从完整 step500 续训至 step1500。promotion_1500.json 仅包含该候选；resume1500.py 调用原生 queue.run，使用原28卡、global batch28、冻结配置和样本/noise调度，严格恢复模型、Adam、28份RNG。每500步保存，1500保存验收后退出；没有3000/6000晋级文件，不自动测评、不启动其他候选。入口日志 resume1500.log，实时训练 STATE.json、runs/event020_feature010/status.json，最终凭据 runs/event020_feature010/STAGE_1500.json。此前 COMPLETE.json 与 COMPARISON.json 属于500步评测，不能用作1500步完成证据。


更新：用户将候选终点改为1661，以28×1661≈31×1500对齐旧基线训练样本量。resume1661.py完整恢复并训练到1661；1500仅按原500步间隔保存、不停止。完成后自动接 experiments/p2_interaction1500_dualseed28_20261111 的同64条双seed28卡比较。先前1500停训约定被本指令替代。
