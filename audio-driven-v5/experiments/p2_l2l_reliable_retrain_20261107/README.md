# L2L 可靠重训（2026-11-07）

用户已要求重新训练，并明确允许临时使用 node11 GPU7；结束后归还推理。
本轮只使用这张卡，不动其余31卡训练。旧两轮保留，不能作为有效重训基线。

当前状态：原轮VQ已在27749步完整保存并安全退出，最优1000步；后续码本初始化诊断3000步已结束，独立短预热predictor诊断569步已停止。全部仅作历史／诊断，不计正式基线。当前先完成论文／代码协议核对，没有自动恢复或test43队列；其他推理未动。


启动核验：主种子137的VQ第1000步已保存完整last／best，最优已推进到1000步；验证重建损失10.0445，相对训练均值基线12.2583改善约18.06%，总目标14.5789优于初始化14.8359。这里只是早期学习迹象，未完成正式学习验收。实测约11.17更新／秒；当前VQ到20万步上限约另需4.95小时，随共享负载变化，后续predictor与重复种子另计。[实际进程／检查点核验](RUNNING_VERIFIED.json)。

当前进度以 [进程及共享资源](LAUNCH.json)、[status.json](status.json)、[训练日志](train.log) 和
[预检结果](preflight/COMPLETE.json) 为准；没有结果文件不代表已完成。


[本次量化诊断、有限实验与停止规则](DIAGNOSIS_AND_STOP_POLICY.md) · [完整保存及停止证据](STOPPED_FOR_DIAGNOSIS.json)。


**依据纠正：** 官方README明确说明VQ loss可能先升后降，不能凭27000步的上升曲线就认定训练故障。初始化RMS/梯度差异也没有证明错误根因。码本初始化、短预热及固定步数门槛均为自定诊断，不是论文复现依据。短预热predictor已在569步停止；当前全部L2L训练停止，先核对原论文／补充与官方实现。

## 原轮已确认方案（现已停止，以下为历史配置）

- 沿用保守397对：train351／validation46，隔离1条与test共享speaker的训练样本。
  实际可组成64帧窗口的来源为326／41，窗口3637／336；全部46条验证片段用于完整rollout。
- 复用经过哈希核验的原始DECA／mel特征，按官方顺序先组成64帧窗口再双边滤波；
  只用训练窗口计算官方归一化统计。旧版整段滤波的统计不沿用。
- 直接使用官方工厂返回的 AdamW／Adam 与 ScheduledOptim，batch32、预热100000步。
  保留官方损失、网络、掩码采样与整批洗牌，不替换为固定学习率。
- 每阶段最多200000更新，至少150000后才允许按验证平台期停止。
  每1000更新验证，保存完整last／best和每10000更新的恢复快照。
- 主种子137，重复种子271／811；不按test成绩挑种子或采样结果。
  只有三轮验证学习检查都通过后，才自动执行test43生成与已有共同DECA评分。
- VQ必须优于初始化，重建相对训练均值基线至少改善5%，且使用至少2个码。
  predictor必须优于初始化，且比训练token unigram基线至少改善1%；
  完整source-only验证rollout须通过有限值／运动方差检查，逐片段恒定输出另行报告。
- 检查点保存模型、优化器、scheduler、全局与FACT内部RNG、采样顺序及位置、选择历史、
  代码和数据指纹。中断恢复必须通过与不中断运行逐项一致的真实网络测试。

## 入口与证据

- [冻结设计](../../docs/superpowers/specs/2026-11-07-l2l-reliable-retrain-design.md)
- [实现与验收计划](../../docs/superpowers/plans/2026-11-07-l2l-reliable-retrain.md)
- [数据核验及预处理记录](DATA_COMPLETE.json)
- [独立实现](../../compare/deployment/l2l_reliable.py)
- [真实官方更新／断点一致性／小样本学习预检](../../compare/deployment/l2l_reliable_preflight.py)
- [旧两轮审计](../p2_l2l_posttraining_inference_check_20261107/README.md)
- [现有统一测评及视频](../common_method_comparison_20261106/README.md)

正式进程使用本目录code_snapshot中的冻结代码；重启时核对代码和所有数据哈希。
用户随后允许与现有GPU7推理叠加；[共享资源授权与上限](RESOURCE_AUTHORIZATION.json)。
训练主进程由LAUNCH.json记录，CPU排队进程已停止；重启时可使用部署目录的l2l_reliable_launch.py --share，勿重复启动。
`STOP`文件或SIGTERM会在安全更新边界保存并退出。不要修改已冻结快照来强行续跑。
学习检查失败会保存FAILED_GATE／FAILED证据并释放GPU，不生成伪有效测试结果。

L2L输出为动作系数，没有公开写实renderer；本地重训不宣称复现DIM未公开的EMOCA设置。
COMPUTE_COMPLETE只表示计算结束；页面发布和浏览器验收另行标记，不混为完成。
