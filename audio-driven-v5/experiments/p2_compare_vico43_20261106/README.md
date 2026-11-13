# compare 参考工作 ViCo test43 实测

**最新统一展示：** [联合模型8000／DyStream／Learning2Listen三方法评分与视频对照](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。两种实际推理视频与speaker、GT四路同步；L2L四套动作结果同页。对应展示验收：`three_method_review/BROWSER_AUDIT.json`。


[打开DyStream全部43条视频与Learning2Listen四套动作曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/) · [浏览器验收](VISUALIZATION_BROWSER_AUDIT.json) · [输入一致性审计](SOURCE_ONLY_INPUT_AUDIT.json) · [独立L2L数值核对](L2L_NUMERICAL_AUDIT.json)。


**当前可运行的参考模型已完成：DyStream视频43条；Learning2Listen四套人物模型各43条，共172条动作输出。另六项仍缺核心生成权重或生成代码，没有计为测试完成。**

[完整JSON](RESULTS.json) · [独立验收](FINAL_AUDIT.json) · [模型状态](MODEL_STATUS.json) · [画质CSV](image_summary.csv)

采用用户确认设置：DyStream target/listener自身音频为全零，真实speaker音频进入other分支，已有listener首帧、官方crop=False；5步／CFG默认值、seed137，EMA权重。DyStream输出动作／视频，不生成listener声音，MP4音轨仅为speaker上下文。Learning2Listen额外使用speaker视频动作，四套person-specific权重全测，不做按GT择优。

## 同一test43的画质与身份

| 模型 | PSNR ↑ | SSIM ↑ | 图像FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| 联合模型8000 | 19.668293 | 0.662447 | 30.116312 | 0.596836 | 0.139126 |
| DyStream listener适配 | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |

两组均原生512×512／25FPS，评分256×256／30FPS。DyStream官方推理原生每条少1帧，保持原始结果，评分映射以最近时间戳取帧并夹到末帧，不改变速度；最大时间误差 0.0533秒。所有43条实际评分副本共16,288帧，已独立解码核对。256评分从官方返回的原始RGB直接抗混叠缩小，避免把原生MP4二次解码损失带入评分。

## BFM运动指标：联合模型与DyStream

同一Deep3DFaceRecon权重、原始系数和GT／speaker特征，exp64＋angle3＋trans3；pose为angle＋trans。FD与FID_delta_fm展示乘100，SND不乘100。±2秒lag、4秒窗口／1秒步长、逐片段等权；RTLCC42条、RWTLCC41条，其余43条。属于已明确参数的公开定义复现，不宣称严格复现CustomListener表1。

| 模型／指标 | exp | angle | trans | pose |
|---|---:|---:|---:|---:|
| 联合8000 FD | 13.112942 | 10.219366 | 6.493092 | — |
| 联合8000 RTLCC | 0.225741 | — | — | 0.269646 |
| 联合8000 RWTLCC | 0.292209 | — | — | 0.332713 |
| 联合8000 FID_delta_fm | 3.714983 | — | — | 0.084429 |
| 联合8000 SND | 2.754344 | — | — | 0.120995 |
| 联合8000 V_D | 0.006784 | — | — | 0.003811 |
| DyStream FD | 15.456253 | 11.240375 | 7.004782 | — |
| DyStream RTLCC | 0.230798 | — | — | 0.268920 |
| DyStream RWTLCC | 0.298834 | — | — | 0.336451 |
| DyStream FID_delta_fm | 3.589592 | — | — | 0.046527 |
| DyStream SND | 3.324052 | — | — | 0.106503 |
| DyStream V_D | 0.005995 | — | — | 0.001988 |

DyStream生成视频漏检总数 0 帧，按同一官方前向填充策略处理，不丢弃视频；完整掩码／关键点／系数在 `dystream/motion/features/prediction/`。关键点复核要求实际网络输入像素一致、系数误差<1e−3；未用于对齐的局部关键点差异单独记录，不隐瞒为完全相同。

## Learning2Listen：独立DECA动作评测

DECA50维表情＋3维头部旋转向量＋3维下颌旋转向量，与上表BFM空间不同，没有平移系数，不混入同一排名。原生30FPS。没有作者的写实renderer，因此PSNR／SSIM／图像FID／CSIM／CPBD均不计算，也不拿3D示意当写实结果。

| 人物模型 | 表情L1 ↓ | 头旋转L1 ↓ | 下颌L1 ↓ | 表情Fréchet ↓ | 头旋转Fréchet ↓ |
|---|---:|---:|---:|---:|---:|
| conan | 0.288010 | 0.148715 | 0.040810 | 7.096596 | 0.092644 |
| fallon | 0.304757 | 0.179173 | 0.048678 | 8.074607 | 0.128112 |
| stephen | 0.267368 | 0.161168 | 0.047367 | 6.482851 | 0.107398 |
| trevor | 0.261020 | 0.181103 | 0.033099 | 5.468226 | 0.161624 |

完整L1／L2／Fréchet／差分Fréchet／SND／V-D／RTLCC／RWTLCC及逐片段结果见 [learning2listen/RESULTS.json](learning2listen/RESULTS.json)。四模型各43条，RTLCC各42条、RWTLCC各41条。

用户明确确认在独立适配器中修正官方循环音频切片`t`→`4*t`；网络、四套VQ／predictor权重及nucleus sampling不改。前置32帧标准化零历史，末端只补齐VQ计算窗口，输出裁回原始时段；不把初始GT或listener未来输入模型。DECA官方粗编码器严格加载，只提系数，不需要下载FLAME几何或渲染器。音频使用[作者issue2的更正代码](https://github.com/evonneng/learning2listen/issues/2)：16kHz、400窗／160步／512FFT、128维原始mel功率，Lanczos缩放到4T；不是MFCC或log-mel。

## 暂时不能生成的六项

| 项目 | 当前具体缺项 |
|---|---|
| DIM | 未找到核心VQ speaker/listener与最终生成checkpoint的公开可用入口 |
| ViCo baseline | listener/speaker driver checkpoint及配套训练统计未发布 |
| MM2022 | 只发布基线改动，没有训练完成的核心driver checkpoint |
| ReactFace | 核心best_checkpoint.pth没有公开链接，renderer/reference也仍需权限 |
| OmniResponse | 官方推理生成代码、renderer、checkpoint尚未发布 |
| SpeakerVid | 已发布的是数据处理／评分工具及其辅助权重，没有对话生成器 |

重新检查了六个仓库当前HEAD完整树、Releases和相关Issues／评论，未发现新增可用核心生成权重；当前HEAD与已部署版本一致。[本轮官方来源核查](../../compare/deployment/logs/current_generator_audit_20261106.json)。不将环境安装、辅助权重或help成功计为完整生成。

## 文件与复跑

- DyStream：`dystream/generated/{uuid}/generated.mp4`，`dystream/scoring/dystream/{uuid}.listener.mp4`；每条complete.json保存来源、原生帧数和完整时间映射。
- Learning2Listen：`learning2listen/generated/{conan,fallon,stephen,trevor}/{uuid}.npz`，包含30FPS的T×56原始motion；配套逐条JSON。
- 协议：[DyStream](dystream/protocol.json)、[DECA](learning2listen/DECA_PROTOCOL.json)、[L2L](learning2listen/protocol.json)。
- 独立适配器：`compare/deployment/vico_dystream.py`、`vico_deca_extract.py`、`vico_l2l.py`、`vico_metrics.py`、`vico_finalize.py`。仅node8物理GPU7，未动0–6卡训练。

原联合模型的生成、评分和封存验收未修改；所有参考工作均使用同一43条正式test，不含ood42或train398。
