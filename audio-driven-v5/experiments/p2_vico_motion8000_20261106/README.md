# ViCo test43 运动指标补测

[打开全部43条实际生成可视化](http://localhost:8876/listen_benchmark_review/p2_vico43_combined8000_20261106/)：三路同步视频，原生512／评分256、真实音轨、逐条指标和规划文本。[发布记录](VISUALIZATION_PUBLISHED.json) · [浏览器验收](VISUALIZATION_BROWSER_AUDIT.json)。全部43条两种生成分辨率、86路GT／speaker预览、音频来源SHA、同步播放／拖动／变速／切音轨、短片段N/A及手机布局检查通过。

状态：**43条全部完成，129路／48,864帧提取、六组指标和独立最终验收通过；GPU7已释放。** [结果表](RESULTS.md) · [完整JSON](RESULTS.json) · [CSV](summary.csv) · [最终验收](FINAL_AUDIT.json)。最新状态见 [status.json](status.json)。原画质／身份五项评分已经完成并封存，见 [原结果](../p2_vico256_combined8000_20261106/RESULTS.json)。本目录独立增加运动指标，不修改原始结果。

模型为 AV＋0.05 Event＋0.025 video Feature＋0.025 audio Feature＋speaker Whisper 联合模型 step8000。复用全部43条原生512×512／25FPS生成结果对应的256×256／30FPS评分副本。GT与speaker也为256×256／30FPS；129路视频共48,864帧，三路时间原点和帧数完全对应。原生25→30FPS采用最近时间戳，会引入重复帧并影响运动差分与相关性；原始25FPS生成及映射仍在父目录保留。

本次是**公开指标定义的复现，并采用用户明确确认的补测参数**，不能声明与 CustomListener 表1严格相同。其官网 Code 链接目前是占位，未取得作者完整评分实现、确切提取权重及评分归一化配置。

| 项目 | 冻结设置 |
|---|---|
| 3DMM | ViCo官方仓库的Deep3DFaceRecon，ResNet50、epoch20，严格加载权重 |
| 人脸点与对齐 | face-alignment1.3.5 TWO_D，SFD，首个人脸；官方逐帧FAN；原align_img，224输入、rescale102 |
| 提取精度 | FP32，关闭TF32；原始网络系数，无z-score或按GT拟合 |
| 系数 | exp64＋angle3＋trans3；pose指angle＋trans；完整257维另存 |
| FD | 每条视频内逐帧逐坐标L1平均，exp／angle／trans分别报告 |
| RTLCC | speaker-GT与speaker-pred的逐坐标Pearson相关曲线L1；延迟−60…60帧，步长1帧；正延迟表示listener滞后；完整延迟范围覆盖42/43条，1秒片段N/A |
| RWTLCC | 完整120帧窗口，每30帧一个；不补齐残余窗口；窗口等权，再视频等权；覆盖41/43条，1秒与3秒片段N/A |
| 常量序列 | 相关性分母≤1e−12置0，记录未定义坐标数 |
| FID_fm | 对每条视频的系数分布计算高斯Fréchet平方距离；样本协方差ddof1，再视频等权 |
| FID_delta_fm | 同上，改用相邻帧系数差分；不跨视频，不除以dt |
| SND | 原始尺度FID_fm＋FID_delta_fm |
| V-D | 时间总体方差ddof0，坐标平均、视频等权；附GT参考方差 |
| 展示倍率 | FD与FID_delta_fm乘100；其余不乘；保留所有原始值 |
| GPU | 仅node8物理GPU7；4个独立提取进程；CPU评分与审计 |

批量FAN及共享模型的并发FAN在真实视频核验中均产生过实际裁脸差异，已拒绝用于正式结果，检查记录保留在 [revisions](revisions)。正式采用单进程内逐帧FAN、跨视频独立进程并行；批量重建使用原网络。三路小批各64帧、抽查24帧，上游逐帧关键点和裁脸像素完全相同，batch1与批量重建的系数误差最高1.10e−5，见 [BATCH_PARITY_smoke_speaker.json](BATCH_PARITY_smoke_speaker.json)、[GT](BATCH_PARITY_smoke_gt.json)、[prediction](BATCH_PARITY_smoke_prediction.json)。正式每个worker首条视频还会重复抽检。

人脸检测失败沿用官方前向填充；开头缺失使用标准5点回退；整条视频均无人脸直接报错。逐帧检测掩码、多脸数量、对齐参数均缓存，不掩盖覆盖率。

协议与来源：[protocol.json](protocol.json) 与 [SCORING_PROTOCOL.json](SCORING_PROTOCOL.json)。两条短片段的N/A及覆盖数方案经用户明确确认；原提取协议和全部特征回执保持不变，评分修订独立封存，原始源码按原SHA存于 `frozen_source/`。协议包含每路输入、权重、BFM基、代码和父结果SHA256。提取缓存位于 `features/{speaker,gt,prediction}/{uuid}.npz`，保留 `motion`、`coeff`、`landmarks`、`detected`、`face_counts`、`transform_params`；同名JSON记录时间及SHA。已生成 `RESULTS.json`、`summary.csv`、`FINAL_AUDIT.json` 和 `COMPLETE.json`。

实现：[提取](../../scripts/p2_vico_motion_extract.py)、[指标](../../scripts/p2_vico_motion_metrics.py)、[控制器](../../scripts/p2_vico_motion_run.py)、[独立审计](../../scripts/audit_p2_vico_motion.py)。7项数值测试涵盖短片段N/A、解析Fréchet距离、秩亏协方差、已知时延、常量序列、方差、GT自比较和错误对齐；独立最终审计还使用SciPy协方差积sqrtm与NumPy corrcoef核查真实结果。

重算已有系数无需GPU：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 compare/.envs/vico/bin/python \
  scripts/p2_vico_motion_metrics.py --work experiments/p2_vico_motion8000_20261106
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 compare/.envs/vico/bin/python \
  scripts/audit_p2_vico_motion.py --work experiments/p2_vico_motion8000_20261106
```

定义来源：[CustomListener §4.2–4.3及表1](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_CustomListener_Text-guided_Responsive_Interaction_for_User-friendly_Listening_Head_Generation_CVPR_2024_paper.pdf)、[TH-PAD §4.1（逐片段FID及SND）](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Talking_Head_Generation_with_Probabilistic_Audio-to-Visual_Diffusion_Priors_ICCV_2023_paper.pdf)、[ViCo官方提取代码](https://github.com/dc3ea9f/vico_challenge_baseline)。
