# compare 官方推理部署

**三方法统一展示已接入：** [同页查看联合8000与DyStream的视频和评分、Learning2Listen四套动作结果](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。联合与DyStream可并排同步播放，L2L没有写实推理视频，单独展示DECA动作评分／曲线；全部43组已验证。

## 2026-11-06：当前可运行工作的ViCo test43已实测

| 工作 | 现在可运行的范围 | 本轮完成 |
|---|---|---|
| DyStream | 完整视频生成 | 43/43条视频＋官方画质／身份五项＋六组BFM运动评分 |
| Learning2Listen | 动作参数生成，作者未公开写实renderer | Conan／Fallon／Stephen／Trevor各43条，共172条；独立DECA动作评分 |
| DIM／ViCo baseline／MM2022／ReactFace | 缺核心生成checkpoint | 未执行有权重生成，不能计为测试完成 |
| OmniResponse／SpeakerVid | 现有公开仓库不包含完整生成链路 | 未执行论文生成模型 |

[实测完整结果](../../experiments/p2_compare_vico43_20261106/RESULTS.md) · [独立验收](../../experiments/p2_compare_vico43_20261106/FINAL_AUDIT.json) · [打开视频与动作曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。

DyStream按用户确认使用自身音频全零、speaker进入other分支、已有首帧crop=False；官方5步／CFG默认，seed137，512/25→256/30。它不生成listener声音。L2L额外使用speaker视频动作；四套权重严格配对，30FPS，作者issue2的128维mel，用户确认修正audio切片4t，标准化零历史与计算尾补齐。DECA与BFM空间分开，不混算图像指标。新增适配器在本目录 `vico_*.py`，旧模型与联合8000结果未改。六项官方当前HEAD／Releases／Issues复查证据在 [current_generator_audit_20261106.json](logs/current_generator_audit_20261106.json)。


目标机器：node8 / root@172.29.104.237:22，固定物理 GPU 7。工作目录和权重位于共享目录，不需要复制代码。20491 用作 DyStream Web 端口，SSH 使用 22。

## 当前状态

| 工作 | 官方权重 | 已验证范围 |
|---|---|---|
| DyStream | `DyStream/checkpoints/last.ckpt`、`DyStream/tools/` 已下载 | Python 3.11 / Torch 2.10.0，完整人像+音频→视频验证成功；Web 已启动 |
| Learning2Listen | Conan / Fallon / Stephen / Trevor 均已下载解包 | Conan 官方测试数据 32×64×56 动作参数推理及逐帧 pickle 保存成功；四组配对严格加载全部通过 |
| DIM | 未找到核心生成权重公开链接 | 独立环境导入 seq2seq / seq2seq_pretrain 成功，无法进行有权重推理 |
| ViCo baseline | 未找到核心 listener / speaker checkpoint 公开链接 | 独立环境 eval.py --help 通过，无法进行有权重推理 |

## 从任意共享目录节点调用

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5/compare
python deployment/run.py dystream-ui --port 20491
python deployment/run.py dystream-batch
python deployment/run.py l2l --config configs/vq/delta_v6.json --checkpoint models/delta_v6_er2er_best.pth --speaker conan --num-samples 32 --save
```

启动命令使用现有集群 SSH 密钥设置，不复制凭据。`node8.sh` 将 CUDA_VISIBLE_DEVICES 强制设为 7；PyTorch 内显示的 cuda:0 对应物理 7 号卡。运行时不使用官方 run.sh（其内部硬编码为 GPU 0）。

Learning2Listen 默认会测试 1024 个序列；`--num-samples 32` 是新增的可选限制，默认行为保持原样。输出位于 `learning2listen/src/outputs/<video>/results/<tag>predicted/*.pkl`，包含 DECA expression (50) 和 pose (6)。作者未发布写实渲染流水线，不能将这些参数文件称为视频结果。自定义数据需先按该项目 README 准备 DECA 人脸参数与 128 维 mel 特征，不能直接输入任意 wav 和人像。

Conan 配置：`configs/vq/delta_v6.json`；其余人物配置：`configs/vq/{fallon,stephen,trevor}_delta_v6.json`，对应 checkpoint 为 `models/{fallon,stephen,trevor}_delta_v6_er2er_best.pth`。`--speaker` 指定测试数据来源；当前只下载 Conan 测试所需的四个数组，不包含训练集。

DyStream 自定义批量输入依照 `DyStream/data_json/sample_files.json` 格式准备 JSON，通过 override `data.test_meta_paths=[/absolute/path/input.json]` 传入。支持单方音轨和双方独立音轨，具体字段以官方 README 为准。

## 权重来源和缺项核查

- DyStream 官方：[Hugging Face](https://huggingface.co/robinwitch/DyStream)，按 README 下载 tools/** 和 checkpoints/**；音频编码器 [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) 已预取到 node8 HF 缓存。
- L2L 官方：[README](https://github.com/evonneng/learning2listen#download-model)；四个 Google Drive ID 保存在 download_l2l.py。官方 HTTP 数据 [conan_data.tar](http://learning2listen.berkeleyvision.org/conan_data.tar) 支持 Range，只取测试所需数组；HTTPS 端点存在证书域名错误，采用 README 的原始 HTTP 地址。
- DIM：[官方项目页](https://boese0601.github.io/dim/)、[代码](https://github.com/Boese0601/Dyadic-Interaction-Modeling)、[请求预训练模型的 issue #2](https://github.com/Boese0601/Dyadic-Interaction-Modeling/issues/2)。Releases 为空，README 没有权重链接，issue 中未找到作者的下载回复。推理所需 VQ speaker、VQ listener、最终生成模型、适配 EMOCA 的 renderer 权重仍缺。当前代码已包含后来补齐的 models/lib。
- ViCo：[官方项目页](https://project.mhzhou.com/vico/)、[代码](https://github.com/dc3ea9f/vico_challenge_baseline)。Releases 为空；README 要求先训练再使用 `saved/baseline_listener/checkpoints/Epoch_500.bin` 等文件。PIRender、ArcFace 的权重不等同于该工作的核心生成权重。还需要训练集音频与视频的 mean/std、对应特征及渲染资源。
- 已额外检索公开网页和 Hugging Face model API；未找到与 DIM / ViCo 对应且有可验证来源的生成模型 checkpoint。不能据此断言任何地方都不存在。

`logs/*_issues.json`、`logs/*_issue*_comments.json`、`logs/*_releases.json` 保存本次 GitHub API 核查结果。不创建随机初始化结果来替代官方模型结果。

## 兼容处理

- DyStream 官方依赖有 `mediapipee` 拼写错误，改为 `mediapipe==0.10.21`；将 OpenCV 两个包统一到 4.11.0.86 以兼容其 NumPy 1.26.4 与 MediaPipe；补充缺失的 pytorch-fid。独立 requirements 在本目录。
- L2L 官方要求修改 torch 的 ConvTranspose1d 实现以绕过 replicate padding 校验；在项目自身该层改用 zeros padding，保持与作者所述绕过校验后实际 ConvTranspose1d 运算一致，不改共享 torch 安装。
- `.envs/l2l`、`.envs/dim`、`.envs/vico` 通过 venv 复用 node8 已有 lm-listener 的 PyTorch 2.5.1+cu124 基础包，各自额外依赖独立安装；DIM 目前仅完成模块导入验证，未验证其无权重的最终行为。

## 验证记录

- `logs/l2l_smoke32.log`：Conan 官方测试 32 段，输出 `(32, 64, 56)`，保存成功。抽查 pickle 的 exp `[1,50]`、pose `[1,6]` 均为有限值。
- `logs/l2l_trevor_smoke.log`：Trevor 模型以同一 Conan 输入测试生成 `(32,64,56)`；仅用于验证运行链路，日志中的 L2 不是论文指标，也不作为模型排名。
- `logs/l2l_weights_verified.json`：四个人物的 VQ 与 predictor 严格配对加载通过。
- `weights_manifest.json`：18 个模型与归一化统计文件的大小、SHA256，以及四个官方仓库的 commit。
- `logs/vico_help.log`：ViCo 原始 eval.py 参数解析验证通过。

- `logs/dystream_smoke_result.json`：人像和 2 秒音频完整推理生成 49 帧 512×512 H.264 + AAC，25 fps，1.96 秒；首次含模型加载共约 36 秒，峰值 PyTorch 分配显存约 6.44 GiB。抽取第 1 秒画面确认有人脸且无明显渲染损坏。样例见 `dystream_smoke.mp4`。
- Web 地址： http://172.29.104.237:20491 ，HTTP 页面和 `/config` 已验证。服务进程 node8 PID 3459084，日志 `logs/dystream_ui.log`。内网地址需要能访问该集群网络；可用 SSH 转发 `ssh -L 20491:127.0.0.1:20491 root@172.29.104.237`（使用你的集群登录凭据），然后浏览器打开 http://localhost:20491 。
- MediaPipe 在 headless 容器里 GPU delegate 无 EGL，已将 face_detector.py 的默认 delegate 改为 CPU；生成及神经渲染仍是 GPU 7。VGG16 官方权重已缓存并验证 SHA256 `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`。
- DyStream `pip check` 通过；各环境安装结果记录为 `requirements-*.lock.txt`。

官方 main.py 的 override 原本把 `data.test_meta_paths=[...]` 当成字符串逐字符读取；已增加 YAML 列表解析，只修复列表参数类型。两处仓库内兼容改动保存为 `dystream_compat.patch` 和 `l2l_compat.patch`。下载后的 face_detector.py 使用 CPU 的额外改动由 download_dystream.py 自动应用。

- `logs/dystream_batch.log`：官方 main.py 批量入口完整测试退出码 0，生成 `DyStream/outputs/20261105_123225_compare_gpu7/dyadic/single_reconstruct/compare_smoke_gpu7.mp4`（49 帧，512×512，H.264+AAC）。官方配置 eval_metrics=false，所以日志中 metrics={} 为预期；本次仅验证推理，不声称复现论文指标。

**整体尚未完成：DIM 与 ViCo 核心权重仍缺，详见 missing_weights.json；不能将仅安装环境视为推理部署成功。**

## 扩展公开权重核查

已通过 GitHub API 枚举并检查 DIM 的 10 个公开 fork、ViCo 的 12 个公开 fork：22 棵文件树均完整返回、无请求错误；候选模型文件仅有随代码附带的 LPIPS 小型权重，没有核心生成模型。逐个对比 fork README 与官方 README，也未发现新的权重下载链接。完整证据保存在 `logs/public_forks_checkpoint_audit.json`，核查脚本为 `audit_public_forks.py`。此外精确搜索代码中的最终 checkpoint 文件名仍未找到可用下载入口。

复查时 node8 Web PID 3459084 存活，HTTP 200，进程环境 CUDA_VISIBLE_DEVICES=7。DIM / ViCo 仍需要作者或已有持有者提供训练完成的核心 checkpoint，当前没有已知的链接可以交给用户手工下载。

目标已标记为受阻：同一核心权重缺失问题经三轮核验仍未解决。已完成的 DyStream Web/批量推理及 Learning2Listen 动作参数推理保持可用；DIM、ViCo 不计为部署完成。详细验收与阻塞证据见 `BLOCKED_AUDIT.json`。

## 新增四个项目（第二批）

已完成新项目 README、官方发布资源和权重入口核查，部署记录与可运行命令见 [ADDED_PROJECTS.md](ADDED_PROJECTS.md)。新增项目仍有核心生成代码／权重发布缺项；详细状态以 STATUS.json 为准。ReactFace 两个 Drive 文件夹返回 401，用户已申请权限；等待资源与核心 checkpoint。
