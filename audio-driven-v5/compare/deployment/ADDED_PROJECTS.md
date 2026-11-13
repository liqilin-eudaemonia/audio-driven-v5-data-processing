# 新增四个工作的部署记录

统一节点 `172.29.104.237:22`，物理 GPU 7。所有生成模型的要求仍保留；安装依赖、下载辅助模型不等于完成生成推理。

| 项目 | 完成的部署工作 | 无法完成生成推理的原因 |
|---|---|---|
| ReactFace | `.envs/reactface`；原生自定义推理 `--help` 通过；PyTorch3D 编译扩展加载通过；FaceVerse V2 在 GPU 7 加载并初始化成功 | 核心 `best_checkpoint.pth` 无公开下载链接；官方 PIRender 和参考参数两个 Drive 文件夹返回 401，用户已申请权限 |
| MM2022-ViCoPerceptualHeadGeneration | 根据 README 将四个修改文件集成到独立 `deployment/workspaces/mm2022`，修改 grid_sample 边界填充；复用 ViCo 独立环境，`eval.py --help` 通过 | 只发布基线修改，没有训练后的核心 driver checkpoint；官方 U2Net 分割权重无法自动获取，链接已提供给用户 |
| OmniResponse | 已核查 README、全部已提交文件、公开 Releases / Issues / 分支 / HF 模型检索 | 官方 TODO 明确未发布推理代码、renderer、模型 checkpoint，仓库只有文档和评测代码；不能部署论文中的生成模型 |
| SpeakerVid-5M-Code | 官方 63 个 curation/metrics 文件已全部下载并校验通过；`.envs/speakervid`；原生亮度检测与 GPU 7 上原生 FVD 特征提取已通过 | 仓库发布的是数据整理和评测流水线，不包含论文所述 AR 视频对话生成器；这些工具不能替代生成模型作效果比较 |

## 可调用入口

从 compare 目录执行，自动 SSH 到 node8 并固定 GPU 7：

```bash
# 仅显示参数，不需要核心权重
python deployment/run.py reactface --help
python deployment/run.py mm2022 --help

# ReactFace：等核心 checkpoint 和所有 renderer/reference 资源齐全后使用
python deployment/run.py reactface \
  --speaker-video /absolute/path/speaker.mp4 \
  --speaker-audio /absolute/path/speaker.wav \
  --listener-portrait /absolute/path/portrait.png \
  --checkpoint /absolute/path/best_checkpoint.pth \
  --output-dir /absolute/path/output --window-size 8 --momentum 0.9

# SpeakerVid 原生亮度检测，仅数据工具，不是视频生成
python deployment/run.py speakervid-luminance \
  --total 1 --index 0 --num_proc 1 \
  --video_list_path /absolute/path/video_list.txt \
  --save_dir /absolute/path/luminance_output
```

ReactFace 的 FaceVerse 2 原始下载是一个 NPY 文件（并非 zip），已放到 `ReactFace/external/FaceVerse/data/faceverse_simple_v2.npy`。来源为 [FaceVerse 官方 README](https://github.com/LizhenWangT/FaceVerse) 提供的 [V2 下载](https://drive.google.com/file/d/1_ooP9hvR7kUUO8WhtXRU_D4nM5fr8BT_/view)。Wav2Vec2 预训练编码器已链接到 node8 的 `facebook/wav2vec2-base-960h` 缓存。

等待权限或手动下载的官方链接：

- [ReactFace renderer](https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA)：需要 `cur_model_fold.pth`，放 `ReactFace/external/PIRender/`。
- [ReactFace reference files](https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4)：需要 `mean_face.npy`、`std_face.npy`、`reference_full.npy`，放 `ReactFace/external/FaceVerse/`。
- [MM2022 U2Net](https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P/view)：`u2net_human_seg.pth`，放独立工作副本的 `PIRender/`。

以上都不是 ReactFace 或 MM2022 的核心 driver 模型；获取它们后仍需要匹配的训练完成 checkpoint。

## 验证证据

- `logs/ReactFace_public_audit.json` 等四个文件：GitHub API 的 Releases、Issues、分支结果。四仓库 Releases 均为空；本次相关 Issues 中未找到额外权重链接。
- `logs/*_HF_model_search.json`：精确项目名与 VidChat 模型检索，只有 SpeakerVid 的数据处理辅助模型仓库。
- `logs/reactface_help.log`、`logs/faceverse_smoke.log`、`logs/mm2022_help.log`：环境及入口验证。
- `logs/speakervid_luminance/`：原生亮度检测真实视频结果。
- `logs/speakervid_fvd_verified.json`：真实视频 32 帧 FVD 特征输出 `[1,400]`，所有数值有限，设备 H200 / 物理 GPU 7。这不是多视频 FVD 总分，也不是生成效果排名。
- SpeakerVid 原生 FVD 加载代码的绝对开发机路径改为当前仓库相对路径，公开 assets 链接到 README 要求的目录。算法和特征计算未修改。

官方来源：[ReactFace](https://github.com/lingjivoo/ReactFace)、[MM2022](https://github.com/hzwer/MM2022-ViCoPerceptualHeadGeneration)、[OmniResponse](https://github.com/awakening-ai/OmniResponse)、[SpeakerVid](https://github.com/Dorniwang/SpeakerVid-5M-Code)、[SpeakerVid 工具模型](https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models)。

ReactFace 的 `pip check` 已通过；其依赖调整仅写入 `.envs/reactface`。SpeakerVid 的 63 个资源文件全部通过大小校验，其中 LFS 文件逐个验证官方 SHA256，见 `logs/speakervid_assets_verified.json`。

ReactFace 真实视频输入处理验证通过：8 帧窗口输出 `[8,3,224,224]`，音频 `[5120]`，见 `logs/reactface_input_smoke.log`。MM2022 的 border padding 使用越界采样测试在 GPU 7 验证通过，见 `logs/mm2022_border_verified.log`。这些是组件测试，不作为完整模型生成成功的证据。
