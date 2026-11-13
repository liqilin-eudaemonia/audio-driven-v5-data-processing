# J 点头检测＋官方门控内 ASR 发声线索 → Qwen-Omni 自然语言描述打标：固定归档 v4

固定版本标识：`listen-j-nod-vocal-asr-omni-caption-v4-20261103`（涵盖生产 `pipeline_id: 00ba25c37dfd76665753f084552b01d7c8721ce3e3792ad4309c77f0dd128a31` 与定版发声补充 `native-av-first-plus-nod-vocal-asr-v4`）。

本目录统一归档 **给出 25 fps GLARE 高召回点头检测结果及官方 VAD∪ASR 门控内真实发声线索（经三级完备 ASR 规范化规则）后，由 Qwen3-Omni-Instruct（30B MoE）一次性生成整段英文自然语言描述** 的全部定版源码、三级完备 ASR 规范化脚本、24 条三列对照实验、单元测试、冻结配置、Prompt 全文与交接说明（共 160 个文件）。

## 1. 核心入口与目录结构

| 用途 | 归档路径（`code/` 下保持原项目相对路径） |
|---|---|
| 点头＋三级完备 ASR 发声线索定版入口（v4） | [`code/scripts/run_listen_native_caption_vocal_pilot.py`](code/scripts/run_listen_native_caption_vocal_pilot.py) |
| 24 条三列有声对照页面构建脚本 | [`code/scripts/build_listen_native_caption_vocal_review.py`](code/scripts/build_listen_native_caption_vocal_review.py) |
| 24 条三列对照实验说明、v4 Prompt 与完整结果 | [`code/experiments/listen_native_caption_nod_vocal24_20261103/README.md`](code/experiments/listen_native_caption_nod_vocal24_20261103/README.md)、[`prompt.txt`](code/experiments/listen_native_caption_nod_vocal24_20261103/prompt.txt)、[`summary.json`](code/experiments/listen_native_caption_nod_vocal24_20261103/summary.json) |
| 三级完备 ASR 规范化与 v4 Prompt 单元测试 | [`code/tests/test_native_caption_vocal_asr.py`](code/tests/test_native_caption_vocal_asr.py) |
| 生产冻结配置与 40 个源码哈希 | [`code/experiments/listen_j_omni_20261030/config.json`](code/experiments/listen_j_omni_20261030/config.json) |
| 生产仅点头版 Prompt（`first_plus_evidence`） | [`code/experiments/listen_j_omni_20261030/prompt.txt`](code/experiments/listen_j_omni_20261030/prompt.txt) |
| 生产流水线说明与结果字典 | [`code/experiments/listen_j_omni_20261030/README.md`](code/experiments/listen_j_omni_20261030/README.md) |
| 50 条定版对照试验说明与配置 | [`code/experiments/listen_native_caption_first_evidence50_20261030/README.md`](code/experiments/listen_native_caption_first_evidence50_20261030/README.md) |
| Prompt 模板与四字段点头证据渲染 | [`code/src/listen_benchmark/native_caption_nod.py`](code/src/listen_benchmark/native_caption_nod.py) |
| Omni 推理、官方并集静音与点头校验核心模块 | [`code/src/listen_benchmark/j_omni.py`](code/src/listen_benchmark/j_omni.py) |
| 多机队列、预取、ROI 裁剪复用与尾部协作模块 | `code/src/listen_benchmark/j_omni_*.py`、`shared_raw_decode.py`、`frame_crop.py` |
| 25 fps 点头检测与跨机 CPU 分担入口 | [`code/scripts/listen_visual_native8_nod.py`](code/scripts/listen_visual_native8_nod.py)、[`code/scripts/listen_j_annotation_nod.py`](code/scripts/listen_j_annotation_nod.py)、[`code/scripts/listen_j_corrected_nod.py`](code/scripts/listen_j_corrected_nod.py) |
| 画质前置筛查与修正版裁框模块 | `code/src/listen_visual_quality/`、`code/src/listen_benchmark/j_visual_gate.py`、`j_quality_pipeline.py` |
| 多机 Worker、Producer 与 Watchdog 守护入口 | [`code/scripts/listen_j_omni.py`](code/scripts/listen_j_omni.py) |
| 全套交接文档（9 份） | `code/related/P2_*.md` |
| 归档清单与逐文件 SHA256（160 个文件） | [`manifest.json`](manifest.json) |
| 只读完整性与 SHA256 校验脚本 | [`verify.py`](verify.py) |

## 2. 打标方案与三级完备 ASR 发声补充规则（v4 统一版）

1. **视觉输入**：同一个 listener、修正版有界裁框缩放至 `512×512`、原生视频 `8 fps`，保留完整时长。
2. **音频输入与静音净化**：读取独立 listener 原始长音轨中对应 `[round(start*sr), round(end*sr)]` 区间，仅使用**官方 listener 词／句／VAD 时间区间并集**（前后各扩边 100 ms、10 ms 淡入淡出，区间外置零，不删减时长）。
3. **补充证据（点头候选 ＋ 发声区间与三级完备 ASR 线索 ＋ 全静音标记）**：
   - `SUPPLEMENTARY NODDING INTERVALS`：由同片段、同裁框 `25 fps` MediaPipe 姿态＋GLARE `recall` 参数检出，传 `id`、`start`、`end`、`observation`（`"head nodding"`）。
   - `SUPPLEMENTARY VOCAL RESPONSE INTERVALS`：在官方门控保留的真实发声区间内，合并提供我们跑过的 ASR（经官方门控交集＋多字中文过滤）与官方转写／音频事件候选，并经**三级完备规则（`normalize_asr_cue_for_english_prompt`）**规范化：
     - **Tier 0（纯英文）**：原样保留英文词句；
     - **Tier 1（有官方英文词）**：当 `asr_text` 含非 ASCII 字符但同窗口内有 `official_text` 时，直接采用官方英文词；
     - **Tier 2（无官方词时的封闭拟声词白名单）**：将纯拟声单字映射为精确英文拟声音节（`嗯/唔/हम्म/うん→"mhm"`、`姆→"mm"`、`哼/हम→"hm"`、`哈→"ha"`、`呵/嘿→"heh"`、`嘻→"hee"`、`哎/诶→"eh"`、`哦/噢→"oh"`、`啊/呀→"ah"`、`哇→"wow"`、`呃→"uh"`、`耶→"yeah"`、`啧→"tsk"`、`咳→cough/throat clearing`）；
     - **Tier 3（跨语种实义词谐音幻觉兜底，如 `对/是/好/我/não/é`）**：置 `asr_text=None` 并保留经过官方门控验证的发声时间窗 `{"start": a, "end": b, "observation": "audible spoken response or vocalization"}`，让 Omni 直接听辨该时间窗波形。
   - `listener_audio_is_silent`：由净化后实际送模波形是否全零生成；为 `true` 时严禁虚构可听说话或发声。
4. **模型与解码参数**：
   - 模型：`/private/ckpts/Qwen3-Omni/Instruct`（30B MoE 视听多模态指令模型）。
   - 解码：`temperature=0`、`max_tokens=4096`、首轮 `repetition_penalty=1.05`（异常最多重试一次，惩罚 `1.12`），结构化单段纯 ASCII 正则 `StructuredOutputsParams(regex=r"[A-Za-z][ -~]*")`，每次生成前调用 `llm.reset_mm_cache()`。

## 3. 校验与回归测试

### 3.1 校验归档与工作区源码 SHA256（160 个文件）

```bash
python3 releases/listen_j_nod_omni_caption_v1_20261103/verify.py
```

### 3.2 运行 CPU 单元测试（共 23 个测试文件、117 项测试）

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

# 1) 运行 22 个 J Omni / 点头 / 三级完备 ASR / 队列 / 解码裁剪单元测试（107 项）
/private/software/conda/envs/vllm/bin/python -m unittest \
  tests/test_native_caption_vocal_asr.py \
  tests/test_j_omni_pipeline.py \
  tests/test_j_omni_crop.py \
  tests/test_j_omni_prefetch.py \
  tests/test_j_omni_queued.py \
  tests/test_j_omni_sources.py \
  tests/test_j_omni_stream.py \
  tests/test_j_omni_tail.py \
  tests/test_j_omni_tail_recovery.py \
  tests/test_j_omni_visual_gate.py \
  tests/test_j_omni_batch_canary.py \
  tests/test_j_omni_watch_journal.py \
  tests/test_j_omni_watch_observation.py \
  tests/test_j_omni_watch_recovery.py \
  tests/test_j_nod_cpu_share.py \
  tests/test_j_visual_backfill.py \
  tests/test_j_quality_pipeline.py \
  tests/test_j_quality_parallel.py \
  tests/test_roi_frame_crop.py \
  tests/test_shared_raw_decode.py \
  tests/test_shared_visual_decode.py \
  tests/test_shared_decode_format_guard.py

# 2) 运行画质门控数值回归测试（10 项）
/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_omni_20261030/visual_quality_rollout/runtime_env/bin/python \
  -m unittest tests/test_listen_visual_quality.py
```
