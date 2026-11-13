# 点头候选＋官方门控内 ASR 发声线索融入 Omni 自然语言描述：v4 统一版与 24 条三列对照

日期：2026-11-03。固定版本：`native-av-first-plus-nod-vocal-asr-v4`（`prompt_style: first_plus_nod_and_vocal_asr_en_v4`）。

按用户要求，借鉴“官方 listener 词／句／VAD 并集确认真实发声区间并静音其余部分”的逻辑，在真实发声区间内补充我们跑过的 listener ASR（经官方门控交集＋多字中文过滤＋**三级完备 ASR 规范化规则**）与官方转写／音频事件线索，交由 `Qwen3-Omni-Instruct`（30B MoE）在**不输入旧 caption** 的前提下，根据原生 `8 fps` 视频、净化音轨、点头候选和发声线索一次性生成整段英文自然语言描述。

- [三列可播放有声对照 HTML（端口 8876）](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_vocal24_20261103/)
- [本轮 v4 Prompt 全文](prompt.txt) · [逐条完整 Prompt](prompts/) · [逐条 v4 输出与证据](results/) · [上一版原 ASR 提示输出备份](results_v1_raw_asr.json) · [三版汇总对比 JSON](summary.json)
- 准备与推理入口：[`scripts/run_listen_native_caption_vocal_pilot.py`](../../scripts/run_listen_native_caption_vocal_pilot.py)
- 页面构建入口：[`scripts/build_listen_native_caption_vocal_review.py`](../../scripts/build_listen_native_caption_vocal_review.py)
- 单元测试：[`tests/test_native_caption_vocal_asr.py`](../../tests/test_native_caption_vocal_asr.py)

---

## 1. 三级完备 ASR 规范化规则（`normalize_asr_cue_for_english_prompt`）

针对全量 722,706 段数据的全部 **21,393 种音频事件原文（288,425 个有效音频事件）** 扫描验证，采用以下四层（Tier 0 + 三级非 ASCII 漏斗）确定性规则处理 `asr_text`，确保送入 Prompt 的证据 **100% 纯 ASCII 英文且零遗漏**：

1. **Tier 0（纯英文 / 弯引号规整后纯英文，占 20,074 种 / 248,270 个事件）**：
   - 将 Unicode 弯引号/破折号规整为标准 ASCII 后，若全部为 ASCII 字符则原样保留英文词句。
2. **Tier 1（有官方英文转写 `official_text` 时优先采用官方英文词）**：
   - 当 `asr_text` 含非 ASCII 字符但同一发声窗口内已有官方英文词（如官方为 `"Sure."`、`"Mm-hmm."`、`"Yeah."`、`"Wow."`），直接以 `official_text` 为准，消解跨语种谐音偏差。
3. **Tier 2（无官方词时，封闭拟声词白名单映射为纯粹英文拟声音节，占 24 种 / 30,145 个事件）**：
   - `"嗯"` / `"唔"` / `"हम्म"` / `"हमम"` / `"うん"` → `asr_text="mhm"`, `observation="affirmative vocalization"`
   - `"姆"` → `asr_text="mm"`, `"哼"` / `"हम"` → `asr_text="hm"`
   - `"哈"` → `asr_text="ha"`, `"呵"` / `"嘿"` → `asr_text="heh"`, `"嘻"` → `asr_text="hee"`（`observation="laughing syllable"`）
   - `"哎"` / `"诶"` → `asr_text="eh"`, `"哦"` / `"噢"` → `asr_text="oh"`, `"啊"` / `"呀"` → `asr_text="ah"`, `"哇"` → `asr_text="wow"`, `"呃"` → `asr_text="uh"`, `"耶"` → `asr_text="yeah"`, `"啧"` → `asr_text="tsk"`, `"咳"` → `asr_text=None, observation="cough or throat clearing"`
4. **Tier 3（白名单之外的跨语种实义词谐音幻觉兜底，占 1,295 种 / 10,010 个事件）**：
   - 对单字中文实义词（如 `"对"`、`"是"`、`"好"`、`"我"`、`"你"`）或其他语种谐音词（如 `"não"`、`"então"`、`"é"`），当窗口内无官方英文词时，自动置 `asr_text=None` 并保留经过官方门控验证的发声时间窗 `{"start": a, "end": b, "observation": "audible spoken response or vocalization"}`，让 `Qwen3-Omni-30B` 直接根据该时间窗内的音频波形听辨描述，既不漏掉发声区间，也不被错字带偏。
5. **解码层结构化正则约束**：
   - vLLM `StructuredOutputsParams(regex=r"[A-Za-z][ -~]*")` 锁死单段纯 ASCII 可打印字符输出。

---

## 2. v4 统一版 Prompt 设计

- **正文 1–5 段与 `first_plus_evidence` 逐字完全一致**，避免改动主指令段落引起的英文样本描述波动。
- **末尾补充证据说明**：
  ```text
  Supplementary nod intervals: id is an identifier; start/end are pair-relative seconds; observation names the movement. Naturally include supported nodding, without treating intervals as counts or an empty list as proof of absence. Do not describe this metadata.
  Supplementary vocal intervals: id is an identifier; start/end are pair-relative seconds within the unmuted official listener audio gate; asr_text/official_text/observation provide candidate spoken words or vocalizations detected in that window. Listen carefully to the supplied audio at each interval and naturally integrate audible spoken responses (quoting intelligible words in English) or nonverbal vocalizations (quoting phonetic syllables such as "heh", "ha", "mhm", "uh-huh", "hmm", or "oh" when provided) in temporal order with concurrent visual actions. Do not mention detector metadata, interval IDs, or numeric timestamps.
  listener_audio_is_silent=true means the supplied waveform is entirely zero: do not claim audible speech or vocalizations. Otherwise describe only what you hear.
  ```

---

## 3. 24 条样本在 `172.29.189.119` 上的完整重跑验证

- **运行配置**：`172.29.189.119` 空闲显存卡（GPUs 6,7，`tensor_parallel_size=2`），24/24 条样本全部在统一 v4 Prompt 下重新推理完成（`finish_reason=stop`，非 ASCII 字符数 = 0），完成后已释放显存。
- **三列对照结论**：
  1. **第一版（仅点头线索，无 ASR 发声提示）**：14 条 J 大脸有声样本和原 50 条试点中的 `S0866 ("Exactly")`、`S0835 ("Ed")`、`S1485 (两次 "Okay")` 漏掉发声或误写。
  2. **上一版（点头 ＋ 原 ASR 发声提示）**：找回了英文发声，但 `#1 (S2043)` 将笑声中的两声 `"嘿"` 误写成打招呼的 `saying "Hey" twice`，`#21 (S0200)` 直接在英文段落中输出了汉字 `"嗯"`。
  3. **本轮新生成（v4：点头 ＋ 三级完备修正拟声 ASR 线索）**：
     - `#1 (S2043)`：`The listener begins by smiling slightly with their eyes crinkled, then laughs audibly, producing a series of short "heh" sounds followed by a longer laugh... During this time, they also nod their head once.`（既保留具体 `"heh"` 笑声音节指导，又补齐了点头）。
     - `#8 (S3358)`、`#16 (S0200)`、`#19 (S0157)`、`#21 (S0200)`：全部输出纯正英文拟声词（`"Mm-mm"`, `"mhm"`, `"Hmm."`）并与点头时序自然结合。
     - 其余 19 条英文样本（如 `#7 "Exactly"`, `#9 两次 "Okay."`, `#10 中段 "Uh huh" + 片尾 "Yeah"`, `#11 "Absolutely"`, `#14 "I understand"`, `#20 过滤串音保留 "Right."`, `#23 "All right." / "Okay"`）全部稳定输出高质量英文段落。
