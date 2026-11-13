# 当前train＋dev 500h Gemma文本缓存（2026-11-05）

范围为当前284,218段／500.00125913h：train 281,060、dev 3,158，原split保留，test不加入。仅提取Gemma文本条件，不启动AV VAE、Feature teacher、Whisper、打标或训练。状态以chain_status.json、CANARY.json和COMPLETE.json为准。

复用已完成的Qwen H协议、逐行SHA索引和同一份caption_prompt，逐条核对Qwen指导H的来源／caption绑定。Gemma原生路径为NativeTextConditioner.encode_batch→原LTX文本连接器；直接文本编码，不使用聊天模板。共享固定正向为GEMMA，固定负向为空字符串；逐样本指导文本为GEMMA＋Listener response description＋最新caption全文，与Qwen input_plan的文本完全一致。

## 执行和验收

四机node8（172.29.104.237）、node17、node12、node4各8卡，每卡1个常驻Gemma，batch32，原生1024-token路径；超长使用原生分档扩展至8192且拒绝截断。共享flock动态任务队列每块256条，2线程后台写盘，最多64条在途写入，expandable_segments。参考related/P2_J新数据缓存与四机32卡提取_20261101.md的Gemma batch32/64数值与吞吐试验，选择已证明更快的batch32。

先对沿用Qwen验收的32条真实分层样本验证完整token、原生相同batch的逐值一致性、Qwen/Gemma全文及SHA一致性、固定正负文本和缓存回读。各payload原子发布、校验SHA并逐张量回读；全量结束由32进程核对成员、原split、所有prompt绑定、回执、身份和文件大小，通过后才写COMPLETE.json。失败块保留并隔离，不记作成功。

旧缓存和源代码不改写。本轮入口scripts/p2_j_current_gemma_cache.py，protocol.json绑定Gemma/LTX模型、tokenizer、代码与Qwen来源协议，execution.json记录实际执行配置。CPU只有文本读取与tokenization，不解码视频或读取音轨。

## 读取

```python
import sys
sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/scripts")
from p2_j_current_gemma_cache import Reader
r = Reader("/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train_dev500h_gemma_cache_20261105")
fixed = r.fixed()  # positive / negative
text_guided = r.load(sample_id)  # positive, prompt_sha256, tokens等
```

固定文本保存cache/text/shared；逐样本指导文本保存cache/text_guided/<sample_id>/payload.pt和complete.json。视频文本维4096，音频文本维2048；沿用LTX原生输出mask，不能把输出mask当成输入token数量。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_j_current_gemma_cache.py status
# 如controller退出，先看failed及日志；恢复同一协议：
$PY scripts/p2_j_current_gemma_cache.py chain
```

AV VAE尚未启动，等待用户讨论具体规格后再处理。

**全量已完成并通过验收：** 284,218条guided文本＋共享固定正负文本，总guided payload约3.578TB；32条原生同batch逐值对照及全文一致性通过，32进程最终审计63.52秒通过。见[完成回执](COMPLETE.json)。所有提取worker已退出；后续AV增量缓存单独运行。
