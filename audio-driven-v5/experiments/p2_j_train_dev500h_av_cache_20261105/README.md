# 当前train＋dev 500h AV VAE增量缓存（2026-11-05）

用户已确认沿用旧约定，并授权空卡立即接手。范围保持284,218段／500.00125913h，train 281,060＋dev 3,158，test不加入。状态以chain_status.json、CANARY.json和COMPLETE.json为准。

- 276,780段直接引用已核验的旧AV payload，约398.25GB，不复制、不修改旧文件。
- 896段保留旧listener音频latent，只重新编码已转正的视频。
- 6,542个新成员编码视频和listener音频。
- 合计只需补齐7,438段视频、6,542段音频，不重新计算整批500h。

## 数学定义和来源

512×512、25fps，从实际源TAR视频按最近真实PTS取帧、使用最终裁框；已转正成员读取恢复后的源视频。尾部重复最后一帧补成8n+1，视频VAE BF16，首帧单独编码后写回第一个latent。listener音轨严格沿用原官方word/sentence/VAD并集、100ms边缘和10ms淡变配方，保留完整时间；原生音频VAE FP32计算、BF16保存，原生补零、端点token处理和因果支持mask不变。

新成员直接调用p2_listen.runtime.encode_vae。视频单独重算仅复制该函数的视频运算序列，并保留旧音频latent及mask；验收时与包含真实音频编码的完整原生路径逐值对照。完整视频首帧必须与已完成Qwen输入的RGB SHA一致；listener净化波形必须与最新caption的原音频证据SHA一致。不会用试听MP4作为训练输入。

旧VAE引用绑定此前全量CPU核验结果：全部277,676个重合旧payload已重算完整SHA256，来源、最终裁框、音频配方及缓存契约通过。原审核的精确PTS特例继续随映射保留。

## 调度和验收

四机node8/node17/node12/node4，逐卡检查GPU UUID、计算进程及空闲显存；空卡接手，已占用卡等待，未停止其他作业。原生长片段验收后按实测显存选择每卡1或2个常驻进程，实际见execution.json。每worker 4个独立CPU媒体producer、预取4条、2线程原子异步写盘；共享flock动态队列每块16条，长片段优先领取，减少尾部等待。单块失败隔离并保留，其他块继续，失败不算完成。

先做8条真实GPU验收，覆盖长短、dev、转正与音频复用；原生完整AV对照必须逐值一致。写入时SHA及逐张量回读校验。提取结束后，32个CPU进程核对全部284,218成员的新旧混合映射、source split、来源、完整任务及缓存回执；通过才写COMPLETE.json。

## 读取

```python
import sys
sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/scripts")
from p2_j_current_av_cache import Reader
r = Reader("/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train_dev500h_av_cache_20261105")
av = r.load(sample_id)  # 自动选择核验旧payload或新payload
pack = r.training_sample(sample_id, variant="guided")  # 联合新Qwen H、Gemma和AV
ids = r.ids()  # 完整284,218个选集成员，保留原split，默认不再按train过滤
```

Reader返回原生condition/layout/text/av/support等字段。这里只提供缓存及联合读取，没有启动训练，也没有新增Whisper／Feature teacher／Event标签。

入口scripts/p2_j_current_av_cache.py；protocol.json冻结当前H、Gemma、复用计划、模型与代码。读取过程核对旧payload原始identity及SHA，新caption与源split保留在当前来源记录。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_j_current_av_cache.py status
# 如controller退出，先检查failed、canary和节点日志，再恢复同一协议：
$PY scripts/p2_j_current_av_cache.py chain
```

**全量混合缓存已完成并通过验收。** 当前284,218段全部覆盖：引用276,780份旧AV，新增7,438份组合payload（约10.83GB），其中896份复用旧音频。64个提取worker均正常退出；全量审计及train/dev、整份复用/部分复用/新增三类的Qwen＋Gemma＋AV联合读取、原生训练输入及loss mask准备通过。见[完成回执](COMPLETE.json)、[联合回读](JOINT_READBACK.json)。未启动训练。
