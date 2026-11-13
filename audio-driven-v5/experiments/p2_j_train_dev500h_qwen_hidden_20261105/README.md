# 当前train＋dev 500h原始Qwen hidden states（2026-11-05）

用户已确认使用原始、未做规划SFT的Qwen2.5-Omni-3B。输入为当前独立验收后的284,218段／500.00125913h，其中train 281,060段、dev 3,158段；原split保留，test不加入。**已全部完成并通过验收。** 两种H各284,218条，共568,436份payload，合计约4.277TB（十进制）；失败块0，96个worker均正常退出。32进程最终审计70.42秒通过，见[完成回执](COMPLETE.json)、[完成摘要](completion_summary.json)、[32卡实际产出验收](LIVE_ACCEPTANCE.json)。正式提取耗时约17.43分钟，包含模型加载、CPU预取、写盘及尾部收尾。

本轮只提取Qwen H：直接输入input，以及原生assistant位置追加完整最新caption指导文本的input_plan。模型冻结，第8／18／28／35层，每层2048维。沿用原显式system、原始多模态前缀、Gemma固定说明＋Listener response description＋caption全文、原生im_end，拒绝截断。不重新生成标签，不提取Gemma、AV latent、独立teacher，不启动训练。

## 数据和代码

- 当前清单：[selected.jsonl](../listen_j_upright_av500h_train_dev_v3_20261105/selection/selected.jsonl)。SHA256：d1ad15af128fb31833e1fd18ebbaae374839c21a45cce382288171d8ed7eb971。
- 独立运行入口：../../scripts/p2_j_current_qwen_hidden.py。protocol.json冻结本轮清单、逐行SHA、模型依赖、prompt和代码。
- 复用p3_listen.features.prepare_prefix/prefix、p2_listen.features.extract_input、p2_plan_cache.encoding.extract_planned_input，以及原生音频采样点修复、视频首帧解码和原子缓存读写。
- 新增内容仅为当前清单与已转正来源的适配、H专用共享队列及调度；旧数据和旧提取代码不修改。普通caption复核标记按用户接受尺度保留。
- 参考图为最终512×512裁框首帧，音频为配对speaker同一绝对区间；listener音频不进入Qwen。只对source完全相同的旧首帧进行像素SHA核验后复用，其余从最终来源重新解码；已转正来源读取恢复后的画面。

## 加速和验收

参考related/P2_J新数据缓存与四机32卡提取_20261101.md和related/P2_规划SFT后500h隐藏特征接续提取_20261102.md。四机node8/node17/node12/node4各8卡，node8地址172.29.104.237。每卡3个常驻模型进程；前两个各6个CPU producer，第三个2个，每进程预取8条、2个异步writer、Torch主线程2、producer线程1。每块128条，共享flock动态领取，完成顺序消费；单块失败隔离后其他块继续，失败不记成功。只管理本轮子进程。

先对32条分层真实样本进行CPU与GPU验收，覆盖train/dev、主集/补充、转正来源、长短片段和长caption；所有processor输入对照原始解码逐值一致，前8条及转正样本对照独立原生H提取逐值一致。每个payload写入时校验SHA并逐张量回读，最后32个CPU进程核对全体成员、split、来源、caption、回执、身份与文件大小，通过才写COMPLETE.json。

缓存位于cache/input/<sample_id>/payload.pt和cache/input_plan/<sample_id>/payload.pt，使用脚本Reader(work).load(sample_id, part='input'或'input_plan')读取。Reader返回原生condition、layout等字典；本轮没有完整AV训练包。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_j_current_qwen_hidden.py status
# controller终止后，先检查failed、节点日志和chain_status，再恢复：
$PY scripts/p2_j_current_qwen_hidden.py chain --per-gpu 3
```

旧500h的H不因样本ID相同就直接复用：本轮两个H均按新协议计算；指导H严格绑定最新caption。只复用经检查一致的首帧，减少无关整段解码和重复IO。

## 完成后的读取示例

```python
import sys
sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/scripts")
from p2_j_current_qwen_hidden import Reader
r = Reader("/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train_dev500h_qwen_hidden_20261105")
direct = r.load(sample_id, part="input", device="cpu")
guided = r.load(sample_id, part="input_plan", device="cpu")
# direct["condition"] / guided["condition"] 保留原生ConditionInputs字段
# hidden: [1, 4, token_length, 2048]；原split由protocol和逐样本身份保留
```

首轮独立对照曾因内存WAV游标未复位而失败；已修复对照读取并重跑全部32条通过。历史日志及原协议保留在revisions/canary_audio_cursor/；生产编码器和原始音频未改动。
