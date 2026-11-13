# 当前500h speaker Whisper增量缓存（2026-11-05）

**已全部完成。** 当前284,218段，train 281,060＋dev 3,158，原split保留、test不加入。277,676份旧Whisper直接引用，新提取6,542份；全量视图以[MIXED_COMPLETE.json](MIXED_COMPLETE.json)为准。[COMPLETE.json](COMPLETE.json)只记录6,542份新payload的原生验收。

编码直接调用未修改的p2_whisper_av.cache.extract／FrozenWhisper：FP32、关闭TF32、最后一层1280维、50Hz、30秒窗与2秒上下文、原不重复核心时间网格。只读配对speaker，不使用listener门控、不去前导静音、不调用ASR decoder。每卡batch16、4读取线程、双缓冲预取和单线程异步写盘。32rank各完成真实Encoder的短音频＋27秒跨窗batch/逐条验证，源文件身份、音频时间、mask和每份新payload SHA全部核验。四机32卡新编码6,542份完成，无失败或缺失。

复用依据为独立aux审计的speaker原轨／归一化身份／片段时间，以及旧模型、数值配置、完整验收和逐样本缓存身份。旧文件不复制、不改写，读取时仍验证SHA。画面转正不会触发speaker特征重算。

```python
import sys
sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/scripts")
from p2_j_current_whisper_cache import Reader
r = Reader("/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train_dev500h_whisper_cache_20261105")
x = r.load(sample_id)  # 原生SpeakerFeatures(hidden, valid, bounds)
```

每个进程独立构造Reader。新payload在cache/<sample_id>，旧payload通过独立旧Reader核验后读取。新适配器只在本进程为原生提取器提供当前来源接口，没有改写原模块或数值函数。代码入口scripts/p2_j_current_whisper_cache.py，协议绑定adapter和原代码／模型SHA。未启动训练。
