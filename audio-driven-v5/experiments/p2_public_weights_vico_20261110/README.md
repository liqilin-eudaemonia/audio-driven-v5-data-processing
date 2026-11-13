# 公开权重优先：ViCo 推理与统一评分入口

2026-11-10起按最新分工执行：GLARE由用户推进；这里优先复用官方公开权重的推理结果，补齐共同评测，尽快提供可填表数字。不等待500h重训，未使用短跑诊断checkpoint充作正式基线。

入口绝对路径：`/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_public_weights_vico_20261110/README.md`。

[最终状态](status.json) · [最后执行阶段](ACTIVE.json) · [冻结输入和协议](protocol.json) · [输入验证](INPUT_AUDIT.json)。**2026-11-10 04:25 UTC已完成**：[统一EMOCA结果](RESULTS.md)、[全部指标CSV](metrics.csv)、[完成回执](COMPLETE.json)。两项baseline各43条，与我们两版共用16,238个有效评分帧；全部结果文件SHA核验通过。

**同时可引用：[已完成画质及L2L动作数字](EXISTING_RESULTS.md)、[画质CSV](existing_image_metrics.csv)。** 此处是对旧封存实测的汇集；本轮新增的DIM口径EMOCA分数见上方独立结果。

## 第一批与已有产出

| 方法 | 公开权重推理现状 | 本轮工作与填表边界 |
|---|---|---|
| DyStream | 43/43推理及统一EMOCA评分已完成 | 表情FD 31.715231、Pose FD 0.067969；listener自身音轨为零，speaker音频进入other分支 |
| Avatar Forcing，音频共同输入适配 | 43/43推理及统一EMOCA评分已完成 | 表情FD 37.633272、Pose FD 0.068397；speaker视觉条件为零，不能冒称原版完整输入 |
| Avatar Forcing，真实speaker视觉 | 43/43推理、129路SPECTRE提取及8项评分均已完成 | [实测与论文对照](../p2_avatarforcing_table4_recheck_20261110/RESULTS.md)：8项均未在论文两位小数精度下匹配；与本页EMOCA分支分开报告 |
| Learning2Listen | 原172条动作已完成；新增GPAvatar写实渲染接入与完整30帧视频测试通过，172条视频／评分后台接续中 | [接入入口、样例及实际状态](../p2_l2l_public_renderer_20261110/README.md)。标为“L2L＋GPAvatar公开渲染器”，与[原生DECA动作评分](../p2_compare_vico43_20261106/RESULTS.md)分开；官方原写实流程仍未发布，[原下载包复核](L2L_RENDERER_AUDIT.json)保留 |
| DIM／ViCo／ListenFormer | 既有资源核查尚未取得配套核心公开生成权重 | 继续检查官方资源；不把重建器、运动codec或本地短训权重标成官方预训练生成器 |

已有[三方法画质与DECA动作表](../p2_vico43_unified_20261107/RESULTS.md)已经有数字，可立即用于组内进度表。这里新增的FD是DIM分布Fréchet定义，不能与旧表的BFM逐帧L1同名FD混用。

## 共同读出设计

固定官方43个listener UUID、seed137、每条一次采样，不做按GT择优。复用已有DyStream／Avatar Forcing视频，验证逐条视频、生成回执、首帧与官方权重哈希。两者使用公开预训练权重，没有在本轮500h更新，没有自然语言caption条件。

统一以EMOCA_v2_lr_mse_20提取pose6＋exp50，直接调用已验证的原有编码与官方FAN裁剪函数。资产、许可确认和GT／speaker／我们两版的172份特征复用旧封存产出，不改写原结果。新增仅86份baseline视频特征。

计分遵循listener 1:T与真实speaker 0:T-1；GT、speaker、两项baseline及我们两版共用全部四模型的有限帧交集。新增检测失败可能使可评分帧减少，因此我们两版也在本次共同帧集重算，旧成绩保留。逐条缺帧、实际覆盖和原始native Var都保存在JSON。

FD、P-FD、MSE、SID、rPCC沿用固定的DIM公开实现；Var采用此前用户指定的逐系数跨帧方差再平均，ddof=0。CSV／Markdown／JSON均已输出，可直接填写本地实测行。公开权重的训练数据与原生条件不同于我们模型，本地表不声称受控500h重训比较或原论文表格精确复现。

## 执行、资源与复查

仅node11物理GPU7；编码worker最多占整卡5.5%显存，启动要求空闲至少32GiB。现有Avatar Forcing流程和其他任务保持运行；不恢复暂停的我们模型DIM best10队列。

原执行命令，仅供核对或失败后恢复：

后台控制器已完成全部阶段，smoke／extract／score实际退出码均为0。86条视频／权重哈希、86份实际音频／静音／首帧条件条目及172份旧特征验证通过。编码器全部644状态严格加载；原生编码和独立拆分最大差0，OpenCV／FFmpeg首帧像素差0，batch1／16最大差约7.15e-7。86项新增提取全部完成且无缺脸帧；共同帧交集只排除了原我们模型已有的7帧。344个逐片段MSE经独立float64重算，最大差2.06e-7；发布指标函数核对和GT自对照通过。见[最终验收](FINAL_AUDIT.json)。无需重新执行生成或评分。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python scripts/p2_public_weights_vico.py run
```

控制器持独占锁，完成标记存在的阶段会跳过；失败可查看`smoke.log`、`extract.log`、`score.log`与对应`*_exit.json`。恢复前核对源码和协议，不能在冻结后改公式续算同一目录。

完成后的只读完整性检查：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('experiments/p2_public_weights_vico_20261110/COMPLETE.json')
for name, expected in json.loads(p.read_text())['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('sealed result files verified')
PY
```

官方来源：[DyStream](https://github.com/RobinWitch/DyStream)、[Avatar Forcing](https://github.com/TaekyungKi/AvatarForcing)、[Learning2Listen](https://github.com/evonneng/learning2listen)、[DIM](https://github.com/Boese0601/Dyadic-Interaction-Modeling)、[ListenFormer](https://github.com/liushenme/ListenFormer)。可用性以本地成功取得并严格核验的生成权重为准。
