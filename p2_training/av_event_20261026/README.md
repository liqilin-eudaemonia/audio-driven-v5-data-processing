# P2 AV / AV＋Event 训练代码备份

2026-10-26。版本标签：`p2-av-event-no-feature-20261026`。本目录独立保存关闭 feature decoder 的训练版本，源码来自 `/private/liqi_workspace/project_code/audio-driven-v5`，同时保存同级 `audio-driven-v4/src` 的复用依赖。

- [实现、配置与运行说明](snapshot/audio-driven-v5/related/P2_AV与事件监督简化实现.md)
- [新训练配置](snapshot/audio-driven-v5/configs/p2_listen_av_event_v1.json)、[E0/E1 配对协议](snapshot/audio-driven-v5/configs/p2_listen_av_event_pair_v1.json)
- [训练入口](snapshot/audio-driven-v5/scripts/p2_listen.py)、[训练实现](snapshot/audio-driven-v5/src/p2_listen/training.py)、[模型构建](snapshot/audio-driven-v5/src/p2_listen/model.py)
- [原项目总索引快照](snapshot/audio-driven-v5/README.md)
- [源版本验证记录](snapshot/audio-driven-v5/experiments/p2_listen_av_event_validation_20261026/README.md)
- [逐文件 SHA256 清单](BACKUP_MANIFEST.json)、[备份副本验证](BACKUP_VALIDATION.json)

`training.feature_loss_enabled=false` 时不构建两路 feature decoder、不读取 teacher、不计算 feature loss，也不创建 features 优化器组。E0 用 AV loss 更新 Connector，事件头读取 detach(U)；E1 用 AV＋0.05×Event loss 更新同一份 U。两组只需 input、VAE、text、events 缓存。未指定开关的历史配置仍启用 feature 分支，首轮实验必须显式选择新配置。

源版本已通过 119 项 CPU 测试、两种模式的单/双进程对照和 50 段无 teacher 的真实缓存检查。本次另在本目录副本中重跑 CPU 测试，结果见 `validation/unit_tests.json`。简化结构尚未完成真实 LTX GPU S0，未启动正式效果训练；历史完整 feature 版本的 GPU 记录不能代替新结构验收。

## 恢复与复现

保留 snapshot 下两个同级目录的布局，即可使用相对路径 `reuse_root=../audio-driven-v4`。推荐复制到新的空目录，避免覆盖现有实验。下列命令从本备份目录执行：

```bash
# 目标必须是新目录；替换为实际恢复位置。
P2_RESTORE=/path/to/new_p2_restore
mkdir "$P2_RESTORE" && cp -a snapshot/. "$P2_RESTORE/"

# 使用已安装 torch、numpy 等依赖的 Python 环境；CPU 测试不加载预训练模型。
cd "$P2_RESTORE/audio-driven-v5"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python scripts/check_p2_listen_code.py --output /tmp/p2_av_event_restore_tests.json
```

也可直接在 `snapshot/audio-driven-v5` 运行上述测试。`validation/environment.json` 记录本次复测环境；本备份不捆绑 Python 环境、Omni/LTX/Gemma 等外部仓库或模型权重。真实训练还需要原项目说明中的环境、模型资源、媒体索引、标签与缓存。

完整性检查从本备份目录执行：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('BACKUP_MANIFEST.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
print('Verified', manifest['count'], 'snapshot files')
PY
```

仅备份源码、配置、文档、示例和小型验证记录；未上传媒体、权重、检查点、特征张量或凭据。原验证目录 runtime 中的缓存/数据软链接未复制，50 段真实缓存验证只能在原数据可访问时重跑。历史文档和总索引中的其他实验路径、旧版备份路径可能仅在源机器存在。已有 teacher 缓存和修改前的完整版本快照仍保留在源机器。

检查点恢复受严格源码、配置、数据和路径身份校验约束。恢复源码目录不等于允许旧检查点跨目录、跨模式直接续训；新实验使用独立 output，遵循实现文档的恢复规则。
