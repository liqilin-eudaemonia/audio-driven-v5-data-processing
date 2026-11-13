# 50 小时 train 子集随机复核

- 服务端页面：[http://127.0.0.1:8876/listen_benchmark_review/train50h_random/](http://127.0.0.1:8876/listen_benchmark_review/train50h_random/)
- 已通过当前在线 Cursor 的 remote CLI 打开并触发当前窗口的端口转发。使用自动打开的新标签页，或 Cursor「端口」面板中 **8876** 对应的本地地址，并加 `/listen_benchmark_review/train50h_random/`。此前的本机 63034 地址已失效，不再固定引用。[恢复验证记录](cursor_access_20261026.json)。
- [原始抽样数据](review.json)、[构建检查](build_check.json)、[浏览器检查](browser_check.json)、[桌面截图](review_desktop.png)。

固定随机种子 `20261026`，从已冻结的 30,415 段 train 子集中均匀无放回抽 50 段；再按七类分别独立均匀无放回抽 5 段。合并 sample_id 去重，本次共 85 段，没有重复抽中。默认显示整体随机 50 段；分类补充入口用于检查稀有类，不能代表总体类别比例。

画面是已经处理好的 512×512 listener 训练裁剪视频，预览复制 H.264 画面流，仅将 listener WAV 编码为 AAC 合入同一个预览播放器。可另行下载原始 listener WAV。没有加入混音或 speaker 音轨，没有修改训练媒体。

展示原始人脸短边 p10/p50、预裁框大小、检测有效比例、原始整合事件描述与时间区间。点事件或时间线条带会播放其前后各 0.2 秒范围，支持循环、速度切换、类别筛选和标签 JSON 下载。每条的源 JSON 全文保留。

85 段已核对来源媒体哈希和标签计数，并在浏览器中逐段验证 512×512 解码和拖动定位；验证 listener 音轨解码、事件跳转、206 范围请求、下载、空结果搜索、中文字体与手机布局。复用现有 8876 常驻服务，未启动新的临时服务或改变训练数据集。

生成入口：`scripts/build_train_subset_review.py`；模板：`scripts/train_subset_review.html`。复现使用新的输出目录：

```bash
python scripts/build_train_subset_review.py \
  --dataset experiments/p2_train50h_backchannel_face_20261026 \
  --output data_previews/listen_benchmark_review/train50h_random_rebuild \
  --seed 20261026 --overall 50 --per-class 5 --workers 8 \
  --font /tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

`--font` 可指定另一份 NotoSansCJK-Regular.ttc；不传时使用浏览器系统字体。
