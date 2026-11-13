# 整段事件人工核验页面优化（2026-11-12）

入口：[视觉四类](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/review_visual.html) · [声音三类](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/review_audio.html)

## 使用

1. 填核验者代号（如 rater01），完整播放每段素材。
2. 对每个类别点“出现／未出现／无法判断”，同类重复出现只记存在。不填表示未核验，不等于未出现。
3. 点选生成可编辑的核验说明；有疑问可补充原因。不需要填写事件时间，不评判语境适当性。
4. 点击下一段，或点击编号跳转。“定位未完成”可查漏；进度保存在同一浏览器、同一网址。
5. 完成后点击“导出本页 CSV”。视觉页导出 8×4=32 行；声音页另导出 8×3=24 行。未做完也可导出，未标注状态留空。文件下载至浏览器下载目录，交回文件即可，不需要手填表格。

画面侧倒可旋转观看，并勾选“原素材画面方向异常”。旋转只影响显示，角度和问题标记随 CSV evidence 导出；不更改输入视频、自动检测或生成结果。视觉预览无声，不能从微笑推断可听笑声。

## 实现与边界

- 构建脚本：`scripts/p2_event_presence_review_packet.py`；共享模板：`scripts/p2_event_presence_review_ui.html`。
- 已有匿名清单与空白表存在时，只重建页面，不调用检测协议验证或重置人工核验状态。首次生成缺少核验资料时仍调用原 review(work)。原媒体索引一致性检查保留。
- 对外页面只包含匿名 media_id、时长、播放路径和类别；不嵌入模型身份或检测答案。
- CSV 沿用 `media_id,modality,class,state,evidence,rater_id`，UTF-8 无 BOM，按标准 CSV 转义引号、逗号和换行。不会把空白自动转成 absent。
- localStorage 按媒体清单身份和模态隔离。同一浏览器同一页是一个核验工作区；多人独立核验应使用不同浏览器配置，不要接着他人的点选改代号。视觉／声音页分别填同一个核验者代号。
- 本次仅更新展示；没有完成真实人工标注，也没有重算自动事件指标。历史侧倒素材问题仍需独立迁移测试媒体及生成条件。

## 验证

- `BROWSER_CHECK.json`：隔离测试浏览器，实际加载 8 个视频和 8 个音频；点选、导航、刷新恢复、旋转和问题标记、CSV 下载、移动端无横向溢出、无 JS 异常。
- `CSV_CHECK.json`：原 `reporting.human_labels` 读取测试导出，分别识别 4／32／24 个已标注单元；未标注空白、中文、引号、逗号、换行和方向备注保留。
- `desktop.png`、`mobile.png`：未预填答案的页面截图。
- `TEST_ONLY_*.csv` 为自动化交互测试产生的虚拟标签，**不是人工标注，不能用于模型评测**。测试浏览器与用户浏览器隔离。
- `before/` 保存原脚本、页面及展示契约备份。

重建命令：

```bash
python scripts/p2_event_presence_review_packet.py \
  --work experiments/p2_event_presence_pilot_20261108/run \
  --output data_previews/listen_benchmark_review/event_presence_pilot_20261108
```
