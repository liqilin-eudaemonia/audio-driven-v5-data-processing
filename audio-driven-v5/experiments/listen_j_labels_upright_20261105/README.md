# 新全量标签统一入口：原画转正＋重复重打标（2026-11-05）

本目录已发布完成，可作为后续行为统计和训练／测试选集的统一标签入口。主文件 [labels.jsonl](labels.jsonl)，**1,107,313条**；[UPRIGHT_AUDIT.json](UPRIGHT_AUDIT.json)独立验收通过，[EXPORT_COMPLETE.json](EXPORT_COMPLETE.json)绑定摘要与验收SHA。

自动通过及待复核均保留，保留复核标记130,347条。原画横置747个来源／1,699段，1,690段已转正并重新检测、打标，9段因转正后视觉不可用而剔除。方向正常的宽屏、正常歪头均保留。

全部36条原重复标记均已重打标：36条保留、0条剔除。新标签正常结束且无病态长循环；普通复核标记继续保留。最终剔除共9条，见[excluded.jsonl](excluded.jsonl)；原有1条无新标签例外保持单列，见[exceptions.jsonl](exceptions.jsonl)。[coverage.jsonl](coverage.jsonl)完整归账1,107,323个目标ID。

| split | 可用标签 |
|---|---:|
| train | 1,086,207 |
| dev | 8,822 |
| test | 12,284 |

[replacements.jsonl](replacements.jsonl)记录每条旧／新caption及SHA；[EXPORT_SUMMARY.json](EXPORT_SUMMARY.json)给出全量文件大小、SHA和版本。未受影响的原标签逐行字节保留，改写原行0；转正／重复修复各行有自己的caption_identity_version，应逐行读取，不能假定全池仍是同一个旧cache身份。`training_admitted`仍为false，表示尚未执行新的训练选集。

转正行的source已指向独立原生裁框转正素材的TAR/NUT字节范围，时间轴、sample ID、pair、split／group、participant和净化音轨保持原样。原生分辨率未缩放；档案SHA、每段首尾PTS和代表帧像素对照均核验，新的25fps检测输入也核验。转正行的quality、face_p10／p50、crop_provenance和detector SHA来自新素材；orientation_restoration保留完整旧来源／质量及修复依据。原素材、旧caption、旧缓存没有覆盖。

node8新地址172.29.104.237的GPU0–7以8个独立模型副本运行，单卡最多4条并发、CPU预取与同次解码复用。启动、GPU实际利用率及完整打标结果见[执行记录](../listen_j_upright_regen_20261105/README.md)。

AV500h当前范围已按用户确认改为train＋dev：最终284,218段／500.00125913h，源train 281,060段、源dev 3,158段，保留原split，test未加入。当前成员、更新后的类别分布和独立验收见[新版统一选集入口](../listen_j_upright_av500h_train_dev_v3_20261105/README.md)，机器读取入口见[CURRENT_AV500H.json](CURRENT_AV500H.json)。全部新caption的冻结v3统计复用原结果，无需重打标或再跑LLM；原train-only清单留档，标签清单及原SHA不变。训练张量缓存尚未重建，规划SFT分支另行保留其范围。
