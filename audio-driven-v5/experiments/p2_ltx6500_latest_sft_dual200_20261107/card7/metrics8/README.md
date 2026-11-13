# GPU7评分并发4→8

2026-11-07，解码400条已完成，按用户要求继续优化评分吞吐。仅将本评测protocol中的metrics_workers_per_gpu与metrics_inference_workers_per_gpu由4改为8；正在运行的前4个分片保留，原控制器从protocol实时读取并发容量，直接领取剩余4个评分分片，无需重启。

node11物理GPU7仍为唯一允许的设备。原8个固定评分分片、metric_batch_size=64、CPU像素线程4、网络／人脸／表情设备、无损GT缓存、采样、特征与汇总公式均不变；最后仍基于全部200条特征计算FID/FVD。所有完成结果保留，不改变其他训练或恢复Goal／聊天巡检。

APPLIED.json为配置变更；BEFORE_PROTOCOL.json为原配置，BASELINE.json为生效前评分数量，BENCHMARK.json在有限实测后生成。评分进度看上级metrics_status.json和metric_shards逐条结果。最终完成以FINAL_AUDIT.json为准。

有限实测：并发增至8路后103.87秒新增64条视频评分，约0.616条/秒（包含新4路初始化）。当时累计94/400条视频评分，最新输出无指标错误；所有8个进程实际配置均仅node11 GPU7。预热后15次GPU采样平均52.2%，最高显存45.75GiB，平均功耗212.9W。当前评分仍有CPU像素／读取阶段，不能将解码阶段80.4%的采样或温度当作评分阶段利用率，也不宣称已达到最优。
