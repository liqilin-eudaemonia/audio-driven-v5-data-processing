# 四组Event × 双seed · 四机28卡评测

范围：Event 0.05／0.01／0.20／1.00，Feature两路各0.025；统一500步检查点、同64条、生成seed137／238，共512条。原seed137已有latent继续复用，SFT2752／Qwen H／Gemma／speaker Whisper缓存共享；Euler30／视频CFG3／音频CFG7及512分辨率保持。

资源：node12／node8／node4 GPU0–7＋node11 GPU0–3，共28卡。node11 GPU4–7不参与；旧GPU7评分队列已停并保留产物，其余服务未停止。

已完整保存并暂停Feature弱组step484，模型／Adam／28份RNG及SHA通过，28卡释放通过。见TRAINING_PAUSED_ACCEPTED.json。训练STOP保留；评测结束不自动续训。

沿用evaluate28.py共享领取、逐卡释放、最多4路VAE解码和2路评分、就绪分片评分、GT无损RGB／特征缓存、CPU预取2／像素4线程、GPU人脸／表情及batch64，最终原生全量bind逐行核验并合并全部特征。

事件评分沿用原生prepare／candidates／Qwen3-Omni infer／score及参考标签冻结逻辑，只将独立匿名media分配给28卡：每个media恰好一个worker，实际host／GPU UUID另行记录并验证，采样／提示／类别定义保持。空闲卡采用80%显存预算、关闭CPU权重offload；其完整GPU验收待实际阶段运行。模型身份／库版本必须在28卡一致，原生检测函数的generate语法树一致性检查通过。

测试：5项新增检查及8份真实评测配置检查通过；原40项双seed／评分回归保持。CPU检查不替代实际GPU验收。配置和源码指纹见PLAN.json／TESTS_ACCEPTED.json。

入口：driver.py run（已启动，勿重复启动）。进度STATUS.json；进程CONTROLLER.json；错误FAILURE.json；各任务evaluations/<组>_500_seed<seed>/eval28.log及status.json。停止：本目录STOP；只停止本评测，训练继续保持暂停。

全部结束写COMPARISON.json／COMPLETE.json：每seed指标和均值、来源组配对bootstrap、两个seed是否方向一致；FID／FVD各在64条上计算后平均，不合并成128条独立样本。不自动晋级、不声称全局最优。
