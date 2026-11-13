# 前序串行验收导致接续空等

teacher 全部285,638条已提取，失败0，但旧watch逐条串行审计，尚未写COMPLETE，因此32卡释放后新H接续器仍在等待。

修复：使用独立 `scripts/p2_j_parallel_audit.py --processes 32` 并行执行同样的成员、来源、contract、identity、payload SHA回执和文件大小检查。不改teacher数学定义或源代码，不略过验收。32进程全量验收耗时25.14秒，285,638条全部通过，写入原协议的COMPLETE。随后停止已冗余的本轮串行watch PID1957125；接续器自动进入资源检查及GPU canary。

先尝试16线程受Python解析开销影响，仅约750条/秒，已结束该次只读审计，最终采用32进程。正例以及来源、身份、hash回执、文件大小篡改共5项检查通过。验收结果及执行源代码随此说明保存。

GPU canary 随后发现 PEFT disable_adapter 上下文退出会重新启用 LoRA 参数梯度；下一条被 frozen 检查拒绝。已在对照的 finally 中显式恢复 requires_grad=False / eval，增加连续两轮对照回归。旧协议和失败结果完整移入 revisions/canary_freeze_restore_v1。修正后新协议的32条CPU检查、13项回归、32条GPU canary全部通过；其中8条原生H与优化H逐值一致，禁用LoRA复现旧H，32条新H均实际改变且联合读取验证通过。
