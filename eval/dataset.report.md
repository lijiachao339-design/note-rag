# 评测集构建报告

- 候选问题：**180** 条（来自 60 篇笔记）
- 通过校验：**180** 条
- 重复丢弃：**0** 条
- 不可回答问题：**18** 条（候选 18 条）
- 数据集总行数：**198** 行 → `eval\dataset.jsonl`

## 丢弃原因分布

| 原因 | 条数 |
| --- | --- |

## 难度分布（通过校验的正样本）

| 难度 | 条数 |
| --- | --- |
| easy | 57 |
| medium | 97 |
| hard | 26 |

平均每篇笔记 3.00 条问题，覆盖 60 篇笔记。

## 负样本清单（已保留）

| 问题 | 最稀有关键词 | 出现文档数 |
| --- | --- | --- |
| Kubernetes 的 Pod 驱逐策略怎么配置软硬阈值？ | `kubernetes` | 0 |
| Rust 的借用检查器如何处理生命周期标注中的协变？ | `注中` | 0 |
| C++ 里 SFINAE 和 concepts 在重载决议中的差别是什么？ | `sfinae` | 0 |
| Android 的 WorkManager 如何保证跨进程任务只执行一次？ | `workmanager` | 0 |
| SwiftUI 里 StateObject 和 ObservedObject 的销毁时机有何不同？ | `stateobject` | 0 |
| Blender 的几何节点如何用场驱动顶点位移？ | `顶点` | 0 |
| 用 CRISP-DM 做信贷违约预测，第一步该交付什么？ | `做信` | 0 |
| 住房贷款利息的个税专项附加扣除标准是多少？ | `个税` | 0 |
| AlphaFold2 的 Evoformer 为什么用三角乘法更新？ | `alphafold2` | 0 |
| Unity DOTS 里 Burst 编译器不支持哪些 C# 特性？ | `持哪` | 0 |
| FPGA 上实现 AXI4-Stream 时 TREADY 反压怎么处理？ | `反压` | 0 |
| InnoDB 的 next-key lock 如何避免幻读？ | `免幻` | 0 |
| WebGL 的 transform feedback 如何实现 GPU 粒子状态回写？ | `粒子` | 0 |
| 跨时钟域握手时两级同步器为什么处理不了多比特信号？ | `么处` | 0 |
| PCR 扩增 3kb 片段时延伸时间按什么规则设置？ | `扩增` | 0 |
| 权责发生制与收付实现制对利润表的影响有何不同？ | `对利` | 0 |
| Stockfish 的 NNUE 评估网络怎么做增量更新？ | `做增` | 0 |
| 为什么说 FIFO 页面置换算法会出现 Belady 异常？ | `belady` | 0 |
