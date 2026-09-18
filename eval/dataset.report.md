# 评测集构建报告

- 候选问题：**180** 条（来自 60 篇笔记）
- 通过校验：**180** 条
- 重复丢弃：**0** 条
- 不可回答问题：**18** 条（候选 18 条）
- 数据集总行数：**198** 行 → `eval\dataset.jsonl`
- 有译文孪生（记 grade 1）：**177** / 180 条
- 标记为词法泄漏（`leak: true`，df <= 2）：**31** / 180 条

## 丢弃原因分布

| 原因 | 条数 |
| --- | --- |

## 难度分布（通过校验的正样本）

| 难度 | 条数 |
| --- | --- |
| easy | 57 |
| medium | 97 |
| hard | 26 |

平均每篇笔记 3.00 条问题，覆盖 60 篇笔记（全语料 1947 篇，覆盖率 3.1%）。

## 词法泄漏分布（问题与目标笔记共享的最稀有 token 的 note 级 df）

问题是模型读着目标笔记生成的，会把该笔记的稀有标识符逐字抄进去。df 越低，
越是「不检索也能精确匹配」。`leak: true` 的条目**没有被删除**，只是被标记出来，
这样任何按分组出数的报告都能把它们单列（判据：df <= 2）。

| 共享的最稀有 token 的 df | 条数 |
| --- | --- |
| <=2 | 31 |
| 3-10 | 39 |
| 11-50 | 70 |
| >50 | 40 |

## 被丢弃的负样本

无。18 条候选的最佳笔记覆盖率全部低于 60%（最高 55%）。

## 负样本清单（已保留）

`覆盖率` = 全语料里覆盖该问题内容词最多的那一篇覆盖了几成，越低越可信。
`最稀有关键词` 仅供参考，**不再是判据**（它只要问题里有任意一个冷门片段就会放行）。

| 问题 | 覆盖率 | 覆盖最多的笔记 | 最稀有关键词 | df |
| --- | --- | --- | --- | --- |
| Kubernetes 的 Pod 驱逐策略怎么配置软硬阈值？ | 33% | `notes__implemented__feature__2026-06-18-compaction-capability-seam.zh__bf327288.md` | `pod` | 0 |
| Rust 的借用检查器如何处理生命周期标注中的协变？ | 45% | `notes__archived__architecture__2026-08-11-trajectory-conversation-context-assembly.zh__b48de83e.md` | `注中` | 0 |
| C++ 里 SFINAE 和 concepts 在重载决议中的差别是什么？ | 38% | `notes__implemented__feature__2026-09-05-sidebar-text-preview-and-file-tree.zh__46844dcc.md` | `sfinae` | 0 |
| Android 的 WorkManager 如何保证跨进程任务只执行一次？ | 53% | `notes__implemented__architecture__2026-07-23-client-plugin-loading-model.zh__c8777eef.md` | `workmanager` | 0 |
| SwiftUI 里 StateObject 和 ObservedObject 的销毁时机有何不同？ | 36% | `notes__archived__architecture__2026-07-12-scoped-layers-store.zh__ba383a2f.md` | `observedobject` | 0 |
| Blender 的几何节点如何用场驱动顶点位移？ | 33% | `notes__archived__bug-fix__2026-07-28-themed-scrollbars-and-reserved-gutter.zh__98f18d67.md` | `blender` | 0 |
| 用 CRISP-DM 做信贷违约预测，第一步该交付什么？ | 29% | `notes__archived__architecture__2026-08-11-repository-naming-contract-and-rename-ledger.zh__2fd27673.md` | `crisp-dm` | 0 |
| 住房贷款利息的个税专项附加扣除标准是多少？ | 16% | `notes__archived__architecture__2026-08-11-repository-naming-contract-and-rename-ledger.zh__2fd27673.md` | `个税` | 0 |
| AlphaFold2 的 Evoformer 为什么用三角乘法更新？ | 27% | `deepseek-harness__.agents__notes__README.zh__af5a1fbe.md` | `alphafold2` | 0 |
| Unity DOTS 里 Burst 编译器不支持哪些 C# 特性？ | 55% | `notes__implemented__feature__2026-07-06-sandbox.zh__c0e231f8.md` | `持哪` | 0 |
| FPGA 上实现 AXI4-Stream 时 TREADY 反压怎么处理？ | 30% | `notes__archived__architecture__2026-07-19-gui-layering-and-rpc-protocol.zh__34572700.md` | `axi4-stream` | 0 |
| InnoDB 的 next-key lock 如何避免幻读？ | 38% | `notes__archived__feature__2026-07-27-tmux-location-context.zh__70151974.md` | `innodb` | 0 |
| WebGL 的 transform feedback 如何实现 GPU 粒子状态回写？ | 36% | `notes__archived__architecture__2026-08-06-subagent-list-identity-projection.zh__92329a4b.md` | `webgl` | 0 |
| 跨时钟域握手时两级同步器为什么处理不了多比特信号？ | 26% | `notes__archived__bug-fix__2026-07-31-code-runtime-python-settlement-fixes.zh__cdc8dc32.md` | `么处` | 0 |
| PCR 扩增 3kb 片段时延伸时间按什么规则设置？ | 43% | `notes__archived__bug-fix__2026-07-31-code-runtime-python-settlement-fixes.zh__cdc8dc32.md` | `3kb` | 0 |
| 权责发生制与收付实现制对利润表的影响有何不同？ | 29% | `notes__archived__architecture__2026-07-15-lsp-capability-seam.zh__59de0e8f.md` | `付实` | 0 |
| Stockfish 的 NNUE 评估网络怎么做增量更新？ | 33% | `notes__archived__architecture__2026-07-19-gui-layering-and-rpc-protocol.zh__34572700.md` | `nnue` | 0 |
| 为什么说 FIFO 页面置换算法会出现 Belady 异常？ | 36% | `notes__archived__bug-fix__2026-07-31-code-runtime-python-settlement-fixes.zh__cdc8dc32.md` | `belady` | 0 |
