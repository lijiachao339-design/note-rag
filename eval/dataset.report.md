# 评测集构建报告

- 候选问题：**270** 条（来自 90 篇笔记）
- 通过校验：**269** 条
- 重复丢弃：**0** 条
- 不可回答问题：**27** 条（候选 30 条）
- 数据集总行数：**296** 行 → `eval\dataset.jsonl`
- 有译文孪生（记 grade 1）：**266** / 269 条
- 标记为词法泄漏（`leak: true`，df <= 2）：**66** / 269 条
- 语言：**135** 条英文 + **134** 条中文

## 丢弃原因分布

| 原因 | 条数 |
| --- | --- |
| answer_span 在源笔记中找不到（疑似幻觉） | 1 |

## 难度分布（通过校验的正样本）

| 难度 | 条数 |
| --- | --- |
| easy | 78 |
| medium | 147 |
| hard | 44 |

平均每篇笔记 2.99 条问题，覆盖 90 篇笔记（全语料 1947 篇，覆盖率 4.6%）。

## 词法泄漏分布（问题与目标笔记共享的最稀有 token 的 note 级 df）

问题是模型读着目标笔记生成的，会把该笔记的稀有标识符逐字抄进去。df 越低，
越是「不检索也能精确匹配」。`leak: true` 的条目**没有被删除**，只是被标记出来，
这样任何按分组出数的报告都能把它们单列（判据：df <= 2）。

| 共享的最稀有 token 的 df | 条数 |
| --- | --- |
| <=2 | 66 |
| 3-10 | 72 |
| 11-50 | 83 |
| >50 | 48 |

## 被丢弃的负样本（有笔记覆盖了它的内容词，可能其实有答案）

- How do I stop Android WorkManager from running the same job twice across processes? —— `notes__implemented__architecture__2026-06-20-generic-long-running-tool-runtime__f61a505c.md` 覆盖了它 78% 的内容词
- What extension time should I use when amplifying a 3 kb fragment by PCR? —— `notes__archived__feature__2026-07-07-plan-mode__d0c358a7.md` 覆盖了它 71% 的内容词
- Why can a two-stage synchroniser not carry a multi-bit signal across clock domains? —— `notes__archived__feature__2026-07-22-durable-subagent-catalog-and-list-agents__4b871cbe.md` 覆盖了它 62% 的内容词

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
| How do I configure a Kubernetes PodDisruptionBudget for a StatefulSet? | 25% | `notes__archived__architecture__2026-07-15-lsp-capability-seam__a9ef44c9.md` | `poddisruptionbudget` | 0 |
| What does the Rust borrow checker do with covariant lifetime parameters? | 56% | `notes__implemented__architecture__2026-07-08-agent-scope-contexts__698c5b07.md` | `covariant` | 1 |
| How does C++ SFINAE differ from concepts in overload resolution? | 57% | `notes__archived__architecture__2026-06-17-filesystem-capability-seam__f569c13b.md` | `sfinae` | 0 |
| When is a SwiftUI StateObject deallocated compared with an ObservedObject? | 43% | `notes__archived__bug-fix__2026-08-06-plan-narrow-viewport-regression__d23029d9.md` | `deallocated` | 0 |
| How does InnoDB next-key locking prevent phantom reads? | 43% | `notes__archived__architecture__2026-07-28-consolidated-tui-presentation__2708cd73.md` | `innodb` | 0 |
| How do I implement transform feedback for a GPU particle system in WebGL? | 50% | `notes__implemented__feature__2026-07-07-mcp-client-plugin__9cf4946f.md` | `particle` | 0 |
| How does accrual accounting differ from cash accounting on the income statement? | 50% | `notes__archived__bug-fix__2026-07-30-source-checkout-workdir-distinction__26aa1f14.md` | `accrual` | 0 |
| How does Stockfish update its NNUE evaluation network incrementally? | 57% | `notes__archived__architecture__2026-08-11-repository-naming-contract-and-rename-ledger__880869b8.md` | `nnue` | 0 |
| Why does the FIFO page replacement algorithm suffer from Belady's anomaly? | 56% | `notes__archived__architecture__2026-08-11-repository-naming-contract-and-rename-ledger__880869b8.md` | `belady's` | 0 |
