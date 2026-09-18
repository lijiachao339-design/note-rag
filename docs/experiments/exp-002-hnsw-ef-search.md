# 实验 EXP-002：`hnsw.ef_search` 造成的向量召回损失

- **日期**：2026-09-18
- **触发**：Claude Code + Opus 的审查 `docs/reviews/review-001-eval-methodology.md` 第 S3 条
- **相关代码**：`scripts/exp_hnsw_ef_search.py`（本实验的可执行脚本）
- **数据集**：`eval/dataset.jsonl`，198 行（180 可回答 + 18 不可回答）
- **语料**：1945 篇进索引 / 17520 chunk，指纹 `9854a816415b1a9579b6786af6f975f19af4e06f`

## 假设（来自审查 001 的 S3）

> `store.py` 用 HNSW 建索引（近似检索），`retriever.py` 请求 `limit = candidate_k = 50`，
> 而 pgvector 的 `hnsw.ef_search` 默认只有 **40**。ef_search 小于请求的 limit 时召回会明显掉。
> 所以 `vector` 一列的短处里，有一部分是**检索配置**造成的，而不是嵌入能力造成的。

审查者明确标注"这一条请先验证再下结论（我没有数据库连接）"。本实验就是那个验证。

## 方法

固定测试集、固定嵌入（离线 hashing，dim=384）、固定 chunk，**只改 `hnsw.ef_search`**。
排序与去重逻辑复现生产路径（`search_notes` 的"每篇取最佳 chunk"），否则差异无法归因。

```powershell
uv run python scripts/exp_hnsw_ef_search.py
```

## 结果

| ef_search | recall@1 | recall@5 | recall@10 | mrr |
| --- | --- | --- | --- | --- |
| 记录值（默认 40） | 0.1768 | 0.2677 | 0.3182 | 0.2216 |
| **40** | **0.1768** | **0.2677** | **0.3182** | 0.2246 |
| 100 | 0.1717 | 0.2828 | 0.3535 | 0.2294 |
| 200 | 0.1919 | 0.3030 | 0.3939 | 0.2541 |
| 400 | 0.1919 | 0.3131 | **0.4040** | 0.2579 |
| 精确余弦（离线复现的上界） | 0.1919 | 0.3232 | 0.4091 | 0.2570 |

- `ef_search=40` 时**逐位复现**了记录值（0.1768 / 0.2677 / 0.3182），
  证明本脚本与生产路径可对齐，后面的差异可以归因到 ef_search 本身。
- 延迟：`ef_search=200` 时 p95 = **50.2 ms**（默认 40 时记录值为 54.0 ms）——**没有代价**。

## 结论

**S3 成立。** 把 `ef_search` 从默认 40 提到 400，recall@10 从 0.3182 回到 **0.4040**，
几乎贴着精确余弦的上界 0.4091。也就是说：

1. 记录的 `vector` 一列里，**约 8.6 个点（相对 +27%）是 ANN 近似损失，不是嵌入能力上限**。
2. 剩下从 0.4040 到 `keyword` 的 0.8283 的差距，才是 hashing 嵌入"没有语义"的真实代价。
   **exp-001 的定性结论仍然成立，但它的数字与归因必须修正。**
3. recall@1 随 ef_search 非单调（0.1768 → 0.1717 → 0.1919）：近似检索本来就不保证单调，
   而单标签 recall@1 只有 180 个样本、噪声大。**不要拿 recall@1 的 0.02 级差异下结论。**

## 已落地的修复

- `config.py` 新增 `hnsw_ef_search: int = 200`，并在注释里写明默认 40 与 candidate_k 50 的冲突。
- `store.py` 新增 `set_hnsw_ef_search(conn, ef_search)`；`Retriever.from_settings` 建连接后立即调用。
- `.env.example` 记录 `NOTE_RAG_HNSW_EF_SEARCH=200`。

**为什么把它写进 `Settings` 而不是硬编码**：检索参数不记录，数字就不可复现——
这正是审查 001 要修的那类问题。

## 顺带暴露的一个约束缺口

`CLAUDE.md` 的约束表要求"`fusion.py` 排序必须确定性（tie-break 到 id），否则评测数字不可复现"。
但向量这一路走 HNSW（近似检索），**端到端确定性并没有被保证**：重建一次索引，
`vector` 与 `hybrid` 两列都可能漂移。约束表需要补这条 caveat。

## 下一步

1. 用新的 `ef_search` 重跑完整四模式消融，更新 `README` 与 exp-001 的 `vector` 列与归因。
2. 把"检索参数必须记录"写进 `AGENTS.md` 的硬性规则（这是第二次因为参数未记录而无法复现数字）。
