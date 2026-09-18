# 设计 006：BM25 倒排索引

> 对应任务书：`docs/reviews/prompt-006-bm25-inverted-index.md`
> 作者：Claude Code + Opus 5。**本文写作阶段未修改 `src/`**——下面所有数字来自
> 在 scratchpad 里搭的一次性原型（读现有 `BM25` 的内部状态，不落进仓库）。

---

## 0. 先修正一个前提：瓶颈不是"全量扫描"

任务书（和我自己在 impl-004 里写的）说瓶颈是"每次查询全量扫描 17524 个文档"。
**实测下来这只对了一小半。** 拆解一次 `search()`（296 条真实查询，本机）：

| 成分 | 耗时 | 占比 |
| --- | --- | --- |
| `tokenize(query)` **在每个文档上重算一次**（0.026 ms × 17524） | ~454 ms | **~83%** |
| 遍历 17524 个文档的空循环 | 0.5 ms | 0.1% |
| 其余（dict 查找、打分、排序） | ~90 ms | ~17% |
| **合计（`search` 实测中位数）** | **~460–550 ms** | |

根因在 `bm25.py:129`：`score()` 内部写着 `for term in tokenize(query)`，而 `search()`
（`bm25.py:143`）对**每个文档**调用一次 `score()`。于是查询被分词了 17524 次。
`idf(term)` 同理，每个 (文档, 词) 组合都重算一次 `math.log`。

验证：只把 `tokenize`/`idf` 提到循环外、其余一字不改，
**462 ms → 119 ms（省 74%），且 30 条查询的 `(doc_id, score)` 逐位相同。**

**这不改变"要建倒排索引"的结论**，但它改变了报告的诚实度：
如果只做倒排却不提这一点，读者会以为 28× 的加速全部来自数据结构。

### 0.1 倒排索引省的到底是什么

| 量 | 数值 |
| --- | --- |
| 文档数 | 17,524 |
| 唯一 term 数 | **177,604** |
| 总 posting 数（Σ 每文档唯一 term） | **2,957,912** |
| 总 token 数 | 4,147,987 |
| 文档长度 | 中位 146 / 均值 237 / p95 791 |
| 查询的唯一 term 数 | 中位 **34** |
| 每查询要遍历的 posting 数（Σ df over query terms） | 中位 **42,914** |
| query term 的 postings **并集** | **17,524 = 100% 的文档** |

**关键事实：并集是 100%。** 语料里有 `agent`（df = 17,524，出现在每一篇）、
`note`（17,512）、`the`（9,521）这类词，所以"只对候选集打分"**并不会减少文档数**。

倒排真正省掉的是**无效的 (文档, 词) 组合**：
现状要做 17,524 × 34 ≈ **596,000 次 dict 查找**，其中只有 **42,914 次**命中（tf > 0）。
倒排直接只遍历那 42,914 个 posting——**内层操作减少约 14×**。

> 所以正确的说法是"跳过 tf=0 的组合"，不是"缩小候选集"。

---

## 1. postings 的数据结构

### 1.1 选型

```python
_slots: list[str | None]  # slot -> doc_id（None = 已删除的空洞）
_index_of: dict[str, int]  # doc_id -> slot
_free: list[int]  # 可复用的 slot
_doc_terms: list[tuple[str, ...] | None]  # slot -> 该文档的唯一 term（删除与 score 要用）
_doc_len: list[int]  # slot -> token 数
_total_len: int  # running sum -> avgdl 变成 O(1)
_postings: dict[str, list[int]]  # term -> **升序**的 slot 列表
_posting_tfs: dict[str, list[int]]  # term -> 平行的 tf 列表
_df: Counter[str]
_norms: list[float] | None  # slot -> 长度归一化项的缓存，惰性重建
```

**postings 用 slot 下标（int）而不是 doc_id 字符串**：内层循环用 int 索引一个 list，
比 dict[str] 查找快得多，也省内存。

**tf 与 df 怎么存**：tf 放在与 postings 平行的第二个 list（实测最大 tf = 48）；
df 本来打算继续用 `Counter` 维护。

> **实现时的偏离（2026-09-18，见 impl-006 §3.3）**：`_df` 被**删掉了**。
> 一个 term 的 df 就是它 posting 列表的长度，单独再存一份 `Counter` 是同一事实的第二份副本
> ——既多花 0.39 s 的构建时间与一份 177,604 项的 Counter，又给"两份数据不同步"留了口子。
> 现在 `document_frequency(term)` 直接返回 `len(self._postings.get(term, ()))`，
> `idf()` 在其之上。上面的结构清单里那一行 `_df: Counter[str]` 因此不存在于最终实现。

**为什么 postings 必须保持升序**：`score()` 与 `remove()` 都要"给定 term 和 slot 求 tf/位置"，
升序才能用 `bisect` 做 O(log df) 查找，不必再存一份每文档的 `{term: tf}`。
批量构建时 slot 单调递增，`append` 天然有序；只有复用 `_free` 里的 slot 时才需要
`bisect.insort`，而那是冷路径（先比一次末尾元素，常见情况仍走 `append`）。

### 1.2 内存预算（实测，不是估算）

估法：先用 `collect_chunks` 得到真实的 17,524 个 chunk，预先分词（把分词的临时对象排除在外），
再用 `tracemalloc` 分别测四种布局的常驻内存。**唯一 term 数 177,604 是数出来的，不是猜的。**

| 布局 | 常驻内存 | 说明 |
| --- | --- | --- |
| A. 只有倒排（postings + tfs） | 88.2 MB | 2.96M 个 int 对象是主要成本 |
| **B. 倒排 + 每文档 term 元组（本设计）** | **111.6 MB** | A + 2.96M 个字符串引用（8 B 各）= +23.4 MB |
| C. 现状（每文档一个 `Counter`） | 78.7 MB | 每篇一个 dict，~170 项 |
| D. 现状 + 倒排（两份都留） | 167.0 MB | 改动最小但最费内存 |

**选 B，代价是相对现状 +33 MB（+42%）。** 不选 D 的理由：多 88 MB 只为省掉重写
`remove()`/`score()` 的功夫，不划算；而且两份结构并存必然有一天会不同步。

`array('i')/array('H')` 版本另测过：缓冲区只要 16.9 MB，但**容器与对象开销另有 ~42 MB**，
查询还慢 6%（`array` 取值要现造 int 对象，而 list 里的小 int 有缓存）。
**不采用**，理由写在第 6 节。

全量常驻（含 `_df`、`_slots`、`_doc_len` 等）预计从现状的 **246.7 MB** 变到 **~280 MB**，
实现完会实测复核。

---

## 2. 查询打分流程与公式等价性

### 2.1 流程

```
q_tokens = tokenize(query)                    # 每次查询只做一次（现在做 17524 次）
idf      = {t: self.idf(t) for t in set(q_tokens)}   # 每个唯一 term 只算一次 log
norms    = self._ensure_norms()               # slot -> k1*(1-b+b*dl/avgdl)，惰性缓存
scores   = [0.0] * len(self._slots)           # 稠密数组：并集本来就是 100%
for term in q_tokens:                         # **按 query token 顺序，含重复**
    docs, tfs = self._postings.get(term), ...
    w = idf[term]
    for j, slot in enumerate(docs):
        tf = tfs[j]
        scores[slot] += w * (tf * (k1 + 1)) / (tf + norms[slot])
out = [(doc_id, s) for slot, s in enumerate(scores) if s > 0.0 and slots[slot] is not None]
return heapq.nsmallest(limit, out, key=lambda it: (-it[1], it[0]))
```

### 2.2 公式逐项对齐

| 量 | 现状 | 新实现 | 是否逐位相同 |
| --- | --- | --- | --- |
| `avgdl` | `sum(_doc_len.values()) / len(_doc_len)` | `_total_len / live_docs`（running sum） | **是**——同样是一次浮点除法；分子是整数精确和 |
| `idf(term)` | 每个 (doc, term) 重算 | 每个唯一 term 算一次 | **是**——纯函数，值相同 |
| 长度归一化 | `k1*(1-b+b*dl/avgdl)` 在 `denom` 里现算 | 预计算进 `norms[slot]`，表达式**逐字相同** | **是** |
| 单项贡献 | `idf * (tf*(k1+1)) / denom` | 同一表达式、同一结合顺序 | **是** |
| 累加顺序 | `for term in tokenize(query)`（**含重复**），跳过 tf=0 | 同样按 `tokenize(query)` 顺序外层遍历；某文档只会在它出现在该 term 的 postings 时被加 | **是**（见 2.3） |
| 过滤 | `score > 0.0` | 同 | 是 |
| 排序 | `sort(key=(-score, doc_id))[:limit]` | `heapq.nsmallest(limit, key=(-score, doc_id))` | **是**（见 2.4） |

### 2.3 浮点累加顺序：不变，这是刻意设计的

现状对文档 d 的累加是「按 `tokenize(query)` 的顺序，跳过 tf=0 的 term」。
新实现外层遍历 `tokenize(query)`、内层遍历该 term 的 postings，于是**对任意固定的 d，
贡献到达的先后顺序完全一致**（d 不在 postings[t] 里 ⟺ 现状的 tf=0 被 `continue` 跳过）。
初值同为 `0.0`。**所以不是"差异小到可以忽略"，而是逐位相同。**

> **一个必须保留的既有怪癖**：`tokenize()` 返回**带重复**的列表，而 `score()` 直接遍历它。
> 所以查询里出现两次的词会被**计两次分**。这不是标准 BM25（标准做法是按唯一 term 计一次，
> 或用 query tf 加权）。我**原样保留**，因为本轮的硬要求是逐位等价。
> 若要改成标准做法，那是一次**会改变所有历史数字**的独立变更，应当单独立项并重跑基线。

原型验证：30 条真实查询，新旧 `(doc_id, score)` 列表 **0 条不同**。

### 2.4 `heapq.nsmallest` 与 `sorted(...)[:n]` 等价

CPython 文档明确 `nsmallest(n, it, key)` 等价于 `sorted(it, key=key)[:n]`。
本场景的 key 是 `(-score, doc_id)`，而 `doc_id` 在索引内唯一 ⇒ **key 是全序、不存在并列**，
所以稳定性差异无从体现。实测：17,524 条构造数据上两者结果**完全相同**，
耗时 **11.5 ms → 2.0 ms**。

这一步值得做，因为倒排之后**排序就成了新的瓶颈**：打分只剩几毫秒，
而"收集 + 排序 17,524 条"要 14.7 ms。

---

## 3. `remove()` 怎么办：墓碑 vs 立即清理

| 方案 | 删除成本 | 查询成本 | 空间 | 语义 |
| --- | --- | --- | --- | --- |
| 墓碑标记 | O(该文档 term 数)，只改 df | **每个 posting 多一次 `is None` 判断** | 陈旧 posting 不回收 | 需要在查询里过滤 |
| **立即清理（本设计）** | O(Σ df over 该文档的 term)，实测 ~亚毫秒 | **零额外开销** | 立即回收 | 与现状完全一致 |

**选立即清理。** 决定性理由是：墓碑的代价落在**内层热循环**上——而内层循环正是本轮要优化的东西。
删除是冷路径（今天 `build_keyword_index()` 每次重建，`remove()` 只被 `add()` 的重建分支
和测试调用），把成本挪到冷路径是正确的方向。

实现：对 `_doc_terms[slot]` 里的每个 term，`bisect_left` 定位后 `del` 两个平行列表的同一位置；
posting 列表空了就删 key；`_df[term] -= 1`，减到 0 就删。然后清空 slot、压入 `_free`、
`_total_len -= _doc_len[slot]`、`_norms = None`。

**slot 复用与有序性**：复用 `_free` 里的 slot 会破坏 postings 的升序，所以 `add()` 在
插入时先比一次末尾元素——大于就 `append`（批量构建的常见情况），否则 `bisect.insort`。

**`_norms` 缓存的失效**：`avgdl` 一变，所有文档的归一化项都变。所以 `_norms` 是一个
**惰性缓存**，任何写操作把它置 `None`，下一次查询 O(n) 重建。
"构建一次、查询很多次"的模式下这是免费的；而这也顺手修掉了现状
`add()` 每次都 `sum(_doc_len.values())`（O(n)）造成的 **O(n²)** 构建——实测那一项就占 **2.3 s**。

---

## 4. 等价性怎么证明（可执行）

三层，逐层收紧：

1. **单元层**：`tests/test_bm25.py` 现有 11 条行为契约必须**原样通过**（不改断言）。
2. **打分层（新增测试）**：用固定种子造一批文档与查询，断言
   `new.search(q, limit=-1) == old.search(q, limit=-1)`（**`==` 逐位**，不设容差）。
   把旧实现作为参考实现内联进测试，这样它不会随重构漂移。
3. **端到端层**：在真实语料（17,524 chunk）+ 真实评测集的 296 条查询上，
   对比新旧 `(doc_id, score)` 列表，要求 **0 条不同**；再跑一次
   `note-rag eval`，要求 **strict 视图下 `keyword` 的 recall@1/5/10、MRR、nDCG@k
   与 `eval/out/ablation.json` 里的基线逐位相同**。

**容差定为 0**（第 2.3 节论证了累加顺序不变）。若出现任何差异，按任务书要求
**不调容差**，而是定位根因；只有在确认根因不可消除时才另立新基线。

---

## 5. 分阶段实施

| 步 | 内容 | 验收判据 |
| --- | --- | --- |
| 1 | 把 `tokenize(query)`/`idf` 提到查询循环外（**不动数据结构**） | `pytest` 绿；296 条查询与旧实现逐位相同；延迟应降到 ~120 ms |
| 2 | 引入 postings + slot + `_total_len` + `_norms` 惰性缓存，重写 `search`/`score`/`add`/`remove` | 同上判据 + 内存实测；延迟应降到 ~15 ms |
| 3 | `sorted[:n]` → `heapq.nsmallest` | 同上判据；延迟再降 ~3 ms |
| 4 | 端到端：重跑 `note-rag eval`，比对 `ablation.json` 基线 | strict 视图下 `keyword` 各指标逐位相同 |
| 5 | 按 Batch B 的方法复测延迟（3 轮重复、模式交错），判定是否 < 320 ms | 报中位数 + 极差 + 可区分性 |

每一步都可以独立回滚，且第 1 步单独就能拿到 74% 的收益——**万一第 2 步出问题，有退路**。

---

## 6. 明确不做的事

| 不做 | 为什么 |
| --- | --- |
| **高 df term 剪枝 / 停用词表** | 收益很大（`agent` 的 df = 100%，idf ≈ 2.9e-5，几乎不影响排序却要遍历 17,524 个 posting），**但它会改变分数**，与本轮"逐位等价"的硬要求直接冲突。应当单独立项，并按 exp 流程重跑基线。 |
| **WAND / MaxScore 等动态剪枝** | 同上，会改变"哪些文档被打分"，且只有在 top-k ≪ N 时才划算。当前 k=50、N=17.5K，先把常数项降下来收益更直接。 |
| **`array('i')` 紧凑 postings** | 实测缓冲区省 16.9 MB，但容器开销另有 42 MB，查询还慢 6%。等语料上到百万级、或真的要压内存时再说。 |
| **把 tf 压成 `uint8`/变长编码** | 最大 tf 只有 48，确实能压，但同上：现在的瓶颈不是内存。 |
| **并行/多进程打分** | 单次查询已降到 ~15 ms，引入进程池的开销比收益大。 |
| **把 BM25 换成 Postgres 全文检索** | `bm25.py` 的存在理由之一就是"不依赖数据库也能跑评测"（模块 docstring 第 1 条）。 |
| **修"查询重复词计两次分"这个怪癖** | 见 2.3：会改变所有历史数字，必须单独立项。本轮如实记录。 |
| **把 `build_keyword_index()` 改成后台异步** | 属于 API 生命周期的问题，见下。本轮只**测量**启动成本并给建议，不改 `api.py`。 |

## 7. 顺带要回答的问题：API lifespan 里同步建索引还合适吗

现状 `build_keyword_index()` 在 `api.py` 的 lifespan 里同步执行：从 Postgres 读 17,524 行 →
`BM25.extend()`。实测 `extend()` 本身 **3.71 s**（其中约 **2.3 s** 是 `add()` 里
`sum(_doc_len.values())` 造成的 O(n²)），加上读库与分词，启动大约在 5 s 量级。

本设计把 avgdl 改成 running sum，**这 2.3 s 直接消失**。实现完会实测新的构建时间，
再据此判断"5 s 的冷启动是否还需要异步化"——如果降到 2 s 以内，
为它引入后台刷新 + 陈旧读的复杂度就不划算了（`retriever.py` 的 `_ensure_index()`
已经保证了"漏建索引不会静默降级"这一更重要的性质）。
