# 任务书 003：评测口径修复 Batch A（数据层）+ 重跑消融

> 交给 **Claude Code + Opus 5**（交互式）。你在这个仓库里可以直接改代码、连 Postgres、跑评测。
> DSH 负责准备这份任务书、复核产物、做 git/CI 记账。
>
> **改动范围**：`src/note_rag/notes.py`、`src/note_rag/cli.py`、`scripts/build_eval_dataset.py`
> **允许**：本地 commit（一条，信息写清改了什么与为什么）
> **不允许**：`git push`（DSH 会 review diff 后推送）、改动 BM25 参数、改动检索阈值

---

## 背景：你上一轮（审查 001）审出的问题，现在来修

完整结论在 `docs/reviews/review-001-eval-methodology.md`。本轮只修**数据层**的四条
（S2/S10、S5、S8、S1），它们都会改变数据集或评测的有效性，所以必须一起修完再重跑。
报告层的问题（S4 拒答表、S6 改叫法、S7 语言与聚类 bootstrap）留到 Batch B。

**已由 DSH 完成、不要重复做**：
- S3（`hnsw.ef_search`）已修：`config.py` 新增 `hnsw_ef_search: int = 200`，
  `store.set_hnsw_ef_search()` 在 `Retriever.from_settings` 里调用。证据见
  `docs/experiments/exp-002-hnsw-ef-search.md`（recall@10 0.3182 → 0.4040，p95 不变）。
- S9（落盘 `ablation.json`）已修。

---

## 要修的四条

### 1) S2 + S10：让索引器与评测集共用同一个文件发现规则，并加一条断言

**根因**（你的原文）：`notes.py` 的 `iter_vault_files` 跳过"任一路径段以 `.` 开头"的文件，
而扁平化语料里有文件名本身就以 `.agents__` 开头；`build_eval_dataset.py` 用
`corpus_dir.glob("*.md")`（pathlib 的 glob **包含**点开头文件）。结果 1947 篇里只有 1945 篇进了索引，
q0001–q0003 的目标笔记从未被索引——对所有模式永久记 0。

**请这样修**：

- **优先修索引器一侧**：只跳过点开头的**目录**（`.obsidian` / `.trash` / `.git` 等），
  **不要**因为文件名以 `.` 开头就跳过一个 `.md` 文件——那会静默丢掉真实内容，正是这个 bug 的本质。
  请自行确认这个改动对真实 Obsidian 库也安全（我们只枚举 `.md`/`.markdown`）。
- 同时把 `build_eval_dataset.py` 改为使用 `note_rag.notes.iter_vault_files`，
  让两侧**共用同一个规则**（防御性：即使将来规则再变，也不会再错位）。
- 在 `cmd_eval` 里加开场断言（S10，你的原话是"投入产出比最高的一条修复"）：
  `qrels` 里出现的所有笔记必须存在于已索引的 note 集合中，否则**直接报错**并列出前几个未知项，
  而不是静默记 0。为此可以在 `Retriever` 上加一个只读方法（例如 `indexed_notes()`）
  返回已索引的 note 路径集合——请顺带加对应单测（无需数据库，用 monkeypatch 的假 store 即可，
  参考 `tests/test_retriever.py` 的现有写法）。

### 2) S5：负样本稀有度闸门有两个 bug，它从未拒绝过任何东西

- **大小写 bug**：`tokenize()` 会 lowercase，但 df 是拿小写 token 去**原始大小写**正文里做子串匹配
  → `kubernetes` 记成 df=0，而 `Kubernetes` 实际出现在 2 篇里。
- **判据方向反了**：现在取**最小** df → 只要问题里存在任意一个冷门片段就通过，
  于是 `注中` / `么处` / `持哪` 这类中文二字组垃圾全部过关，18/18 通过。

**请这样修**：修正大小写；把判据换成你建议的方向（内容词的 df **上界**、
或用现成 `BM25` 的 top-1 分数与"可回答问题 10 分位"比较）。修正后**必须重新生成
`eval/dataset.report.md`**，让"被丢弃的负样本"那一节真的可能非空。
**注意**：你已抽查确认 18 条负样本本身都是不可回答的（"数据是好的，坏的是闸门"），
所以修好闸门后大概率仍是 18 条通过——**这是可以接受的结果，不要为了"让闸门看起来有效"
而调阈值把好样本丢掉**。如实报告即可。

### 3) S8：把译文孪生加进 `relevant_notes`（换嵌入模型前必做）

语料是严格双语镜像（972 篇 `.zh__*` + 973 英文，按 slug 配对）。60 篇目标笔记里 **59 篇**
存在内容等价的译文孪生，现在它被判为不相关。今天纯词法检索跨不过语言边界所以没污染，
但**换真实多语言嵌入后孪生会挤进 top-k，造成假性 recall@1 下降**。

请按你的建议实现：编译期把孪生加进 `relevant_notes`；**推荐直接用 `metrics.py` 已支持的分级相关**
（原文 grade 2、孪生 grade 1），这样 `nDCG` 才终于有判别意义（呼应 S6）。
注意：改成分级相关会**改变现有指标的语义与数值**，请在报告里明确列出"改动前后"的对照。

### 4) S1：加词法泄漏闸门（最高优先级那条）

问题是由模型**读着目标笔记**生成的，会系统性地把该笔记的稀有标识符抄进问题：
df ≤ 2 那一组 `keyword` R@5 高达 **1.000**，df > 50 组只有 **0.676**。

请按你的最小修复方案实现：

1. 用 `note_rag.bm25.tokenize` 对全语料算一次 note 级 df（一次性）。
2. 对每条候选问题，取"问题 ∩ 目标笔记"的 token 里最小的 df；`min_df <= 2` 的条目标记
   `leak: true`（**不要直接删**——被标记的样本本身是好材料），并在报告里单独出分布表。
3. **报告里按 df 分组出数**（≤2 / 3–10 / 11–50 / >50），不要只给总均值。

---

## 修完之后必须做的三件事

1. **重新 ingest**（`uv run note-rag ingest`）——S2 修好后索引会多出那 2 篇被跳过的文件。
   容器应该已经在跑（`docker compose ps` 确认；不在就 `docker compose up -d pg redis`）。
2. **重新编译数据集**：`uv run python scripts/build_eval_dataset.py`
3. **重跑四模式消融**：`uv run note-rag eval --dataset eval/dataset.jsonl --out eval/out/ablation.md`

## 必须产出的报告（DSH 靠这份报告接手）

请新建 `docs/reviews/impl-003-callibration-batch-a.md`，包含：

1. **每条的改法**：文件:行号 + 为什么这样改（尤其 S2 你选了修索引器还是改语料命名，以及理由）
2. **门禁结果**：`uv run ruff check .` / `ruff format --check .` / `uv run mypy` / `uv run pytest`
   四条的实际输出（**必须是真跑过的**，不要写"应当通过"）
3. **口径修复前后对照表**：改动前的数字（`keyword` R@1/5/10/MRR = 0.5758/0.7879/0.8283/0.6650 等）
   与改动后逐模式对照，并说明**哪些差异是口径造成的、哪些是 S2 补进来的 3 条问题造成的**
4. **S1 的 df 分组表**（四组 × 三模式的 recall@5）
5. **新引入的不确定性**：哪些数字现在不该被引用
6. **仍然没修的问题清单**（S4/S6/S7 留给 Batch B，以及任何你在本轮发现的新问题）

## 硬性约束

- 不要让测试为了通过而放宽断言；不要跳过测试。
- 保持 `mypy --strict` 零错误；不引入新依赖。
- **不要改** BM25 的 `k1`/`b`、RRF 的 `k`、`candidate_k`、chunk 参数——这些要有评测数据支撑才动。
- 报告里的每个数字都要来自你**实际执行**的命令输出。
