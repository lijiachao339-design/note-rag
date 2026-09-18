# 任务书 007：S5 换成真实语义嵌入（本地 CPU，bge-m3）

> 交给 **Claude Code + Opus 5**（交互式）。这一步要联网下载模型、连 Postgres、跑长时间任务。
> 前置：prompt-006（BM25 倒排索引）已完成。**如果你还没做 006，先做 006**——它只改延迟，
> 先做完可以让本轮的延迟数字一次成型，不用测两遍。
>
> **允许**：本地 commit
> **不允许**：`git push`；不要改 `eval/dataset.jsonl` 与 `scripts/build_eval_dataset.py`（DSH 在并行补负样本）
> **不许改**：BM25 的 `k1`/`b`、RRF 的 `k`、`candidate_k`、分词器、chunk 参数、评测口径

---

## 目标与意义（先说清为什么值得花时间）

当前 `vector` 这一路用的是**离线 hashing 嵌入**——它不是语义模型，只是"有损的词法检索"
（token 哈希到 384 个桶）。exp-001 已经量化过它的上限。

换成真实语义嵌入之后，才可能出现 **`hybrid` > `keyword`**。这是简历上"混合检索把 recall@5
从 X 提到 Y"这句话唯一的成立条件。**在此之前，任何版本的 hybrid>keyword 都是编的。**

用户选择本地 CPU 方案（而不是 API）的理由是：**免费、无 key、可离线复现**——
面试官能自己跑出同样的数字。代价是这一步耗时较长。

## 硬性顺序（顺序错了会白跑）

### 第 0 步：先测吞吐，再决定全量跑

**不要一上来就嵌 17524 个 chunk。** 先：

1. 装载模型，对 **200 个 chunk** 计时，算出 **chunks/秒**。
2. 外推全量（17524 chunk）的 ETA，并把 ETA **报告出来**。
3. 如果 ETA 超过约 3 小时，先停下来汇报，一起决定是否换更小的多语言模型
   （备选顺序：`BAAI/bge-m3` → `intfloat/multilingual-e5-small`（384 维，快得多））。
   全量跑之前先确认，比跑了两小时发现要重来划算。

**模型下载**：`BAAI/bge-m3` 约 2.2 GB。国内直连 `huggingface.co` 经常失败或极慢；
若下载失败，设置 `HF_ENDPOINT=https://hf-mirror.com` 重试（国内常用镜像）。
若报 `sentencepiece` 缺失，补装它（XLM-R 分词器需要）。**下载后校验模型维度是 1024。**

### 第 1 步：实现 `LocalEmbedder`，但**不要把 torch 拖进默认依赖**

`src/note_rag/embed.py` 已有 `Embedder` Protocol 与 `build_embedder()` 工厂，照着扩展：

- 实现 `LocalEmbedder`：**懒导入** `sentence_transformers`（顶层 import 会让没装 torch 的环境直接崩）、
  L2 归一化（与 `RemoteEmbedder` 一致，`store` 用余弦距离）、可配 batch size、打印进度。
- 依赖放进**可选依赖组**（例如 `[project.optional-dependencies]` 的 `local-embed`），
  **不要**进主 `dependencies`：CI 跑 `uv sync --all-groups`，把 torch 拖进去会让每次 CI 多下 ~1 GB。
  同时确认 CI 不装这个 extra 时仍然全绿（`build_embedder` 在无配置时回落到 `HashingEmbedder`）。
- **不要动** `HashingEmbedder`：它是可复现的基线，也是没有模型时的回落路径。

### 第 2 步：修 `ensure_schema` 的维度守卫（`AGENTS.md` 遗留项 6）

现在 `vector(dim)` 是建表时固定的（当前是 384），换 1024 维模型会直接在插入时报一个看不懂的错。
要求：

- `ensure_schema` 检测**已有列的实际维度**，与请求的维度不一致时给出**可操作的报错**
  （说明现状、说明该怎么办）。
- 提供一条**显式的重建路径**（例如 `--recreate` 或单独的 CLI 子命令），而不是自动偷偷 DROP 表。
  自动删数据是危险的默认行为；但"知道怎么删"必须是一行命令。
- 这条要有测试（可以用 monkeypatch 假连接断言 SQL/报错分支，不必真连库）。

### 第 3 步：全量嵌入并重建索引

```powershell
# .env 里设置（示例）
# NOTE_RAG_EMBED_MODEL=BAAI/bge-m3
# NOTE_RAG_EMBED_DIM=1024
uv run note-rag ingest --full
```

**可断点续跑是已经具备的**：`ingest.py` 用 `content_hash` 做增量，每个 batch 写完就 upsert。
所以中途崩了直接重跑即可，已嵌入的会跳过——**请确认这一点真的成立**（跑一次小规模中断再续跑的验证）。
`--full` 是必须的：换模型等于换向量空间，旧向量不能留。

### 第 4 步：重跑消融，并**验证 S8 的预测**

```powershell
uv run note-rag eval --dataset eval/dataset.jsonl --out eval/out/ablation.md
```

用同一套方法（重复测量取中位数）给出新表，并与**已提交的 hashing 基线**（`eval/out/ablation.json`）
逐视图对照：`strict` / `graded` / `strict_no_leak` + 分语言。

**必须明确回答这个可证伪的预测**（来自 review-001 的 S8）：

> 666/269 → 266/269 条问题有一篇**内容等价、只是语言不同**的译文孪生，而它现在也在
> `relevant_notes` 里（grade 1）。今天的纯词法检索跨不过语言边界，所以孪生几乎从不被召回
> （v2 实测 `keyword` 进 top-5 只有 2/266）。
> **换成多语言嵌入之后，孪生会被排到目标笔记紧邻的位置。** 于是：
> - `graded` 视图的 nDCG 应当**上升**（孪生被找回一半的分）
> - `strict` 视图的 `recall@1` 可能**下降**（孪生占掉一个槽位，甚至排在原文之前）

请给出：孪生现在进入 top-5 / top-10 的比例、排在被标注笔记之前的比例、以及
`strict recall@1` 与 `graded nDCG@10` 的实测变化。**预测成立就写成立，不成立就写不成立**——
S8 的价值在于"提前修好了标注，所以没有出现假性回归"，如果实测发现孪生依然不被召回，
那也是一个结论（说明这个嵌入模型在这批数据上的跨语言能力不足）。

## 必须产出的报告

`docs/reviews/impl-007-local-embeddings.md`，包含：

1. 每步改了什么（文件:行号 + 理由）；依赖是怎么放进 optional extra 的、CI 如何不受影响
2. **四条门禁的真实输出**：`uv run ruff check .` / `ruff format --check .` / `uv run mypy` / `uv run pytest`
3. **吞吐与 ETA 实测**（200 chunk 的计时、chunks/秒、全量实际耗时）
4. **新旧向量对照表**（三视图 + 分语言），并归因：哪些差异来自"嵌入有了语义"、
   哪些来自"孪生被召回了"、哪些来自语言配比
5. **S8 预测的验证结论**（成立/不成立 + 数据）
6. **`hybrid` 与 `keyword` 的关系现在是什么**——这是本轮最重要的一句话。
   如果 hybrid 仍不及 keyword，给出你判断的原因（RRF 等权？`candidate_k`？向量仍不够强？），
   并说明下一步该动什么、依据是什么
7. 仍然没做的事 + 下一步建议

## 硬性约束

- 不要让测试为了通过而放宽断言；不要 skip 测试。
- 保持 `mypy --strict` 零错误。
- 报告里每个数字都要来自你**实际执行**的命令输出。
- **不要**为了让 `hybrid` 看起来更好而调 RRF 权重或 `candidate_k`——那要等对照数据出来再谈。
- 发现任务书有错或不可行，直接指出并说明理由（前两轮你都这么做了，两次都对）。
