# 审查请求 001：检索评测口径

> 这是交给 **Claude Code + Opus 5** 的审查任务提示词。由 DSH 通过
> `claude -p --model opus --allowedTools "Read,Glob,Grep"` 非交互调用，产出存为
> `docs/reviews/review-001-eval-methodology.md`。
> 提示词与产出都落盘，保证"每一步用了哪个工具"有据可查。

---

你在审查一个检索项目的**评测口径**。口径一旦错了，后面所有对比数字都要作废，所以请严格、具体，
不要给泛泛的建议。

## 要审的文件（都在当前仓库，用你的只读工具直接读）

- `src/note_rag/metrics.py` —— 指标定义（recall@k / precision@k / MRR / nDCG@k / percentile）
- `src/note_rag/retriever.py` —— `search()` 与 `search_notes()`（note 粒度去重在这里）
- `src/note_rag/cli.py` 的 `cmd_eval()` —— 评测入口，ranking 如何构造
- `scripts/build_eval_dataset.py` —— 评测集编译规则（`answer_span` 逐字幻觉闸门、负样本稀有度筛选）
- `eval/dataset.jsonl` —— 198 行数据集（180 条可回答 + 18 条不可回答）
- `eval/dataset.report.md` —— 构建报告（丢弃原因分布、难度分布、负样本明细）
- `docs/experiments/exp-001-baseline-hashing-embedder.md` —— 当前结论

## 已实测的事实（不需要你重新验证，但可以质疑）

- 语料 1947 篇 markdown，索引 17520 个 chunk
- 数据集：180 条可回答（135 英文 + 45 中文，问题语言与正文语言一致）+ 18 条不可回答
- 基线结果：`keyword` recall@5 = 0.7879 ＞ `hybrid` 0.6111 ＞ `vector` 0.2677；
  p95 延迟 `vector` 54 ms / `keyword` 709 ms
- 当前嵌入是"离线 hashing 嵌入"（token 哈希到 384 维桶），不是语义模型
- reranker 是 no-op，所以 `hybrid_rerank` 与 `hybrid` 完全相同

## 请逐项回答，每条都要给出 `文件:行号` + 最小修复方案

1. **note 粒度去重的位置偏差**：`search_notes()` 先把 chunk 结果去重成"每篇取最佳 chunk"。
   如果一个问题的答案分散在同一篇笔记的多个 chunk 里，当前做法会**低估**还是**高估** recall？
   有没有更中性的口径？
2. **不可回答问题的计入方式**：`metrics.py` 把它们记为 0 分但保留在均值里。
   这个口径在 recall@k / MRR / nDCG@k 上分别意味着什么？
   会不会**惩罚一个合理地返回空结果的检索器**？该不该单独报告这一类？
3. **`answer_span` 逐字校验的边界**：它能挡住哪些幻觉、挡不住哪些？
   特别地：片段真实存在于原笔记，但**并不能回答该问题**——这种情况现在会不会漏过？
4. **语言不均衡**：135 英文 + 45 中文。在纯词法检索下这会不会让某些模式的指标失真？
   应该分语言报告吗？如果分，怎么避免中文那一组样本量太小导致结论不稳？
5. **负样本是否真的不可回答**：抽查 `eval/dataset.jsonl` 里 `answerable: false` 的 5 条，
   用你的文件工具在 `data/corpus/` 下搜关键词确认（`data/` 不在 git 里，但文件在磁盘上）。
   如果发现某条其实能回答，请指出是哪条。
6. **最重要的一个**：用当前这套口径，我能不能诚实地在简历/README 里写
   "hybrid 检索把 recall@5 从 X 提升到 Y"？如果不能，缺什么？
   请明确区分"口径问题"与"模型能力问题"。

## 输出要求

- 用 markdown，按**严重程度从高到低**排序
- 每条：现象 → 为什么是问题 → `文件:行号` → 最小修复方案
- **禁止修改任何文件**，只出审查意见
- 如果某一项你认为当前做法是对的，直接说"无问题"，不要为了凑字数编问题
