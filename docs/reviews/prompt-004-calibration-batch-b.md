# 任务书 004：评测口径修复 Batch B（报告层）+ 延迟测量规范化

> 交给 **Claude Code + Opus 5**（交互式）。你在本仓库可以改代码、连 Postgres、跑评测。
> 前置：Batch A 已完成（提交 `6675ade`），报告在 `docs/reviews/impl-003-callibration-batch-a.md`。
>
> **允许**：本地 commit（一条，信息写清改了什么与为什么）
> **不允许**：`git push`、改动 BM25 的 `k1`/`b`、RRF 的 `k`、`candidate_k`、chunk 参数

---

## 本轮范围：审查 001 剩下的报告层问题 + 你自己提出的延迟噪声问题

Batch A 修完了**数据层**（S2/S10、S5、S8、S1）。本轮修**报告层**，
目标是让"引用某个数字"这件事变得安全。同样改完必须重跑并给出前后对照。

### 1) S4：拆成两张表，并修掉一句错误注释

现状：18 条不可回答问题在 `recall_at_k` / `reciprocal_rank` / `ndcg_at_k` 上**恒等于 0**，
与检索器返回什么完全无关——它只是把所有数字统一乘了 `180/198`，零判别力。

**要求**：

- **排序质量表**：只统计 180 条可回答问题。`cmd_eval` 里按 `row["answerable"]` 切片即可，
  `metrics.evaluate` 本身不用改。
- **拒答表**：单独报。你在 Batch A 里已经找到比 BM25 top-1 更好的置信信号
  （报告 2.2 节，AUC 约 0.96 对 0.729），请把它的选择理由与实现固化下来，报两个数：
  `abstain_rate_on_unanswerable`（越高越好）与 `recall@5_on_answerable @ 同一阈值`（代价）。
  阈值请**在报告里注明是怎么定的**，不要藏在代码里。
- **修掉那句错误注释**：`src/note_rag/metrics.py` 的模块注释声称不可回答问题"就是抓住幻觉式检索器的办法"，
  但代码里这条路径恒返回 0，注释与实现不符。`tests/test_metrics.py` 里
  `test_recall_with_no_relevant_documents_is_zero_not_error` 的注释同样固化了这个误解。两处都要改。
- 注意：`metrics.py` 的 `evaluate()` 返回值现在被 `cmd_eval` 塞进了 `p50_ms`/`p95_ms`，
  而 `tests/test_metrics.py::test_metrics_are_bounded` 断言所有值在 `[0,1]`。
  请把延迟指标与质量指标在数据结构上分开，别让那条断言成为地雷。

### 2) S6：把叫法改对

- `recall@k` 在单标签数据下其实是 **Success@k / Hit Rate@k**（不是教科书 recall）。
  在文档与输出里标注清楚（`README`、`cli.py` 的表头或脚注、`metrics.py` 的 docstring）。
- `precision@k` 恒等于 `recall@k / k`，从 stdout 报告里去掉。
- 你的 S8 已把相关性改成分级（原文 grade 2、孪生 grade 1），所以
  "单标签下 nDCG 与 MRR 是同一个统计量"**已经不成立**——请在报告里用实测证明这一点
  （给出 nDCG@10 与 MRR 现在确实分道扬镳的对照），并把这条结论写进 `README` 的"怎么读这张表"。

### 3) S7：语言切片 + **按笔记聚簇**的置信区间

- 编译期给数据集加 `lang` 字段（用 `re.search(r'[\u4e00-\u9fff]', question)` 判定即可）。
- `cmd_eval` 按 `lang` 切片报告。
- **bootstrap 的重采样单位必须是 `note_path`（聚簇），不是问题**：180 条问题来自 60 篇笔记，
  同一篇笔记的 3 条问题高度相关，按问题重采样会低估方差（实测聚簇 CI 宽 23%）。
- **报区间不报点值**。20 行代码，不引依赖。
- 中文组只有 15 篇笔记。**不要**用更宽的置信区间糊过去，而是明确标注
  `n=15 notes, 仅供参考`，并在报告里写清"要把中文组补到 ~45 篇笔记才可靠"——
  补数据由 DSH 的 `workflow` 扇出另做（写进你的"下一步"清单即可，本轮不要自己生成新问题）。

### 4) 延迟测量规范化（你自己提出的问题）

你的原话：单次串行测量的噪声已经大于模式间真实差异（`hybrid` 的 p50 有时低于 `keyword`，
而它必然跑两路）。请把这件事**修成方法而不是注释**：

- 每个模式重复 N 次（N 由你定并说明理由），报告 **重复之间的中位数**，以及重复间的离散度。
- 在报告里明确：延迟数字的可信区间是什么量级，哪些模式间差异**不可区分**。
- 这一条会直接影响后面 BM25 倒排索引改造的验收判据（709ms → 目标 50ms），所以必须现在定下来。

### 5) 顺手把 Batch A 的产物提交进版本库

DSH 已把 `eval/out/` 从 `.gitignore` 移出（改为只忽略大的 `ablation.rankings.json`）。
请在本轮提交里一并 `git add` 上 `eval/out/ablation.md` 与 `eval/out/ablation.json`——
否则"实验文档的数字无法从提交物复现"这个问题依然存在（你在上轮报告里自己指出的）。

---

## 修完之后

1. 重新编译数据集（若 `lang` 字段与 S4/S6/S7 的改动涉及编译期）：`uv run python scripts/build_eval_dataset.py`
2. 重跑消融：`uv run note-rag eval --dataset eval/dataset.jsonl --out eval/out/ablation.md`
3. 跑四条门禁：`uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy` / `uv run pytest`

## 必须产出的报告

新建 `docs/reviews/impl-004-calibration-batch-b.md`，包含：

1. 每条的改法（文件:行号 + 理由）
2. **四条门禁的真实输出**（跑过的，不是"应当通过"）
3. **改动前后的指标对照**，并明确指出：哪些差异来自"只用可回答问题统计"、哪些来自语言切片、
   哪些来自延迟重复测量
4. **拒答表**：阈值、依据、`abstain_rate_on_unanswerable`、以及同阈值下的 `recall@5_on_answerable`
5. **延迟表**：每模式重复次数、中位数、离散度、以及"哪些模式间差异不可区分"
6. 仍然没修的问题清单 + 下一步（哪些该由 DSH 的 `workflow` 扇出做，哪些该交给嵌入模型那一步）

## 硬性约束

- 不要让测试为了通过而放宽断言；不要 skip 测试。
- 保持 `mypy --strict` 零错误；不引入新依赖。
- 报告里每个数字都要来自你**实际执行**的命令输出。
- 如果发现我这份任务书里有错误或不可行的要求，**直接指出并说明理由**，照做不如做对。
