# AGENTS.md — 项目约定（供 Claude Code / DSH 等编码 agent 读取）

本文件是 agent 进入本仓库的第一入口。请严格遵守；与用户口头指令冲突时以用户为准。

## 项目定位

面向个人 Obsidian 知识库的混合检索服务（向量 + BM25 + RRF），对外暴露 HTTP API 与
MCP Server。这是求职作品集项目，**代码质量与可解释性优先于功能数量**。

## 硬性规则

1. **Python 版本固定 3.12**（本机默认的 3.14 缺很多轮子）。一律用 `uv`，不要用全局 pip 装包。
2. **先测试后实现**：改动 `notes.py` / `fusion.py` / `metrics.py` / `bm25.py` / `ingest.py`
   前，先在 `tests/` 里加一个会失败的测试。
3. **不允许静默失败**：解析、检索、嵌入的错误必须被记录（`SyncStats.failures`）或抛出，
   禁止 `except: pass`。
4. **确定性**：任何排序结果必须可复现（相同输入 → 相同输出）。禁止依赖 `set` 迭代顺序。
5. **检索指标口径不得修改**：`metrics.py` 中"无相关文档时 recall 记 0 但保留在均值里"是刻意设计，
   改它等于篡改评测结果。
6. **改动检索链路后必须重跑评测**并在 PR 描述里贴出前后数字：`uv run note-rag eval`。

## 完成前必须通过

```powershell
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run note-rag ingest --dry-run
```

## 代码风格

- 类型注解完整，`mypy --strict` 零错误；公共函数写 docstring，说明**为什么**这样设计，
  而不是复述代码在做什么。
- 行长 100，`ruff` 自动格式化。
- 注释用中文或英文均可，但**解释取舍**：例如"为什么用 RRF 而不是加权求和"。
- 不引入新依赖除非必要；能用标准库就不加库。新增依赖必须同时更新 `pyproject.toml`。

## 禁止事项

- 不要把真实 vault 内容、真实 API key、`.env` 提交进仓库。
- 不要为了让测试过而放宽断言或加 `pytest.mark.skip`。
- 不要引入 LangChain / LlamaIndex 之类的重型框架封装检索链路（面试需要能逐行讲清）。
- 不要在没有评测数据支撑的情况下修改默认 chunk 大小、RRF k 值或检索权重。

## 有价值的下一步（按优先级）

1. 接入真实 reranker 并产出 `hybrid` vs `hybrid_rerank` 的评测对比
2. 把 embedding 缓存从进程内 dict 迁到 Redis，并统计多副本下的命中率
3. 关键词索引改为增量刷新（当前每次启动全量构建）
4. 中文分词实验：unigram+bigram（现状）vs `pg_bigm` vs `zhparser`
5. 小节合并策略消融：跨同级标题共享父级 breadcrumb 是否提升 recall
