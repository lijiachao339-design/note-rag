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

## 已经踩过的坑（别再踩第二次）

1. **MCP 2.x 改名**：`mcp` 2.x 把 `FastMCP` 改名为 `MCPServer`（`mcp.server.mcpserver`）。
   照抄 v1 示例会 `ModuleNotFoundError`。改完 MCP 代码必须跑一次导入冒烟。
2. **API 返回空命中**：`AppState` 曾漏调 `build_keyword_index()`，导致 `/search` 返回的
   `note_path`/`title`/`text` 全是空字符串，而 `/healthz`、`/metrics` 一切正常。
   已加 `tests/test_retriever.py` 回归测试，并在 lifespan 里启动即建索引。
3. **评分函数的测试写错**：RRF 里"某文档在所有列表都排第 1"时，改权重不会改它的分数
   （只改它与其它文档的差距）。写这类断言前先把公式算一遍。
4. **只有标题没有正文的笔记产出 0 个切块**——这是正确行为（标题属于元数据），
   但写测试时别拿 `"# ok"` 当"可读文件"的样例。
5. **语料目录会让 lint 膨胀**：`data/corpus` 有上千个文件，而仓库尚未 `git init` 时
   `.gitignore` 不生效，`ruff` 会从 26 个文件扫到 2000 个。`pyproject.toml` 里的
   `extend-exclude = ["data", ...]` 必须保留。

## 有价值的下一步（按优先级）

0. **补集成测试**：`api` / `cli` / `mcp_server` / `retriever` 的 DB 路径覆盖率仍是 0%，
   总覆盖率约 48%。CI 里已经有 pg service，把它用起来跑真库集成测试（并断言 `/search` 的命中
   必须带非空 `note_path` 与 `text`——这正是第 2 条坑的防线）。
1. 接入真实 reranker 并产出 `hybrid` vs `hybrid_rerank` 的评测对比
2. 把 embedding 缓存从进程内 dict 迁到 Redis，并统计多副本下的命中率
3. 关键词索引改为增量刷新（当前每次启动全量构建，17520 chunks 约 1 s，暂时可接受）
4. 中文分词实验：unigram+bigram（现状）vs `pg_bigm` vs `zhparser`
5. 小节合并策略消融：跨同级标题共享父级 breadcrumb 是否提升 recall
6. 换嵌入模型时 `vector(dim)` 是建表时固定的：`ensure_schema` 目前不会检测维度变化，
   维度不一致会在插入时报错。应显式检测并给出可操作的报错（或自动重建表）
