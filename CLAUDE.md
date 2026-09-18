# CLAUDE.md — note-rag

> Claude Code 项目配置。与 `AGENTS.md` 内容一致；**两者必须同步修改**，否则不同 agent 的行为会漂移。

## 项目一句话

个人 Obsidian 知识库的混合检索服务：pgvector 向量检索 + 进程内 BM25 + RRF 融合，暴露为
FastAPI 接口和 MCP Server。这是求职作品集项目，**可解释性 > 功能数量**。

## 环境

- Python **3.12**（本机默认 3.14 轮子不全），一律用 `uv`
- Postgres 16 + pgvector、Redis 7：`docker compose up -d pg redis`
- 配置来自 `.env`（模板见 `.env.example`），前缀 `NOTE_RAG_`

## 常用命令

```powershell
uv sync --all-groups                    # 装依赖
uv run note-rag ingest --dry-run        # 只看切块，不连库
uv run note-rag ingest                  # 增量索引
uv run note-rag search "查询词"          # 命令行检索
uv run note-rag eval                    # 四模式消融，输出 markdown 表
uv run note-rag serve --reload           # HTTP API
uv run pytest                           # 测试
uv run ruff check . ; uv run mypy       # 质量门禁
```

## 工作流要求

- **plan mode 先行**：涉及 `retriever.py` / `ingest.py` / `store.py` 的结构性改动，
  先给出方案与取舍，等确认后再写代码。
- 改动检索链路后，必须用 `uv run note-rag eval` 给出前后指标对比。
- 每个实现步骤配一个测试；测试文件的组织方式参考现有 `tests/`。

## 关键设计约束（不要"顺手优化"掉）

| 位置 | 约束 | 原因 |
| --- | --- | --- |
| `notes.py` | chunk id 由 `(note_path, heading_path, char_start)` 派生 | 保证编辑后 id 稳定，增量索引才成立 |
| `notes.py` | 合并切块时不得跨越标题边界 | 跨标题合并会丢失 breadcrumb 并让整篇笔记变成一个 chunk |
| `metrics.py` | 无相关文档的查询记 0 分但保留在均值中 | 否则"拒答"策略会显得指标虚高 |
| `fusion.py` | 排序必须确定性（tie-break 到 id） | 否则评测数字不可复现 |
| `retriever.py` | 四种模式共用同一实现 | 评测与线上走不同代码路径则数字失真 |
| `mcp_server.py` | 文件路径必须校验是否逃出 vault | 模型可以传入 `../../` 之类的路径 |

## 禁止

- 引入 LangChain / LlamaIndex 等重型框架
- 提交 `.env`、真实 vault 内容、真实 API key
- 为了让测试通过而放宽断言或 skip 测试
- 在没有评测数据支撑时修改 chunk 大小 / RRF k / 检索权重
