#!/usr/bin/env bash
# note-rag 开发快捷入口（Git Bash / WSL / Linux / macOS 通用）
#
# 用法:
#   scripts/dev.sh setup     起数据库 + 装依赖 + 生成 .env
#   scripts/dev.sh db        只起 pg/redis
#   scripts/dev.sh dry       干跑切块（不连库）
#   scripts/dev.sh ingest    增量索引
#   scripts/dev.sh search 查询词
#   scripts/dev.sh test      跑测试 + 覆盖率
#   scripts/dev.sh lint      ruff + mypy
#   scripts/dev.sh fmt       自动格式化与修复
#   scripts/dev.sh serve     起 HTTP API（热重载）
#   scripts/dev.sh mcp       起 MCP stdio server
#   scripts/dev.sh eval      跑四模式消融实验
set -euo pipefail

cd "$(dirname "$0")/.."

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "缺少 $1。安装方式：$2" >&2
        exit 1
    }
}

usage() {
    sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    setup)
        need docker "安装 Docker Desktop"
        need uv "winget install --id=astral-sh.uv -e"
        docker compose up -d pg redis
        uv sync --all-groups
        [ -f .env ] || cp .env.example .env
        echo "已就绪。请编辑 .env 里的 NOTE_RAG_VAULT_PATH 指向你的 Obsidian Vault"
        ;;
    db)
        need docker "安装 Docker Desktop"
        docker compose up -d pg redis
        ;;
    db-stop)
        docker compose stop pg redis
        ;;
    dry)
        uv run note-rag ingest --dry-run --preview 5
        ;;
    ingest)
        uv run note-rag ingest
        ;;
    search)
        uv run note-rag search "$@"
        ;;
    test)
        uv run pytest --cov=src/note_rag --cov-report=term-missing
        ;;
    lint)
        uv run ruff check . && uv run ruff format --check . && uv run mypy
        ;;
    fmt)
        uv run ruff format . && uv run ruff check --fix .
        ;;
    serve)
        uv run note-rag serve --reload
        ;;
    mcp)
        uv run note-rag mcp
        ;;
    eval)
        uv run note-rag eval --dataset eval/dataset.jsonl --out eval/out/ablation.md
        ;;
    selfcheck)
        # 没有 pytest / 没有数据库时也能验证核心逻辑
        python .selfcheck.py
        ;;
    help|*)
        usage
        ;;
esac
