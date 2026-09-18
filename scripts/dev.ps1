#!/usr/bin/env pwsh
# note-rag 开发快捷入口（Windows PowerShell 5.1 / PowerShell 7 通用）
#
# 用法:
#   .\scripts\dev.ps1 setup     起数据库 + 装依赖 + 生成 .env
#   .\scripts\dev.ps1 db        只起 pg/redis
#   .\scripts\dev.ps1 dry       干跑切块（不连库）
#   .\scripts\dev.ps1 ingest    增量索引
#   .\scripts\dev.ps1 search 查询词
#   .\scripts\dev.ps1 test      跑测试 + 覆盖率
#   .\scripts\dev.ps1 lint      ruff + mypy
#   .\scripts\dev.ps1 fmt       自动格式化与修复
#   .\scripts\dev.ps1 serve     起 HTTP API（热重载）
#   .\scripts\dev.ps1 mcp       起 MCP stdio server
#   .\scripts\dev.ps1 eval      跑四模式消融实验

[CmdletBinding()]
param(
    [Parameter(Position = 0)][string] $Command = 'help',
    [Parameter(ValueFromRemainingArguments = $true)][string[]] $Rest
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Test-Tool {
    param([string] $Name, [string] $Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "缺少 $Name。安装方式：$Hint"
    }
}

# 用法文本内联而不是从自身文件读回来。
# 原因：Windows PowerShell 5.1 的 Get-Content 默认按 ANSI(GBK) 解码，
# 读 UTF-8 文件会得到乱码；内联 here-string 可以彻底绕开这个坑。
$Usage = @'
用法: .\scripts\dev.ps1 <命令>

  setup      起 pg/redis + 装依赖 + 生成 .env
  db         只起 pg/redis
  db-stop    停掉 pg/redis
  dry        干跑切块（不连数据库，先看质量）
  ingest     增量索引
  search     检索，例：search "混合检索怎么融合两路召回"
  test       跑测试 + 覆盖率
  lint       ruff + mypy
  fmt        自动格式化与修复
  serve      起 HTTP API（热重载）
  mcp        起 MCP stdio server
  eval       跑四模式消融实验
  selfcheck  无 pytest / 无数据库时的离线自检
'@

function Show-Usage {
    Write-Host $Usage
}

switch ($Command) {
    'setup' {
        Test-Tool docker '安装 Docker Desktop'
        Test-Tool uv 'winget install --id=astral-sh.uv -e'
        docker compose up -d pg redis
        uv sync --all-groups
        if (-not (Test-Path .env)) { Copy-Item .env.example .env }
        Write-Host '已就绪。请编辑 .env 里的 NOTE_RAG_VAULT_PATH 指向你的 Obsidian Vault'
    }
    'db' { docker compose up -d pg redis }
    'db-stop' { docker compose stop pg redis }
    'dry' { uv run note-rag ingest --dry-run --preview 5 }
    'ingest' { uv run note-rag ingest }
    'search' { uv run note-rag search @Rest }
    'test' { uv run pytest --cov=src/note_rag --cov-report=term-missing }
    'lint' {
        uv run ruff check .
        uv run ruff format --check .
        uv run mypy
    }
    'fmt' {
        uv run ruff format .
        uv run ruff check --fix .
    }
    'serve' { uv run note-rag serve --reload }
    'mcp' { uv run note-rag mcp }
    'eval' { uv run note-rag eval --dataset eval/dataset.jsonl --out eval/out/ablation.md }
    # 没有 pytest / 没有数据库时也能验证核心逻辑
    'selfcheck' { python .selfcheck.py }
    default { Show-Usage }
}
