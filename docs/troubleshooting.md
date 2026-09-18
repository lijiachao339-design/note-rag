# 疑难排障

本文件收集在 Windows 上跑这个项目时踩过的真实问题。每条都写明**现象 / 原因 / 验证方式 / 修复**，
因为"照抄一段命令"通常解决不了环境问题——你需要知道它为什么坏。

---

## 1. Git Bash 里 `claude`（或任何 npm 安装的 CLI）启动失败

**现象**

```bash
$ claude --version
/c/Users/<user>/AppData/Roaming/npm/claude: line 12:
D:\Anaconda3\Library\c\Users\<user>\AppData\Roaming\npm/node_modules/.../claude.exe:
No such file or directory
```

注意路径里多出来的 `D:\Anaconda3\Library\` —— 这是最关键的线索。

**原因链**（每一步都可自行验证）

1. npm 生成的 `claude` 是一个 POSIX shim，内部用 `cygpath -w` 把 POSIX 路径换算成 Windows 路径；
2. 如果 `~/.bash_profile` 里有 conda init，Anaconda 会把 `.../Anaconda3/Library/usr/bin` 插到 PATH 的很前面；
3. Anaconda **自带一个 `cygpath`**，于是 shim 调到的不是 Git 的，而是 Anaconda 的；
4. Anaconda 的 `cygpath` 以自己的根目录为基准换算 `/c/Users/...`，得到 `D:\Anaconda3\Library\c\Users\...`；
5. 那个路径当然不存在 → `No such file or directory`。

**验证**

```bash
# 登录 shell（双击打开 Git Bash 的默认方式）
bash -lc 'command -v cygpath'
# 非登录 shell（脚本方式）
bash -c  'command -v cygpath'
```

两者输出不同（前者指向 Anaconda，后者指向 `/usr/bin`）就确认是这个原因。

**修复（推荐第 1 个）**

```bash
# 1) 在 ~/.bash_profile 的 conda 块【之后】追加，把 Git 的 /usr/bin 抢回最前
echo 'export PATH="/usr/bin:$PATH"' >> ~/.bash_profile

# 2) 只影响单次调用
PATH="/usr/bin:$PATH" claude

# 3) 不在 Git Bash 里用 conda（如果用 uv 管理 Python，conda 在这里价值不大）
#    用编辑器把 ~/.bash_profile 里的 conda init 那几行注释掉
```

**顺带影响**：修好之前，登录 shell 里裸 `python` 会解析到 Anaconda 的 Python。
所以本项目一律用 `uv run ...`，不要用裸 `python`，否则解释器会随 shell 漂移。

---

## 2. MSYS 会改写传给原生程序的绝对路径

**现象**：`docker run -v /var/lib/postgresql/data ...` 里的路径被悄悄换成
`C:/Program Files/Git/var/lib/postgresql/data`，容器于是挂载了一个不存在的目录。

**验证**

```bash
python -c 'import sys; print(sys.argv[1:])' /var/lib/postgresql/data
# ['C:/Program Files/Git/var/lib/postgresql/data']   ← 被改写了

MSYS_NO_PATHCONV=1 python -c 'import sys; print(sys.argv[1:])' /var/lib/postgresql/data
# ['/var/lib/postgresql/data']                       ← 原样传递
```

**修复**：加 `MSYS_NO_PATHCONV=1`，或写成双斜杠 `//var/lib/postgresql/data`。

---

## 3. PowerShell 5.1 的编码会吃掉中文

Windows 自带的 `powershell.exe` 是 5.1，它有几个和 UTF-8 有关的行为需要注意：

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| 脚本里的中文输出是乱码（如 `鐢ㄦ硶`） | 5.1 用 ANSI(GBK) 解码**没有 BOM** 的 UTF-8 脚本文件 | 把 `.ps1` 存成 **UTF-8 with BOM** |
| Python 脚本的中文输出乱码 | 控制台代码页(GBK) 与 Python 的 UTF-8 stdout 不匹配 | `chcp 65001`，或 `[Console]::OutputEncoding=[Text.Encoding]::UTF8` |
| `Get-Content` 读 UTF-8 文件出现乱码且行粘连 | 同上，按 GBK 解码导致多字节序列吞掉了 `\r` | `Get-Content -Encoding UTF8` |

---

## 4. PowerShell 5.1 的语法差异

写脚本时注意，5.1 上这些会直接报错：

| 写法 | 5.1 支持 | 替代 |
| --- | --- | --- |
| `$a ?? $b`（空合并） | ❌ | `if ($null -ne $a) { $a } else { $b }` |
| `$a?.b`（空条件访问） | ❌ | 显式判空 |
| `Ternary`: `$c ? 'a' : 'b'` | ❌ | `if/else` |

另一个容易踩的坑：**把中文全角引号 `“ ”` 写进双引号字符串里会被当成字符串定界符**，
导致 `Unexpected token` 解析错误。脚本里请只用 ASCII 引号。

---

## 5. `.gitattributes` 与 `core.autocrlf`

Windows 上 `git config core.autocrlf` 常被设为 `true`，这会让 checkout 时把 LF 换成 CRLF。对混合 shell 的项目这会直接损坏可执行文件：

- `*.sh` 带 CRLF → bash 报 `bad interpreter: /bin/sh^M` 或 `$'\r': command not found`
- `Dockerfile` 里 `\` 续行被 CRLF 打断 → 构建报 `/bin/sh: \r: not found`

本仓库用 `.gitattributes` 固定：默认 `eol=lf`，只有 `*.ps1 / *.cmd / *.bat` 用 CRLF。
如果 clone 后才发现，执行一次 `git add --renormalize .` 让规则生效。

---

## 6. `mcp` 2.x 的 API 改名

`mcp` 2.x 把 `FastMCP` 改名为 `MCPServer`：

```python
# ❌ v1 写法，在 mcp>=2 上会 ModuleNotFoundError
from mcp.server.fastmcp import FastMCP

# ✅ v2 写法
from mcp.server.mcpserver import MCPServer
```

照抄旧教程遇到 `No module named 'mcp.server.fastmcp'` 时，先确认装的是哪个大版本。
