# 审查 001：检索评测口径 —— 等待人工执行

**状态：待 Claude Code（Opus）执行。跑完后这个文件会被真实结论覆盖。**

## 为什么这里是空的

DSH 尝试**非交互**调用 Claude Code，被 API 拒绝：

```
$ claude auth status
  loggedIn: true      authMethod: claude.ai      subscriptionType: pro
$ "hi" | claude -p --model opus --allowedTools "Read,Glob,Grep"
  Failed to authenticate. API Error: 403 Request not allowed
$ "hi" | claude -p --model sonnet
  Failed to authenticate. API Error: 403 Request not allowed
```

**登录有效（Pro 订阅正常），但程序化/非交互请求被拒**。所以本项目采用
**文件式人工交接**：DSH 准备提示词与全部实测数据 → 你在 Claude Code 里执行 →
Opus 把结论写回这个文件 → DSH 按结论修。

## 你要做的三步

```powershell
cd D:\AI\projects\note-rag
claude
```

1. `Shift+Tab` 切到 **plan mode**；`/model` 选 **Opus**
2. 粘贴下面这一句（提示词已在仓库里，不用手打）：

```
读取 docs/reviews/prompt-001-eval-methodology.md，按里面的要求执行审查。
只出意见、不要改任何代码。完成后把结论原样写入 docs/reviews/review-001-eval-methodology.md。
```

3. 退出后回来告诉 DSH「审查完了」。

## 这次审查要回答的 6 个问题（详见提示词文件）

1. note 粒度去重（每篇取最佳 chunk）会低估还是高估 recall？
2. 不可回答问题记 0 分并留在均值里，是否会惩罚"合理返回空"的检索器？
3. `answer_span` 逐字校验能挡住哪些幻觉、挡不住哪些？
4. 135 英文 + 45 中文的语言不均衡会不会让某些模式失真？
5. 18 条负样本是否真的不可回答？（要求抽查 5 条并在 `data/corpus/` 里核实）
6. 用当前口径，能不能诚实地写"hybrid 把 recall@5 从 X 提升到 Y"？
