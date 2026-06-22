# tradingagents_cn

这是 A 股版 TradingAgents 的正式工程代码目录。

当前已经包含：

```text
LangGraph StateGraph
ToolNode
Market / News / Fundamentals 三大 Analyst Tool Calling
多模型 LLM factory
A 股行情、新闻、基本面、情绪工具
SQLite 持久化 checkpoint / resume
最终报告生成
交易记忆层
```

项目完整使用说明请优先阅读根目录文档：

```text
使用手册与断点续跑说明.md
```

开发同步和历史记录请阅读：

```text
项目目标与开发日志.md
```

## 目录说明

```text
dataflows/   数据采集层
indicators/  指标层
agents/      多 Agent 分析层
prompts/     Prompt 层
graph/       LangGraph 工作流层
tools/       ToolNode 工具层
llm/         大模型调用层
memory/      记忆层
examples/    可运行示例
```

## 当前开发原则

```text
正式工程优先维护 tradingagents_cn/。
旧 step 文件主要用于学习记录，不再作为主开发入口。
每次重要改动后需要更新项目目标与开发日志。
用户手册类说明写入 使用手册与断点续跑说明.md。
```
