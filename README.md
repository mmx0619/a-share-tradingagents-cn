# A 股版 TradingAgents

这是一个从 0 搭建的 A 股版 TradingAgents 多智能体投研系统。

目标不是简单复制原项目，而是保留 TradingAgents 的核心思想：

```text
LangGraph
ToolNode
Tool Calling
多 Agent 分工
多空辩论
风控辩论
Portfolio Manager 最终决策
结构化输出
记忆层
checkpoint / resume
Reflector 事后收益复盘
A 股 benchmark 超额收益
数据质量校验
工具调用统计
模拟盘自动交易和复盘
无人值守自动模拟交易守护进程
```

并把数据源和提示词改成 A 股语境。

## 当前能力

```text
自然语言问题入口
A 股股票识别和路由
Market / Sentiment / News / Fundamentals Analyst 主动 Tool Calling
公告、新闻、情绪、行情、基本面工具
selected_analysts 动态构图
Summary Agent
Bull / Bear 独立节点化结构化多空辩论
Research Manager
Trader
Aggressive / Conservative / Neutral 独立节点化结构化风控辩论
Portfolio Manager
程序风控护栏和仓位上限
最终 Markdown 报告
SQLite 持久化 checkpoint / resume / 历史任务列表
新闻/公告/情绪/基本面缓存
多模型 LLM factory
data_vendors / tool_vendors 配置入口与真实 vendor_router 路由
TradingAgentsCNGraph 统一类
GraphSetup / Propagator / Reflector 组件化
full_state.json 完整运行状态日志
full_state.json 工具调用轨迹、数据质量提示、反思摘要
paper_trading 本地模拟账户和模拟成交记录
auto_trader 自动候选扫描、循环分析、模拟盘执行日志
沪深主板普通 A 股自动买入硬门禁
CLI 参数化入口
```

## 快速开始

进入项目目录：

```powershell
cd C:\Users\zl\Desktop\股票\A股慢改从0开始
```

安装依赖：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe -m pip install -r requirements.txt
```

运行测试：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe -m unittest discover -s tests
```

运行自然语言入口：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买"
```

也可以安装成可编辑包：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe -m pip install -e .
```

然后运行：

```powershell
tradingagents-cn --question "京东方A能不能买"
```

或者：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe -m tradingagents_cn.cli --question "京东方A能不能买"
```

## 常用命令

查看帮助：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --help
```

指定模型：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "大唐发电行情怎么样" --provider deepseek --model deepseek-chat
```

默认模型兜底顺序：

```text
不传 --provider 时：
DeepSeek -> Gemini

也就是先调用 DeepSeek。
如果 DeepSeek 因余额、限流、网络或服务错误失败，本轮进程会自动切到 Gemini。
```

只用 Gemini：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "大唐发电行情怎么样" --provider gemini --model gemini-2.5-flash
```

指定 Analyst：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买" --analysts market,sentiment,news,fundamentals
```

指定情绪源：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买" --sentiment-sources eastmoney,xueqiu
```

指定数据源 vendor：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买" --tool-vendor realtime_quote=sina --tool-vendor daily_history=tencent --tool-vendor announcements=cninfo
```

说明：

```text
realtime_quote 支持 auto / akshare / eastmoney / sina
daily_history 支持 auto / akshare / eastmoney / tencent
stock_news 支持 auto / akshare / eastmoney
announcements 支持 auto / akshare / cninfo / eastmoney
sentiment 支持 auto / public_web / eastmoney / xueqiu / tonghuashun / taoguba
fundamentals、balance_sheet、cashflow、income_statement 当前主要走 akshare
announcements 是命令行友好别名，程序内部会映射到 announcement_tool
```

指定记忆复盘 benchmark：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买" --benchmark-symbol 000905 --benchmark-name 中证500 --memory-holding-days 10
```

说明：

```text
默认 benchmark 是 000300 / 沪深300。
pending 交易记忆会在未来运行时尝试计算个股收益和相对 benchmark 的超额收益。
成功复盘后，Reflector 会把反思写回 outputs/memory/trading_memory.md。
```

启用模拟盘自动交易：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "京东方A能不能买" --paper-trading
```

说明：

```text
模拟盘只写本地 JSON，不连接券商，不真实下单。
默认账户文件：outputs/paper_trading/account.json
执行逻辑：Portfolio Manager 最终评级 -> TradeSignal -> 风控护栏 -> 模拟买卖。
BUY 会按风控仓位上限和 100 股整数倍生成模拟买单。
BUY 前还有一层硬规则：当前只允许沪深主板普通 A 股自动买入。
创业板、科创板、北交所、ST、退市风险股会直接 skipped，模型给 BUY 也不能绕过。
SELL 只会卖出模拟账户已有持仓，不做融券卖空。
HOLD 或风控阻断会记录 skipped，不会买卖。
```

自定义模拟盘资金和仓位：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --question "大唐发电行情怎么样" --paper-trading --paper-initial-cash 200000 --paper-max-position-pct 0.15 --paper-min-trade-amount 2000
```

自动模拟交易先跑一轮：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_auto_paper_trader.py --allow-after-hours-paper-trading
```

长期循环运行：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_auto_paper_trader.py --loop --scan-interval-seconds 300 --max-candidates 3
```

使用自选股文件：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_auto_paper_trader.py --loop --watchlist outputs/auto_paper_trader/watchlist.txt --max-candidates 5
```

说明：

```text
run_auto_paper_trader.py 会先用行情规则筛候选，再对候选运行完整 TradingAgents 分析。
自动买入候选只保留沪深主板普通 A 股；已有持仓仍会继续复查，方便后续卖出或风控处理。
默认只在 A 股交易时段写入模拟成交。
如果需要非交易时段也按最近价格模拟成交，添加 --allow-after-hours-paper-trading。
自动交易周期日志默认保存到 outputs/auto_paper_trader。
```

断点续跑：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --resume --thread-id single-stock-000725-2026-06-18
```

列出 checkpoint：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --list-checkpoints
```

列出报告：

```powershell
C:\software\Anaconda\envs\feishubot\python.exe .\run_user_question.py --list-reports
```

## 重要文档

详细使用说明：

```text
使用手册与断点续跑说明.md
```

开发历史和同步记忆：

```text
项目目标与开发日志.md
```

正式工程代码：

```text
tradingagents_cn/
```

旧的 step 文件主要用于学习和历史记录，不再作为主开发入口。
