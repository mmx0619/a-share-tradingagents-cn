"""第 20 步：把实时行情接入共享状态工作流。

第 18 步已经有状态版完整多 Agent 工作流。
第 19 步补上了实时行情快照。

当前文件做第 20 件事：

创建 TradingAgentState
  ↓
实时行情节点写入 realtime_quote_text
  ↓
市场分析节点写入 market_report
  ↓
新闻节点写入 news_report
  ↓
综合节点写入 summary_report
  ↓
风控节点写入 risk_report
  ↓
交易员节点写入 trader_plan

这一步仍然保持简单：
实时行情先进入 state，
但暂时不强行改写每个 Agent 的 Prompt。

原因：
先让数据进入状态，
下一步再决定哪些 Agent 应该读取实时行情。
这更接近工程开发里的渐进式改造。
"""

from __future__ import annotations

import os

import step17_agent_state as agent_state
import step18_stateful_workflow as stateful_workflow
import step19_realtime_quote as realtime_quote


def run_realtime_quote_node(state: agent_state.TradingAgentState) -> None:
    """实时行情节点。

    读取：
    - state.symbol

    写入：
    - state.realtime_quote_text

    这个节点不调用大模型，只负责采集实时/近实时行情快照。
    """
    quote = realtime_quote.get_realtime_quote(state.symbol)
    state.realtime_quote_text = realtime_quote.render_realtime_quote_text(quote)


def run_realtime_stateful_workflow(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> agent_state.TradingAgentState:
    """运行带实时行情的状态版完整工作流。

    和第 18 步相比，多了一个实时行情节点：

    realtime_quote_node
      ↓
    market_node
      ↓
    news_node
      ↓
    summary_node
      ↓
    risk_node
      ↓
    trader_node
    """
    state = agent_state.TradingAgentState(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        model=model,
    )

    run_realtime_quote_node(state)
    stateful_workflow.run_market_node(state, provider=provider, model=model)
    stateful_workflow.run_news_node(
        state,
        provider=provider,
        model=model,
        news_max_items=news_max_items,
    )
    stateful_workflow.run_summary_node(state, provider=provider, model=model)
    stateful_workflow.run_risk_node(state, provider=provider, model=model)
    stateful_workflow.run_trader_node(state, provider=provider, model=model)

    return state


def render_realtime_state_report(state: agent_state.TradingAgentState) -> str:
    """把带实时行情的完整状态渲染成报告。"""
    return f"""{agent_state.render_state_summary(state)}

======== 实时行情快照 ========
{state.realtime_quote_text}

======== 日线市场快照 ========
{state.market_snapshot_text}

======== 新闻事件信号 ========
{state.news_events_text}

======== 市场分析师报告 ========
{state.market_report}

======== 新闻 Agent 报告 ========
{state.news_report}

======== 综合汇总报告 ========
{state.summary_report}

======== 风控报告 ========
{state.risk_report}

======== 交易员预案 ========
{state.trader_plan}
"""


def demo_realtime_stateful_workflow() -> None:
    """演示带实时行情的状态版完整工作流。"""
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    state = run_realtime_stateful_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_realtime_state_report(state))


if __name__ == "__main__":
    demo_realtime_stateful_workflow()
