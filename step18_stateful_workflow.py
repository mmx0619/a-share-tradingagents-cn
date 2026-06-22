"""第 18 步：用共享状态串起完整多 Agent 工作流。

第 17 步定义了 TradingAgentState。
它只是“状态容器”，还没有真正跑完整流程。

当前文件做第 18 件事：

创建 TradingAgentState
  ↓
市场分析师 Agent 写入 market_report
  ↓
新闻 Agent 写入 news_report
  ↓
综合 Agent 写入 summary_report
  ↓
风控 Agent 写入 risk_report
  ↓
交易员 Agent 写入 trader_plan
  ↓
输出完整 state 摘要

这一步非常接近 LangGraph 的思想：

每个节点不再只是返回一个字符串，
而是读取 state，并把自己的结果写回 state。

后面真正接 LangGraph 时，
这些函数就可以改造成 LangGraph 节点。
"""

from __future__ import annotations

import os

import step10_real_market_agent_pipeline as market_pipeline
import step11_stock_news as stock_news
import step12_news_event_extractor as event_extractor
import step13_news_agent as news_agent
import step14_research_summary_agent as summary_agent
import step15_risk_control_agent as risk_agent
import step16_trader_agent as trader_agent
import step17_agent_state as agent_state


def run_market_node(
    state: agent_state.TradingAgentState,
    provider: str,
    model: str | None,
) -> None:
    """市场分析节点。

    读取：
    - state.symbol
    - state.start_date
    - state.end_date

    写入：
    - state.market_snapshot_text
    - state.market_report
    """
    result = market_pipeline.run_real_market_agent_pipeline(
        symbol=state.symbol,
        start_date=state.start_date,
        end_date=state.end_date,
        provider=provider,
        model=model,
    )
    state.market_snapshot_text = result.market_snapshot_text
    state.set_market_report(
        report=result.report_text,
        provider=result.provider,
        model=result.model,
    )


def run_news_node(
    state: agent_state.TradingAgentState,
    provider: str,
    model: str | None,
    news_max_items: int,
) -> None:
    """新闻分析节点。

    读取：
    - state.symbol

    写入：
    - state.news_events_text
    - state.news_report
    """
    news_items = stock_news.get_stock_news(state.symbol, max_items=news_max_items)
    news_events = event_extractor.extract_news_events(news_items)
    state.news_events_text = event_extractor.render_news_events_text(news_events)

    result = news_agent.run_news_agent(
        symbol=state.symbol,
        news_events=news_events,
        provider=provider,
        model=model,
    )
    state.set_news_report(
        report=result.report_text,
        provider=result.provider,
        model=result.model,
    )


def run_summary_node(
    state: agent_state.TradingAgentState,
    provider: str,
    model: str | None,
) -> None:
    """综合汇总节点。

    读取：
    - state.market_report
    - state.news_report

    写入：
    - state.summary_report
    """
    if not state.market_report:
        raise ValueError("缺少 market_report，不能运行综合汇总节点。")
    if not state.news_report:
        raise ValueError("缺少 news_report，不能运行综合汇总节点。")

    result = summary_agent.run_research_summary_agent(
        symbol=state.symbol,
        market_report=state.market_report,
        news_report=state.news_report,
        provider=provider,
        model=model,
    )
    state.set_summary_report(
        report=result.summary_text,
        provider=result.provider,
        model=result.model,
    )


def run_risk_node(
    state: agent_state.TradingAgentState,
    provider: str,
    model: str | None,
) -> None:
    """风控节点。

    读取：
    - state.summary_report

    写入：
    - state.risk_report
    """
    if not state.summary_report:
        raise ValueError("缺少 summary_report，不能运行风控节点。")

    result = risk_agent.run_risk_control_agent(
        symbol=state.symbol,
        research_summary=state.summary_report,
        provider=provider,
        model=model,
    )
    state.set_risk_report(
        report=result.risk_report,
        provider=result.provider,
        model=result.model,
    )


def run_trader_node(
    state: agent_state.TradingAgentState,
    provider: str,
    model: str | None,
) -> None:
    """交易员节点。

    读取：
    - state.risk_report

    写入：
    - state.trader_plan
    """
    if not state.risk_report:
        raise ValueError("缺少 risk_report，不能运行交易员节点。")

    result = trader_agent.run_trader_agent(
        symbol=state.symbol,
        risk_report=state.risk_report,
        provider=provider,
        model=model,
    )
    state.set_trader_plan(
        plan=result.trader_plan,
        provider=result.provider,
        model=result.model,
    )


def run_stateful_workflow(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> agent_state.TradingAgentState:
    """运行状态版完整多 Agent 工作流。

    和前面的完整链路相比，这个函数最大的区别是：
    每一步都写入同一个 TradingAgentState。

    返回的 state 里，保留了整个流程的关键中间结果。
    """
    state = agent_state.TradingAgentState(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        model=model,
    )

    run_market_node(state, provider=provider, model=model)
    run_news_node(state, provider=provider, model=model, news_max_items=news_max_items)
    run_summary_node(state, provider=provider, model=model)
    run_risk_node(state, provider=provider, model=model)
    run_trader_node(state, provider=provider, model=model)

    return state


def render_full_state_report(state: agent_state.TradingAgentState) -> str:
    """把完整状态渲染成方便阅读的报告。"""
    return f"""{agent_state.render_state_summary(state)}

======== 市场快照 ========
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


def demo_stateful_workflow() -> None:
    """演示状态版完整工作流。

    默认继续使用 002361。
    这个股票风险比较高，适合观察各个 Agent 如何写入 state。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    state = run_stateful_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_full_state_report(state))


if __name__ == "__main__":
    demo_stateful_workflow()
