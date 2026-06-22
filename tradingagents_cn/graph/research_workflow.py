"""A 股研究工作流。

这个文件是正式工程里的 LangGraph 多节点工作流。

当前节点：

    market
      生成 Market Agent 上下文。

    news
      生成 News Agent 上下文。

    fundamentals
      生成 Fundamentals Agent 上下文。

注意：
    当前工作流先不调用大模型。
    它只把三个 Agent 的 Prompt/context 串起来。

为什么这样做？

    原版 TradingAgents 是多 Agent 图。
    但是如果一上来就把取数、Prompt、LLM、路由、风控全部塞进去，
    会非常难调试。

    所以这里先做正式状态流转：
        数据输入
          -> Market Agent Prompt
          -> News Agent Prompt
          -> Fundamentals Agent Prompt
          -> 完整研究上下文
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from tradingagents_cn.agents import (
    build_fundamentals_agent_context,
    build_market_agent_context,
    build_news_agent_context,
)
from tradingagents_cn.dataflows.realtime_quote import RealtimeQuote
from tradingagents_cn.dataflows.stock_news import StockNewsItem
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


class ResearchWorkflowState(TypedDict, total=False):
    """A 股研究工作流状态。

    symbol:
        股票代码。

    trade_date:
        分析日期。

    indicator_data:
        已经计算好技术指标的历史行情表。

    realtime_quote:
        可选实时/近实时行情。

    stock_news_items:
        个股新闻列表。

    macro_news_text:
        可选宏观新闻文本。

    company_profile_text:
        公司基本资料文本。

    balance_sheet_text:
        资产负债表文本。

    cashflow_text:
        现金流量表文本。

    income_statement_text:
        利润表文本。

    market_prompt / news_prompt / fundamentals_prompt:
        三类 Agent 的 Prompt。

    market_snapshot / news_materials / fundamentals_materials:
        方便调试查看的中间材料。

    data_errors:
        数据准备阶段的非致命错误。
        例如实时行情、新闻或基本面某个公开接口临时失败。
    """

    symbol: str
    trade_date: str
    indicator_data: pd.DataFrame
    realtime_quote: RealtimeQuote | None
    stock_news_items: list[StockNewsItem]
    macro_news_text: str | None
    company_profile_text: str | None
    balance_sheet_text: str | None
    cashflow_text: str | None
    income_statement_text: str | None
    market_prompt: str
    news_prompt: str
    fundamentals_prompt: str
    market_snapshot: str
    news_materials: str
    fundamentals_materials: str
    data_errors: list[str]


def build_market_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Market 节点。

    输入：
        indicator_data
        realtime_quote

    输出：
        market_prompt
        market_snapshot
    """
    context = build_market_agent_context(
        symbol=state["symbol"],
        trade_date=state["trade_date"],
        indicator_data=state["indicator_data"],
        realtime_quote=state.get("realtime_quote"),
    )

    return {
        **state,
        "symbol": normalize_cn_symbol(state["symbol"]),
        "market_prompt": context.prompt,
        "market_snapshot": context.verified_snapshot,
    }


def build_news_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """News 节点。

    输入：
        stock_news_items
        macro_news_text

    输出：
        news_prompt
        news_materials
    """
    context = build_news_agent_context(
        symbol=state["symbol"],
        trade_date=state["trade_date"],
        stock_news_items=state.get("stock_news_items", []),
        macro_news_text=state.get("macro_news_text"),
    )

    return {
        **state,
        "news_prompt": context.prompt,
        "news_materials": context.stock_news_text,
    }


def build_fundamentals_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Fundamentals 节点。

    输入：
        company_profile_text
        balance_sheet_text
        cashflow_text
        income_statement_text

    输出：
        fundamentals_prompt
        fundamentals_materials
    """
    context = build_fundamentals_agent_context(
        symbol=state["symbol"],
        trade_date=state["trade_date"],
        company_profile_text=state.get("company_profile_text"),
        balance_sheet_text=state.get("balance_sheet_text"),
        cashflow_text=state.get("cashflow_text"),
        income_statement_text=state.get("income_statement_text"),
    )

    materials = "\n\n".join(
        [
            context.company_profile_text or "暂未提供公司基本资料。",
            context.balance_sheet_text or "暂未提供资产负债表材料。",
            context.cashflow_text or "暂未提供现金流量表材料。",
            context.income_statement_text or "暂未提供利润表材料。",
        ]
    )

    return {
        **state,
        "fundamentals_prompt": context.prompt,
        "fundamentals_materials": materials,
    }


def build_research_workflow():
    """构建正式 A 股研究工作流。

    当前执行顺序：

        market -> news -> fundamentals -> END

    后续如果要贴近原版 TradingAgents，
    可以继续加入：
        - sentiment；
        - bull researcher；
        - bear researcher；
        - research manager；
        - trader；
        - risk debators；
        - portfolio manager。
    """
    workflow = StateGraph(ResearchWorkflowState)

    workflow.add_node("market", build_market_node)
    workflow.add_node("news", build_news_node)
    workflow.add_node("fundamentals", build_fundamentals_node)

    workflow.set_entry_point("market")
    workflow.add_edge("market", "news")
    workflow.add_edge("news", "fundamentals")
    workflow.add_edge("fundamentals", END)

    return workflow.compile()
