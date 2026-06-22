"""A 股 AKShare 工具。

这个文件是正式工程里的工具层文件。

当前包含一个最小可用工具：

akshare_realtime_quote(symbol)

它的作用是：

1. 接收股票代码。
2. 调用正式数据层里的实时行情函数。
3. 返回适合大模型阅读的行情文本。
4. 可以被 LangGraph 的 ToolNode 执行。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.tools import tool

from tradingagents_cn.agents.market_agent import build_verified_market_snapshot_text
from tradingagents_cn.dataflows.realtime_quote import (
    render_realtime_quote_text,
)
from tradingagents_cn.dataflows.data_quality import (
    render_data_quality_issues,
    validate_daily_history,
    validate_realtime_quote,
)
from tradingagents_cn.dataflows.vendor_router import (
    describe_vendor_route,
    route_daily_history,
    route_realtime_quote,
)
from tradingagents_cn.indicators import add_tradingagents_indicators


@tool
def akshare_realtime_quote(symbol: str, vendor: str = "auto") -> str:
    """查询 A 股实时/近实时行情快照。

    参数：
    - symbol：6 位 A 股股票代码，例如 002361。
    - vendor：数据源，支持 auto、akshare、eastmoney、sina。

    返回：
    - 一段适合人和大模型阅读的行情文本。

    数据来源：
    - 优先东方财富。
    - 东方财富失败后，使用新浪兜底。

    重要说明：
    这里的“实时”是公开网站的实时/近实时行情快照，
    不是券商低延迟行情，也不是逐笔成交数据。
    """
    route = describe_vendor_route("realtime_quote", vendor)
    quote = route_realtime_quote(symbol, vendor=vendor)
    quality_text = render_data_quality_issues(validate_realtime_quote(quote))
    return (
        f"vendor 路由：{route.vendor}（{route.description}）\n\n"
        + f"{quality_text}\n\n"
        + render_realtime_quote_text(quote)
    )


@tool
def get_market_technical_snapshot(
    symbol: str,
    trade_date: str,
    history_calendar_days: int = 420,
    vendor: str = "auto",
) -> str:
    """查询 A 股历史行情并生成已校验技术面快照。

    参数：
    - symbol：6 位 A 股股票代码，例如 000725。
    - trade_date：分析日期，格式 YYYY-MM-DD。
    - history_calendar_days：向前获取多少个自然日历史行情，默认 420 天。
    - vendor：历史行情数据源，支持 auto、akshare、eastmoney、tencent。

    返回：
    - 一段包含开高低收量、均线、MACD、RSI、布林带、ATR、VWMA、MFI 的文本快照。

    这个工具对应 Market Analyst 最核心的数据来源。
    """
    start_date = calculate_tool_history_start_date(trade_date, history_calendar_days)
    route = describe_vendor_route("daily_history", vendor)
    history = route_daily_history(
        symbol=symbol,
        start_date=start_date,
        end_date=trade_date,
        vendor=vendor,
    )
    quality_text = render_data_quality_issues(
        validate_daily_history(history, trade_date=trade_date)
    )
    indicator_data = add_tradingagents_indicators(history)
    return (
        f"vendor 路由：{route.vendor}（{route.description}）\n\n"
        + f"{quality_text}\n\n"
        + build_verified_market_snapshot_text(
            symbol=symbol,
            trade_date=trade_date,
            indicator_data=indicator_data,
        )
    )


def get_akshare_tools() -> list:
    """返回正式工程当前可用的 AKShare 工具列表。

    LangGraph ToolNode 接收的是工具列表，例如：

    ToolNode(get_akshare_tools())

    以后如果增加新闻工具、公告工具、财报工具，
    也可以统一放进这个列表。
    """
    return [
        akshare_realtime_quote,
        get_market_technical_snapshot,
    ]


def calculate_tool_history_start_date(trade_date: str, history_calendar_days: int) -> str:
    """计算工具查询历史行情的开始日期。"""
    end = datetime.strptime(trade_date, "%Y-%m-%d").date()
    start = end - timedelta(days=int(history_calendar_days))
    return start.strftime("%Y-%m-%d")
