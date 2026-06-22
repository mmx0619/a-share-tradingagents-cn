"""工具层。

这里统一暴露正式工程当前可用的工具入口。
"""

from tradingagents_cn.tools.akshare_tools import akshare_realtime_quote
from tradingagents_cn.tools.fundamentals_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_fundamentals_tools,
    get_income_statement,
)
from tradingagents_cn.tools.news_tools import akshare_stock_news
from tradingagents_cn.tools.registry import (
    ToolExecutionResult,
    build_openai_compatible_tools_schema,
    execute_tool_call,
    get_tool_map,
    get_trading_tools,
)

__all__ = [
    "ToolExecutionResult",
    "akshare_realtime_quote",
    "akshare_stock_news",
    "build_openai_compatible_tools_schema",
    "execute_tool_call",
    "get_balance_sheet",
    "get_cashflow",
    "get_fundamentals",
    "get_fundamentals_tools",
    "get_income_statement",
    "get_tool_map",
    "get_trading_tools",
]
