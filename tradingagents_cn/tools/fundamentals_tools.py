"""A 股基本面工具。

这个文件属于正式工程的工具层。

它把 dataflows/fundamentals.py 里的基本面取数函数，
包装成大模型可以 Tool Calling 的工具。

工具设计参考原版 TradingAgents：

    get_fundamentals
    get_balance_sheet
    get_cashflow
    get_income_statement

这些工具只负责返回基本面材料。
不判断财务好坏。
不输出买入、卖出。
"""

from __future__ import annotations

from langchain_core.tools import tool

from tradingagents_cn.dataflows.data_quality import (
    render_data_quality_issues,
    validate_fundamental_texts,
)
from tradingagents_cn.dataflows.vendor_router import (
    route_balance_sheet,
    route_cashflow,
    route_fundamentals,
    route_income_statement,
)


@tool
def get_fundamentals(
    symbol: str,
    recent_periods: int = 8,
    vendor: str = "auto",
    force_refresh: bool = False,
) -> str:
    """获取 A 股综合基本面材料。

    参数：
        symbol:
            6 位 A 股股票代码，例如 002361、600519。

        recent_periods:
            财务历史摘要保留最近多少个报告期。

        vendor:
            综合基本面数据源，支持 auto、akshare。

        force_refresh:
            是否忽略本地缓存，强制重新联网获取。

    返回：
        公司资料和财务历史摘要文本。

    数据来源：
        AKShare 封装的公开数据接口，当前包括东方财富、巨潮资讯、 新浪财经等来源。
    """
    text = route_fundamentals(
        symbol,
        recent_periods=recent_periods,
        vendor=vendor,
        force_refresh=force_refresh,
    )
    return prepend_fundamental_quality("综合基本面材料", text)


@tool
def get_balance_sheet(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    vendor: str = "auto",
    force_refresh: bool = False,
) -> str:
    """获取 A 股资产负债表材料。

    参数：
        symbol:
            6 位 A 股股票代码，例如 002361、600519。

        max_rows:
            最多返回多少行表格。

        max_columns:
            最多返回多少列表格。
            A 股财务表字段很多，限制列数可以避免 Prompt 过长。

        vendor:
            资产负债表数据源，支持 auto、akshare、eastmoney。

        force_refresh:
            是否忽略本地缓存，强制重新联网获取。

    返回：
        Markdown 表格文本。

    数据来源：
        AKShare 的东方财富资产负债表接口。
    """
    text = route_balance_sheet(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        vendor=vendor,
        force_refresh=force_refresh,
    )
    return prepend_fundamental_quality("资产负债表", text)


@tool
def get_cashflow(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    vendor: str = "auto",
    force_refresh: bool = False,
) -> str:
    """获取 A 股现金流量表材料。

    参数：
        symbol:
            6 位 A 股股票代码，例如 002361、600519。

        max_rows:
            最多返回多少行表格。

        max_columns:
            最多返回多少列表格。
            A 股财务表字段很多，限制列数可以避免 Prompt 过长。

        vendor:
            现金流量表数据源，支持 auto、akshare、eastmoney。

        force_refresh:
            是否忽略本地缓存，强制重新联网获取。

    返回：
        Markdown 表格文本。

    数据来源：
        AKShare 的东方财富现金流量表接口。
    """
    text = route_cashflow(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        vendor=vendor,
        force_refresh=force_refresh,
    )
    return prepend_fundamental_quality("现金流量表", text)


@tool
def get_income_statement(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    vendor: str = "auto",
    force_refresh: bool = False,
) -> str:
    """获取 A 股利润表材料。

    参数：
        symbol:
            6 位 A 股股票代码，例如 002361、600519。

        max_rows:
            最多返回多少行表格。

        max_columns:
            最多返回多少列表格。
            A 股财务表字段很多，限制列数可以避免 Prompt 过长。

        vendor:
            利润表数据源，支持 auto、akshare、eastmoney。

        force_refresh:
            是否忽略本地缓存，强制重新联网获取。

    返回：
        Markdown 表格文本。

    数据来源：
        AKShare 的东方财富利润表接口。
    """
    text = route_income_statement(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        vendor=vendor,
        force_refresh=force_refresh,
    )
    return prepend_fundamental_quality("利润表", text)


def get_fundamentals_tools() -> list:
    """返回基本面相关工具列表。

    后续 Fundamentals Agent 会通过这些工具获取：
        - 综合基本面；
        - 资产负债表；
        - 现金流量表；
        - 利润表。
    """
    return [
        get_fundamentals,
        get_balance_sheet,
        get_cashflow,
        get_income_statement,
    ]


def prepend_fundamental_quality(label: str, text: str) -> str:
    """给基本面工具输出补充数据质量提示。"""
    issues = validate_fundamental_texts({label: text})
    return f"{render_data_quality_issues(issues)}\n\n{text}"
