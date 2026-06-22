"""A 股基本面数据层。

这个文件负责获取和整理 A 股基本面材料。

它对应原版 TradingAgents 里的基本面工具思路：

    get_fundamentals
    get_balance_sheet
    get_cashflow
    get_income_statement

当前 A 股版先使用 AKShare 中已经存在的公开数据接口。

注意：
    这个文件只负责取数和整理文本。
    不判断财务好坏，不给买卖建议。
"""

from __future__ import annotations

import pandas as pd

from tradingagents_cn.cache.text_cache import get_or_refresh_text_cache
from tradingagents_cn.dataflows.symbols import (
    normalize_cn_symbol,
    to_market_prefixed_symbol,
)

COMPANY_PROFILE_CACHE_DAYS = 90
FINANCIAL_STATEMENT_CACHE_DAYS = 35
FINANCIAL_HISTORY_CACHE_DAYS = 35

COMMON_REPORT_COLUMNS = [
    "SECURITY_NAME_ABBR",
    "REPORT_DATE",
    "REPORT_TYPE",
    "REPORT_DATE_NAME",
    "NOTICE_DATE",
    "CURRENCY",
]

BALANCE_SHEET_KEY_COLUMNS = [
    "MONETARYFUNDS",
    "ACCOUNTS_RECE",
    "INVENTORY",
    "CURRENT_ASSET_BALANCE",
    "FIXED_ASSET",
    "GOODWILL",
    "ASSET_BALANCE",
    "ACCOUNTS_PAYABLE",
    "SHORT_LOAN",
    "LONG_LOAN",
    "CURRENT_LIAB_BALANCE",
    "NONCURRENT_LIAB_BALANCE",
    "LIAB_BALANCE",
    "PARENT_EQUITY_BALANCE",
    "EQUITY_BALANCE",
    "LIAB_EQUITY_BALANCE",
]

CASHFLOW_KEY_COLUMNS = [
    "SALES_SERVICES",
    "TOTAL_OPERATE_INFLOW",
    "TOTAL_OPERATE_OUTFLOW",
    "NETCASH_OPERATE",
    "TOTAL_INVEST_INFLOW",
    "TOTAL_INVEST_OUTFLOW",
    "NETCASH_INVEST",
    "TOTAL_FINANCE_INFLOW",
    "TOTAL_FINANCE_OUTFLOW",
    "NETCASH_FINANCE",
    "CCE_ADD",
    "BEGIN_CCE",
    "END_CCE",
]

INCOME_STATEMENT_KEY_COLUMNS = [
    "TOTAL_OPERATE_INCOME",
    "TOTAL_OPERATE_INCOME_YOY",
    "OPERATE_INCOME",
    "TOTAL_OPERATE_COST",
    "OPERATE_COST",
    "RESEARCH_EXPENSE",
    "SALE_EXPENSE",
    "MANAGE_EXPENSE",
    "FINANCE_EXPENSE",
    "OPERATE_PROFIT",
    "TOTAL_PROFIT",
    "INCOME_TAX",
    "NETPROFIT",
    "NETPROFIT_YOY",
    "PARENT_NETPROFIT",
    "DEDUCT_PARENT_NETPROFIT",
    "BASIC_EPS",
]

FINANCIAL_HISTORY_KEY_METRICS = [
    "营业总收入",
    "营业成本",
    "归母净利润",
    "净利润",
    "扣非净利润",
    "经营现金流量净额",
    "基本每股收益",
    "每股净资产",
    "每股现金流",
    "净资产收益率(ROE)",
    "总资产报酬率(ROA)",
    "毛利率",
    "销售净利率",
    "资产负债率",
]


def to_eastmoney_finance_symbol(symbol: str) -> str:
    """转换成东方财富财务接口需要的带市场前缀格式。

    AKShare 的东方财富财务报表接口示例参数是：
        SH600519
        SZ000001

    我们内部通常使用 6 位数字代码。
    所以这里统一转换。
    """
    return to_market_prefixed_symbol(symbol).upper()


def get_company_profile(symbol: str, force_refresh: bool = False) -> str:
    """获取公司基本资料。

    当前尝试两个公开来源：

    1. 东方财富个股资料：
       ak.stock_individual_info_em(symbol="600519")

    2. 巨潮资讯公司概况：
       ak.stock_profile_cninfo(symbol="600519")

    返回：
        适合放进 Fundamentals Agent Prompt 的文本。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    cache_result = get_or_refresh_text_cache(
        cache_group="fundamentals",
        cache_key=f"{normalized_symbol}:company_profile",
        fetcher=lambda: _fetch_company_profile(normalized_symbol),
        max_age_days=COMPANY_PROFILE_CACHE_DAYS,
        force_refresh=force_refresh,
    )
    return add_cache_note(
        cache_result.text,
        cache_hit=cache_result.cache_hit,
        cache_days=COMPANY_PROFILE_CACHE_DAYS,
    )


def _fetch_company_profile(symbol: str) -> str:
    """真正联网获取公司基本资料。

    外层 get_company_profile(...) 会先检查本地缓存。
    只有缓存不存在、过期、或者 force_refresh=True 时，
    才会调用这个函数。
    """
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
    sections: list[str] = []

    try:
        eastmoney_info = ak.stock_individual_info_em(symbol=normalized_symbol)
        sections.append(
            "## 东方财富个股资料\n\n"
            + render_dataframe_text(eastmoney_info, max_rows=80)
        )
    except Exception as error:
        sections.append(f"## 东方财富个股资料\n\n获取失败：{error}")

    try:
        cninfo_profile = ak.stock_profile_cninfo(symbol=normalized_symbol)
        sections.append(
            "## 巨潮资讯公司概况\n\n"
            + render_dataframe_text(cninfo_profile, max_rows=80)
        )
    except Exception as error:
        sections.append(f"## 巨潮资讯公司概况\n\n获取失败：{error}")

    return "\n\n".join(sections)


def get_balance_sheet(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
) -> str:
    """获取资产负债表。

    数据来源：
        AKShare 东方财富资产负债表接口：
        stock_balance_sheet_by_report_em
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    cache_result = get_or_refresh_text_cache(
        cache_group="fundamentals",
        cache_key=f"{normalized_symbol}:balance_sheet:rows{max_rows}:cols{max_columns}",
        fetcher=lambda: _fetch_balance_sheet(
            normalized_symbol,
            max_rows=max_rows,
            max_columns=max_columns,
        ),
        max_age_days=FINANCIAL_STATEMENT_CACHE_DAYS,
        force_refresh=force_refresh,
    )
    return add_cache_note(
        cache_result.text,
        cache_hit=cache_result.cache_hit,
        cache_days=FINANCIAL_STATEMENT_CACHE_DAYS,
    )


def _fetch_balance_sheet(symbol: str, max_rows: int = 8, max_columns: int = 30) -> str:
    """真正联网获取资产负债表。"""
    import akshare as ak

    finance_symbol = to_eastmoney_finance_symbol(symbol)
    data = ak.stock_balance_sheet_by_report_em(symbol=finance_symbol)
    key_data = select_key_columns(
        data,
        COMMON_REPORT_COLUMNS + BALANCE_SHEET_KEY_COLUMNS,
    )
    return render_dataframe_text(key_data, max_rows=max_rows, max_columns=max_columns)


def get_cashflow(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
) -> str:
    """获取现金流量表。

    数据来源：
        AKShare 东方财富现金流量表接口：
        stock_cash_flow_sheet_by_report_em
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    cache_result = get_or_refresh_text_cache(
        cache_group="fundamentals",
        cache_key=f"{normalized_symbol}:cashflow:rows{max_rows}:cols{max_columns}",
        fetcher=lambda: _fetch_cashflow(
            normalized_symbol,
            max_rows=max_rows,
            max_columns=max_columns,
        ),
        max_age_days=FINANCIAL_STATEMENT_CACHE_DAYS,
        force_refresh=force_refresh,
    )
    return add_cache_note(
        cache_result.text,
        cache_hit=cache_result.cache_hit,
        cache_days=FINANCIAL_STATEMENT_CACHE_DAYS,
    )


def _fetch_cashflow(symbol: str, max_rows: int = 8, max_columns: int = 30) -> str:
    """真正联网获取现金流量表。"""
    import akshare as ak

    finance_symbol = to_eastmoney_finance_symbol(symbol)
    data = ak.stock_cash_flow_sheet_by_report_em(symbol=finance_symbol)
    key_data = select_key_columns(
        data,
        COMMON_REPORT_COLUMNS + CASHFLOW_KEY_COLUMNS,
    )
    return render_dataframe_text(key_data, max_rows=max_rows, max_columns=max_columns)


def get_income_statement(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
) -> str:
    """获取利润表。

    数据来源：
        AKShare 东方财富利润表接口：
        stock_profit_sheet_by_report_em
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    cache_result = get_or_refresh_text_cache(
        cache_group="fundamentals",
        cache_key=f"{normalized_symbol}:income_statement:rows{max_rows}:cols{max_columns}",
        fetcher=lambda: _fetch_income_statement(
            normalized_symbol,
            max_rows=max_rows,
            max_columns=max_columns,
        ),
        max_age_days=FINANCIAL_STATEMENT_CACHE_DAYS,
        force_refresh=force_refresh,
    )
    return add_cache_note(
        cache_result.text,
        cache_hit=cache_result.cache_hit,
        cache_days=FINANCIAL_STATEMENT_CACHE_DAYS,
    )


def _fetch_income_statement(symbol: str, max_rows: int = 8, max_columns: int = 30) -> str:
    """真正联网获取利润表。"""
    import akshare as ak

    finance_symbol = to_eastmoney_finance_symbol(symbol)
    data = ak.stock_profit_sheet_by_report_em(symbol=finance_symbol)
    key_data = select_key_columns(
        data,
        COMMON_REPORT_COLUMNS + INCOME_STATEMENT_KEY_COLUMNS,
    )
    return render_dataframe_text(key_data, max_rows=max_rows, max_columns=max_columns)


def get_financial_history(
    symbol: str,
    max_rows: int = 12,
    recent_periods: int = 8,
    force_refresh: bool = False,
) -> str:
    """获取财务历史摘要。

    数据来源：
        AKShare 新浪财经关键财务指标：
        stock_financial_abstract

    这类数据适合让基本面 Agent 观察历史变化，
    但具体解释仍交给大模型。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    cache_result = get_or_refresh_text_cache(
        cache_group="fundamentals",
        cache_key=f"{normalized_symbol}:financial_history:rows{max_rows}:periods{recent_periods}",
        fetcher=lambda: _fetch_financial_history(
            normalized_symbol,
            max_rows=max_rows,
            recent_periods=recent_periods,
        ),
        max_age_days=FINANCIAL_HISTORY_CACHE_DAYS,
        force_refresh=force_refresh,
    )
    return add_cache_note(
        cache_result.text,
        cache_hit=cache_result.cache_hit,
        cache_days=FINANCIAL_HISTORY_CACHE_DAYS,
    )


def _fetch_financial_history(
    symbol: str,
    max_rows: int = 12,
    recent_periods: int = 8,
) -> str:
    """真正联网获取财务历史摘要。"""
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
    data = ak.stock_financial_abstract(symbol=normalized_symbol)
    key_data = select_key_financial_history_rows(
        data,
        FINANCIAL_HISTORY_KEY_METRICS,
    )
    recent_data = select_recent_financial_history_periods(
        key_data,
        recent_periods=recent_periods,
    )
    return render_dataframe_text(
        recent_data,
        max_rows=max_rows,
        max_columns=recent_periods + 2,
    )


def get_fundamentals(
    symbol: str,
    recent_periods: int = 8,
    force_refresh: bool = False,
) -> str:
    """获取综合基本面材料。

    这个函数对应原版 TradingAgents 的 get_fundamentals 思路。
    它把公司资料和财务历史摘要合并成一段文本。

    更细的三张表：
        - 资产负债表；
        - 现金流量表；
        - 利润表；

    可以分别调用对应函数。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    return f"""# {normalized_symbol} 综合基本面材料

{get_company_profile(normalized_symbol, force_refresh=force_refresh)}

## 财务历史摘要

{get_financial_history(
    normalized_symbol,
    recent_periods=recent_periods,
    force_refresh=force_refresh,
)}
"""


def add_cache_note(text: str, cache_hit: bool, cache_days: int) -> str:
    """给返回文本加一行缓存说明。

    这样后续大模型和你本人都能看到：
        这份基本面材料是刚刚联网获取的，
        还是来自本地缓存。
    """
    source = "本地缓存" if cache_hit else "联网刷新"
    return f"缓存状态：{source}，缓存有效期 {cache_days} 天。\n\n{text}"


def render_dataframe_text(
    data: pd.DataFrame,
    max_rows: int = 10,
    max_columns: int = 30,
) -> str:
    """把 DataFrame 渲染成适合大模型阅读的 Markdown 文本。

    为什么不直接返回 DataFrame？

    因为后续 Fundamentals Agent 的输入是 Prompt 文本。
    Markdown 表格比 Python 对象更适合放进 Prompt。
    """
    if data is None or data.empty:
        return "暂无数据。"

    original_rows = len(data)
    original_columns = len(data.columns)

    frame = data.copy().head(max_rows)
    frame = frame.iloc[:, :max_columns]
    frame = frame.fillna("")

    notes: list[str] = []
    if original_rows > len(frame):
        notes.append(f"已截取前 {len(frame)} 行，原始共 {original_rows} 行。")
    if original_columns > len(frame.columns):
        notes.append(f"已截取前 {len(frame.columns)} 列，原始共 {original_columns} 列。")

    note_text = ""
    if notes:
        note_text = "\n\n" + "\n".join(f"说明：{note}" for note in notes)

    return frame.to_markdown(index=False) + note_text


def select_key_columns(data: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """从宽财务表中筛选关键字段。

    A 股财务表字段很多，直接截取前 N 列不专业。
    这里按预先定义的关键字段列表筛选。

    如果某些字段在当前数据源中不存在，就自动跳过。
    """
    if data is None or data.empty:
        return pd.DataFrame()

    existing_columns = [
        column
        for column in key_columns
        if column in data.columns
    ]

    if not existing_columns:
        return data

    return data[existing_columns].copy()


def select_key_financial_history_rows(
    data: pd.DataFrame,
    key_metrics: list[str],
) -> pd.DataFrame:
    """从财务历史摘要中筛选关键指标行。

    新浪财务摘要是“指标在行、报告期在列”的宽表。
    所以这里不是筛选列，而是筛选“指标”这一列中的关键行。
    """
    if data is None or data.empty:
        return pd.DataFrame()

    if "指标" not in data.columns:
        return data

    selected = data[data["指标"].isin(key_metrics)].copy()
    if selected.empty:
        return data

    return selected


def select_recent_financial_history_periods(
    data: pd.DataFrame,
    recent_periods: int = 8,
) -> pd.DataFrame:
    """从财务历史宽表中保留最近若干报告期。

    新浪财务历史表的列通常是：
        选项 / 指标 / 20260331 / 20251231 / ...

    这里保留：
        - 选项；
        - 指标；
        - 最近 recent_periods 个报告期。

    这样比直接取前 N 列更明确：
    我们要的是最近报告期，不是随便截列。
    """
    if data is None or data.empty:
        return pd.DataFrame()

    base_columns = [
        column
        for column in ("选项", "指标")
        if column in data.columns
    ]

    period_columns = [
        column
        for column in data.columns
        if _is_report_period_column(column)
    ]

    period_columns = sorted(period_columns, reverse=True)
    selected_periods = period_columns[: max(1, int(recent_periods))]

    selected_columns = base_columns + selected_periods
    if not selected_columns:
        return data

    return data[selected_columns].copy()


def _is_report_period_column(column: object) -> bool:
    """判断某个列名是否像 20260331 这样的报告期。"""
    text = str(column)
    return len(text) == 8 and text.isdigit()
