"""A 股数据源路由器。

这个文件把“选择哪个数据源”的逻辑集中起来。

之前的状态是：

    工具函数直接调用某个 dataflow 函数；
    data_vendors / tool_vendors 只是在配置里记录，真正取数时没有统一路由。

现在的状态是：

    工具函数 -> vendor_router -> 具体数据源函数

这样后续接入 Tushare、财联社、雪球 API 或本地数据库时，
优先在这里加适配器，不需要到每个 Agent 节点里改代码。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents_cn.dataflows import fundamentals as fundamentals_data
from tradingagents_cn.dataflows.announcements import (
    StockAnnouncementItem,
    announcement_frame_to_items,
    calculate_announcement_start_date,
    fetch_cninfo_announcements,
    fetch_eastmoney_announcements,
    get_stock_announcements,
    normalize_announcement_frame,
)
from tradingagents_cn.dataflows.daily_history import (
    get_a_share_daily_history,
    get_daily_history_from_eastmoney,
    get_daily_history_from_tencent,
)
from tradingagents_cn.dataflows.realtime_quote import (
    RealtimeQuote,
    get_realtime_quote,
    get_realtime_quote_from_eastmoney,
    get_realtime_quote_from_sina,
)
from tradingagents_cn.dataflows.sentiment import (
    DEFAULT_SENTIMENT_SOURCES,
    SentimentItem,
    get_stock_sentiment_items,
    parse_sentiment_sources,
)
from tradingagents_cn.dataflows.stock_news import StockNewsItem, get_stock_news
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


class UnsupportedVendorError(ValueError):
    """请求了当前没有实现的数据源。"""


VENDOR_ALIASES: dict[str, str] = {
    "": "auto",
    "default": "auto",
    "ak": "akshare",
    "akshare": "akshare",
    "auto": "auto",
    "eastmoney": "eastmoney",
    "em": "eastmoney",
    "东方财富": "eastmoney",
    "sina": "sina",
    "新浪": "sina",
    "tencent": "tencent",
    "腾讯": "tencent",
    "cninfo": "cninfo",
    "巨潮": "cninfo",
    "巨潮资讯": "cninfo",
    "public_web": "public_web",
    "公开网页": "public_web",
}


SENTIMENT_VENDOR_TO_SOURCES: dict[str, str] = {
    "public_web": DEFAULT_SENTIMENT_SOURCES,
    "eastmoney": "eastmoney",
    "xueqiu": "xueqiu",
    "tonghuashun": "tonghuashun",
    "taoguba": "taoguba",
}


SUPPORTED_VENDORS: dict[str, tuple[str, ...]] = {
    "realtime_quote": ("auto", "akshare", "eastmoney", "sina"),
    "daily_history": ("auto", "akshare", "eastmoney", "tencent"),
    "stock_news": ("auto", "akshare", "eastmoney"),
    "announcements": ("auto", "akshare", "cninfo", "eastmoney"),
    "sentiment": ("auto", "public_web", "eastmoney", "xueqiu", "tonghuashun", "taoguba"),
    "fundamentals": ("auto", "akshare"),
    "balance_sheet": ("auto", "akshare", "eastmoney"),
    "cashflow": ("auto", "akshare", "eastmoney"),
    "income_statement": ("auto", "akshare", "eastmoney"),
}


@dataclass(frozen=True)
class VendorRoute:
    """一次路由决策。"""

    category: str
    vendor: str
    description: str


def normalize_vendor_name(vendor: str | None) -> str:
    """把用户输入的数据源名称规范化。"""
    value = str(vendor or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    return VENDOR_ALIASES.get(value, value)


def ensure_supported_vendor(category: str, vendor: str | None) -> str:
    """校验某类数据是否支持指定 vendor。"""
    normalized = normalize_vendor_name(vendor)
    supported = SUPPORTED_VENDORS.get(category, ())
    if normalized not in supported:
        supported_text = "、".join(supported)
        raise UnsupportedVendorError(
            f"{category} 暂不支持 vendor={vendor}。当前支持：{supported_text}"
        )
    return normalized


def describe_vendor_route(category: str, vendor: str | None) -> VendorRoute:
    """返回当前路由决策说明。"""
    normalized = ensure_supported_vendor(category, vendor)
    descriptions = {
        "auto": "自动兜底路由",
        "akshare": "AKShare 封装公开数据",
        "eastmoney": "东方财富公开数据",
        "sina": "新浪公开数据",
        "tencent": "腾讯证券公开数据",
        "cninfo": "巨潮资讯公开披露数据",
        "public_web": "公开网页情绪源",
        "xueqiu": "雪球公开页面",
        "tonghuashun": "同花顺股吧公开页面",
        "taoguba": "淘股吧公开页面",
    }
    return VendorRoute(
        category=category,
        vendor=normalized,
        description=descriptions.get(normalized, normalized),
    )


def route_realtime_quote(
    symbol: str,
    vendor: str | None = "auto",
    max_retries: int = 2,
    retry_sleep_seconds: float = 1.5,
) -> RealtimeQuote:
    """按 vendor 获取实时/近实时行情。"""
    actual_vendor = ensure_supported_vendor("realtime_quote", vendor)
    normalized_symbol = normalize_cn_symbol(symbol)

    if actual_vendor in {"auto", "akshare"}:
        return get_realtime_quote(
            normalized_symbol,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
    if actual_vendor == "eastmoney":
        return get_realtime_quote_from_eastmoney(normalized_symbol)
    if actual_vendor == "sina":
        return get_realtime_quote_from_sina(normalized_symbol)

    raise UnsupportedVendorError(f"实时行情暂不支持 vendor={vendor}")


def route_daily_history(
    symbol: str,
    start_date: str,
    end_date: str,
    vendor: str | None = "auto",
    max_retries: int = 3,
    retry_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """按 vendor 获取历史日线行情。"""
    actual_vendor = ensure_supported_vendor("daily_history", vendor)
    normalized_symbol = normalize_cn_symbol(symbol)

    if actual_vendor in {"auto", "akshare"}:
        return get_a_share_daily_history(
            normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
    if actual_vendor == "eastmoney":
        return get_daily_history_from_eastmoney(normalized_symbol, start_date, end_date)
    if actual_vendor == "tencent":
        return get_daily_history_from_tencent(normalized_symbol, start_date, end_date)

    raise UnsupportedVendorError(f"历史行情暂不支持 vendor={vendor}")


def route_stock_news(
    symbol: str,
    max_items: int = 10,
    vendor: str | None = "auto",
) -> list[StockNewsItem]:
    """按 vendor 获取个股新闻。"""
    actual_vendor = ensure_supported_vendor("stock_news", vendor)
    normalized_symbol = normalize_cn_symbol(symbol)

    # 当前 AKShare 的个股新闻接口底层主要来自东方财富。
    # 因此 akshare / eastmoney 先路由到同一个实现。
    if actual_vendor in {"auto", "akshare", "eastmoney"}:
        return get_stock_news(normalized_symbol, max_items=max_items)

    raise UnsupportedVendorError(f"个股新闻暂不支持 vendor={vendor}")


def route_stock_announcements(
    symbol: str,
    end_date: str,
    lookback_days: int = 90,
    max_items: int = 10,
    category: str = "",
    vendor: str | None = "auto",
) -> list[StockAnnouncementItem]:
    """按 vendor 获取公司公告。"""
    actual_vendor = ensure_supported_vendor("announcements", vendor)
    normalized_symbol = normalize_cn_symbol(symbol)

    if actual_vendor in {"auto", "akshare"}:
        return get_stock_announcements(
            normalized_symbol,
            end_date=end_date,
            lookback_days=lookback_days,
            max_items=max_items,
            category=category,
        )

    start_date = calculate_announcement_start_date(end_date, lookback_days)
    if actual_vendor == "cninfo":
        frame = fetch_cninfo_announcements(
            normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        source_name = "巨潮资讯"
    elif actual_vendor == "eastmoney":
        frame = fetch_eastmoney_announcements(
            normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        source_name = "东方财富公告"
    else:
        raise UnsupportedVendorError(f"公司公告暂不支持 vendor={vendor}")

    normalized = normalize_announcement_frame(
        frame,
        symbol=normalized_symbol,
        source=source_name,
    )
    items = announcement_frame_to_items(normalized)
    if max_items > 0:
        return items[:max_items]
    return items


def route_sentiment_items(
    symbol: str,
    max_items: int = 10,
    vendor: str | None = "auto",
    sources: str | None = None,
    max_items_per_source: int | None = None,
) -> list[SentimentItem]:
    """按 vendor 获取情绪材料。"""
    actual_vendor = ensure_supported_vendor("sentiment", vendor)
    normalized_symbol = normalize_cn_symbol(symbol)

    actual_sources = sources
    if not actual_sources:
        if actual_vendor == "auto":
            actual_sources = DEFAULT_SENTIMENT_SOURCES
        else:
            actual_sources = SENTIMENT_VENDOR_TO_SOURCES.get(actual_vendor)

    # 如果 vendor 指定了单一情绪源，但 sources 又显式传入多个，
    # 以 sources 为准，因为这是工具调用时更具体的参数。
    if actual_sources:
        parsed = parse_sentiment_sources(actual_sources)
        if not parsed:
            raise UnsupportedVendorError(f"情绪源配置不可用：{actual_sources}")

    return get_stock_sentiment_items(
        normalized_symbol,
        max_items=max_items,
        sources=actual_sources,
        max_items_per_source=max_items_per_source,
    )


def route_fundamentals(
    symbol: str,
    recent_periods: int = 8,
    force_refresh: bool = False,
    vendor: str | None = "auto",
) -> str:
    """按 vendor 获取综合基本面材料。"""
    ensure_supported_vendor("fundamentals", vendor)
    return fundamentals_data.get_fundamentals(
        symbol,
        recent_periods=recent_periods,
        force_refresh=force_refresh,
    )


def route_balance_sheet(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
    vendor: str | None = "auto",
) -> str:
    """按 vendor 获取资产负债表。"""
    ensure_supported_vendor("balance_sheet", vendor)
    return fundamentals_data.get_balance_sheet(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        force_refresh=force_refresh,
    )


def route_cashflow(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
    vendor: str | None = "auto",
) -> str:
    """按 vendor 获取现金流量表。"""
    ensure_supported_vendor("cashflow", vendor)
    return fundamentals_data.get_cashflow(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        force_refresh=force_refresh,
    )


def route_income_statement(
    symbol: str,
    max_rows: int = 8,
    max_columns: int = 30,
    force_refresh: bool = False,
    vendor: str | None = "auto",
) -> str:
    """按 vendor 获取利润表。"""
    ensure_supported_vendor("income_statement", vendor)
    return fundamentals_data.get_income_statement(
        symbol,
        max_rows=max_rows,
        max_columns=max_columns,
        force_refresh=force_refresh,
    )


def list_supported_vendors() -> dict[str, tuple[str, ...]]:
    """返回当前 router 已实现的 vendor 能力表。"""
    return dict(SUPPORTED_VENDORS)
