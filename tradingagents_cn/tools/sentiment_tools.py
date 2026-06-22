"""A 股情绪面工具。"""

from __future__ import annotations

from langchain_core.tools import tool

from tradingagents_cn.cache.text_cache import get_or_refresh_text_cache
from tradingagents_cn.dataflows.data_quality import (
    render_data_quality_issues,
    validate_sentiment_items,
)
from tradingagents_cn.dataflows.sentiment import (
    DEFAULT_SENTIMENT_SOURCES,
    render_stock_sentiment_text,
)
from tradingagents_cn.dataflows.vendor_router import describe_vendor_route, route_sentiment_items


DEFAULT_SENTIMENT_CACHE_TTL_HOURS = 2.0


def get_cached_stock_sentiment_text(
    symbol: str,
    max_items: int = 10,
    vendor: str = "auto",
    sources: str = DEFAULT_SENTIMENT_SOURCES,
    max_items_per_source: int | None = None,
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_SENTIMENT_CACHE_TTL_HOURS,
) -> str:
    """获取带缓存的情绪面文本。"""
    cache_key = (
        f"{symbol}:max_items={max_items}:vendor={vendor}:sources={sources}:"
        f"max_items_per_source={max_items_per_source}"
    )

    def fetcher() -> str:
        route = describe_vendor_route("sentiment", vendor)
        items = route_sentiment_items(
            symbol,
            max_items=max_items,
            vendor=vendor,
            sources=sources,
            max_items_per_source=max_items_per_source,
        )
        return (
            f"vendor 路由：{route.vendor}（{route.description}）\n\n"
            + f"{render_data_quality_issues(validate_sentiment_items(items))}\n\n"
            + render_stock_sentiment_text(items)
        )

    result = get_or_refresh_text_cache(
        cache_group="sentiment",
        cache_key=cache_key,
        fetcher=fetcher,
        max_age_days=hours_to_days(cache_ttl_hours),
        force_refresh=force_refresh,
    )
    status = "本地缓存" if result.cache_hit else "联网刷新"
    return f"缓存状态：{status}\n缓存文件：{result.path}\n\n{result.text}"


@tool
def get_stock_sentiment(
    symbol: str,
    max_items: int = 10,
    vendor: str = "auto",
    sources: str = DEFAULT_SENTIMENT_SOURCES,
    max_items_per_source: int | None = None,
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_SENTIMENT_CACHE_TTL_HOURS,
) -> str:
    """查询 A 股个股社区情绪材料。

    参数：
        symbol:
            6 位 A 股股票代码，例如 000725、600519。

        max_items:
            最多返回多少条社区讨论标题。

        vendor:
            情绪数据源，支持 auto、public_web、eastmoney、xueqiu、tonghuashun、taoguba。

        sources:
            逗号分隔的情绪源。
            可选值包括 eastmoney、xueqiu、tonghuashun、taoguba。

        max_items_per_source:
            每个来源最多取多少条。
            不传时根据 max_items 和来源数量自动分配。

        force_refresh:
            是否忽略本地缓存并重新获取。

        cache_ttl_hours:
            情绪面缓存有效小时数，默认 2 小时。

    返回：
        一段适合大模型阅读的情绪面材料。

    当前来源：
        东方财富股吧、雪球、同花顺股吧、淘股吧公开页面。
    """
    return get_cached_stock_sentiment_text(
        symbol=symbol,
        max_items=max_items,
        vendor=vendor,
        sources=sources,
        max_items_per_source=max_items_per_source,
        force_refresh=force_refresh,
        cache_ttl_hours=cache_ttl_hours,
    )


def get_sentiment_tools() -> list:
    """返回情绪面工具列表。"""
    return [
        get_stock_sentiment,
    ]


def hours_to_days(hours: float) -> float:
    """把小时转换成天，用于通用缓存层。"""
    return max(float(hours), 0.0) / 24
