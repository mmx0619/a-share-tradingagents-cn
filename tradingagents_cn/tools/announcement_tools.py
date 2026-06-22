"""A 股公告工具。"""

from __future__ import annotations

from langchain_core.tools import tool

from tradingagents_cn.cache.text_cache import get_or_refresh_text_cache
from tradingagents_cn.dataflows.data_quality import (
    render_data_quality_issues,
    validate_announcement_items,
)
from tradingagents_cn.dataflows.announcements import (
    render_stock_announcements_text,
)
from tradingagents_cn.dataflows.vendor_router import (
    describe_vendor_route,
    route_stock_announcements,
)


DEFAULT_ANNOUNCEMENT_CACHE_TTL_HOURS = 24.0


def get_cached_stock_announcements_text(
    symbol: str,
    end_date: str,
    lookback_days: int = 90,
    max_items: int = 10,
    category: str = "",
    vendor: str = "auto",
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_ANNOUNCEMENT_CACHE_TTL_HOURS,
) -> str:
    """获取带缓存的公告文本。"""
    cache_key = (
        f"{symbol}:end={end_date}:lookback={lookback_days}:"
        f"max={max_items}:category={category}:vendor={vendor}"
    )

    def fetcher() -> str:
        route = describe_vendor_route("announcements", vendor)
        items = route_stock_announcements(
            symbol=symbol,
            end_date=end_date,
            lookback_days=lookback_days,
            max_items=max_items,
            category=category,
            vendor=vendor,
        )
        return (
            f"vendor 路由：{route.vendor}（{route.description}）\n\n"
            + f"{render_data_quality_issues(validate_announcement_items(items, trade_date=end_date))}\n\n"
            + render_stock_announcements_text(items)
        )

    result = get_or_refresh_text_cache(
        cache_group="announcements",
        cache_key=cache_key,
        fetcher=fetcher,
        max_age_days=hours_to_days(cache_ttl_hours),
        force_refresh=force_refresh,
    )
    status = "本地缓存" if result.cache_hit else "联网刷新"
    return f"缓存状态：{status}\n缓存文件：{result.path}\n\n{result.text}"


@tool
def get_stock_announcements_tool(
    symbol: str,
    end_date: str,
    lookback_days: int = 90,
    max_items: int = 10,
    category: str = "",
    vendor: str = "auto",
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_ANNOUNCEMENT_CACHE_TTL_HOURS,
) -> str:
    """查询 A 股公司公告/信息披露材料。

    参数：
        symbol:
            6 位 A 股代码，例如 000725、600519。

        end_date:
            截止日期，格式 YYYY-MM-DD。

        lookback_days:
            向前查询多少个自然日，默认 90。

        max_items:
            最多返回多少条公告。

        category:
            公告分类，例如 年报、半年报、风险提示、权益分派。
            空字符串表示全部分类。

        vendor:
            公告数据源，支持 auto、akshare、cninfo、eastmoney。

        force_refresh:
            是否忽略本地缓存并重新获取。

        cache_ttl_hours:
            公告缓存有效小时数，默认 24 小时。

    返回：
        适合大模型阅读的公告材料文本。

    数据来源：
        优先巨潮资讯，失败后尝试东方财富公告。
    """
    return get_cached_stock_announcements_text(
        symbol=symbol,
        end_date=end_date,
        lookback_days=lookback_days,
        max_items=max_items,
        category=category,
        vendor=vendor,
        force_refresh=force_refresh,
        cache_ttl_hours=cache_ttl_hours,
    )


def get_announcement_tools() -> list:
    """返回公告相关工具列表。"""
    return [
        get_stock_announcements_tool,
    ]


def hours_to_days(hours: float) -> float:
    """把小时转换成天，用于通用缓存层。"""
    return max(float(hours), 0.0) / 24
