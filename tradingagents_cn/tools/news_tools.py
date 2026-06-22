"""A 股个股新闻工具。

这个文件属于正式工程里的“工具层”。

你可以把它理解成：

用户问：
    002361 最近有什么新闻？

大模型判断：
    我需要调用新闻工具。

ToolNode 执行：
    akshare_stock_news(symbol="002361", max_items=5)

工具返回：
    一段整理好的新闻文本。

大模型再根据这段新闻文本，继续分析利好、利空、风险和交易含义。

注意：
    当前工具层只负责把正式数据层包装成大模型可调用的工具。

    真正的新闻采集逻辑位于：
    tradingagents_cn.dataflows.stock_news
"""

from __future__ import annotations

from langchain_core.tools import tool

from tradingagents_cn.cache.text_cache import get_or_refresh_text_cache
from tradingagents_cn.dataflows.data_quality import (
    render_data_quality_issues,
    validate_stock_news_items,
)
from tradingagents_cn.dataflows.stock_news import (
    render_stock_news_text,
)
from tradingagents_cn.dataflows.vendor_router import describe_vendor_route, route_stock_news


DEFAULT_NEWS_CACHE_TTL_HOURS = 6.0


def get_cached_stock_news_text(
    symbol: str,
    max_items: int = 5,
    trade_date: str = "",
    vendor: str = "auto",
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_NEWS_CACHE_TTL_HOURS,
) -> str:
    """获取带缓存的个股新闻文本。"""
    cache_key = f"{symbol}:max_items={max_items}:trade_date={trade_date}:vendor={vendor}"

    def fetcher() -> str:
        route = describe_vendor_route("stock_news", vendor)
        news_items = route_stock_news(symbol, max_items=max_items, vendor=vendor)
        quality_issues = (
            validate_stock_news_items(news_items, trade_date=trade_date)
            if trade_date
            else []
        )
        return (
            f"vendor 路由：{route.vendor}（{route.description}）\n\n"
            + f"{render_data_quality_issues(quality_issues)}\n\n"
            + render_stock_news_text(news_items)
        )

    result = get_or_refresh_text_cache(
        cache_group="news",
        cache_key=cache_key,
        fetcher=fetcher,
        max_age_days=hours_to_days(cache_ttl_hours),
        force_refresh=force_refresh,
    )
    status = "本地缓存" if result.cache_hit else "联网刷新"
    return f"缓存状态：{status}\n缓存文件：{result.path}\n\n{result.text}"


@tool
def akshare_stock_news(
    symbol: str,
    max_items: int = 5,
    trade_date: str = "",
    vendor: str = "auto",
    force_refresh: bool = False,
    cache_ttl_hours: float = DEFAULT_NEWS_CACHE_TTL_HOURS,
) -> str:
    """查询 A 股个股新闻。

    参数：
        symbol:
            A 股股票代码，例如 002361、600519。

        max_items:
            最多返回多少条新闻。

            为什么要限制条数？
            因为新闻正文会进入大模型上下文。
            如果一次塞进去几十条新闻，提示词会非常长，
            既浪费 token，也容易让模型抓不住重点。

        trade_date:
            分析日期，格式 YYYY-MM-DD。
            传入后会校验新闻是否明显偏旧。

        vendor:
            新闻数据源，支持 auto、akshare、eastmoney。

        force_refresh:
            是否忽略本地缓存并重新获取。

        cache_ttl_hours:
            新闻缓存有效小时数，默认 6 小时。

    返回：
        一段适合人和大模型阅读的新闻文本。

    数据来源：
        AKShare 的 stock_news_em 接口。
        这个接口底层主要来自东方财富个股新闻。

    重要说明：
        这里拿到的是公开新闻源里“最近可获取的新闻”，
        不保证每条都是今天实时发生的新闻。

        对交易系统来说，新闻工具的价值不是“看到标题就买卖”，
        而是给 News Agent 提供原材料，让大模型进一步判断：
        - 有没有重大事件；
        - 是公告、龙虎榜、快讯，还是普通资讯；
        - 对股价影响是强、中、弱；
        - 是否需要进入风控节点。
    """
    return get_cached_stock_news_text(
        symbol=symbol,
        max_items=max_items,
        trade_date=trade_date,
        vendor=vendor,
        force_refresh=force_refresh,
        cache_ttl_hours=cache_ttl_hours,
    )


def get_news_tools() -> list:
    """返回新闻相关工具列表。

    LangGraph 的 ToolNode 需要接收一个工具列表，例如：

        ToolNode(get_news_tools())

    为什么这里返回 list？
        因为后续新闻工具会越来越多，例如：
        - 个股新闻；
        - 公司公告；
        - 财联社快讯；
        - 交易所公告；
        - 舆情评论。

    现在第 49 步只放一个工具：
        akshare_stock_news
    """
    return [
        akshare_stock_news,
    ]


def hours_to_days(hours: float) -> float:
    """把小时转换成天，用于通用缓存层。"""
    return max(float(hours), 0.0) / 24
