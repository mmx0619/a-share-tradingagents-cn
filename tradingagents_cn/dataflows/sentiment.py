"""A 股情绪面数据层。

原版 TradingAgents 使用 Reddit、StockTwits 等美股社区来源。
A 股版对应使用公开中文社区页面：

    - 东方财富股吧；
    - 雪球；
    - 同花顺股吧；
    - 淘股吧。

这些来源大多不是正式财经数据 API，网页结构也可能变化。
所以这里的设计原则是：

    能解析到就返回；
    解析不到就返回空列表；
    不让情绪源失败拖垮完整投研链路。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html import unescape
from typing import Callable, Iterable
from urllib.parse import quote

import requests

from tradingagents_cn.dataflows.stock_directory import find_stock_name_by_symbol
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


DEFAULT_SENTIMENT_SOURCES = "eastmoney,xueqiu,tonghuashun,taoguba"


SENTIMENT_SOURCE_ALIASES: dict[str, str] = {
    "eastmoney": "eastmoney",
    "eastmoney_guba": "eastmoney",
    "东方财富": "eastmoney",
    "东方财富股吧": "eastmoney",
    "guba": "eastmoney",
    "xueqiu": "xueqiu",
    "雪球": "xueqiu",
    "tonghuashun": "tonghuashun",
    "10jqka": "tonghuashun",
    "同花顺": "tonghuashun",
    "同花顺股吧": "tonghuashun",
    "taoguba": "taoguba",
    "淘股吧": "taoguba",
}


@dataclass
class SentimentItem:
    """一条情绪面材料。

    source:
        材料来源，例如东方财富股吧、雪球。

    title:
        讨论标题或网页中可提取的观点标题。

    url:
        原始链接。
    """

    source: str
    title: str
    url: str


def get_stock_sentiment_items(
    symbol: str,
    max_items: int = 10,
    sources: str | Iterable[str] | None = None,
    max_items_per_source: int | None = None,
    stock_name: str | None = None,
) -> list[SentimentItem]:
    """获取 A 股个股情绪材料。

    这里会按 sources 顺序依次尝试多个公开来源。
    单个来源失败时只跳过该来源，不抛出异常。
    """
    normalized = normalize_cn_symbol(symbol)
    source_keys = parse_sentiment_sources(sources or DEFAULT_SENTIMENT_SOURCES)
    if not source_keys:
        source_keys = parse_sentiment_sources(DEFAULT_SENTIMENT_SOURCES)

    total_limit = max(1, int(max_items))
    per_source_limit = max_items_per_source or max(1, math.ceil(total_limit / len(source_keys)))

    items: list[SentimentItem] = []
    for source_key in source_keys:
        items.extend(
            fetch_sentiment_items_from_source(
                source_key=source_key,
                symbol=normalized,
                max_items=per_source_limit,
                stock_name=stock_name,
            )
        )

    return deduplicate_sentiment_items(items)[:total_limit]


def parse_sentiment_sources(sources: str | Iterable[str] | None) -> list[str]:
    """把情绪源配置规范化成内部来源名列表。"""
    if sources is None:
        return []

    if isinstance(sources, str):
        raw_items = [
            item.strip()
            for item in sources.replace("，", ",").split(",")
            if item.strip()
        ]
    else:
        raw_items = [str(item).strip() for item in sources if str(item).strip()]

    normalized: list[str] = []
    for item in raw_items:
        source = SENTIMENT_SOURCE_ALIASES.get(item, SENTIMENT_SOURCE_ALIASES.get(item.lower()))
        if source and source not in normalized:
            normalized.append(source)
    return normalized


def fetch_sentiment_items_from_source(
    source_key: str,
    symbol: str,
    max_items: int,
    stock_name: str | None = None,
) -> list[SentimentItem]:
    """按来源名调用对应采集函数。"""
    fetchers: dict[str, Callable[..., list[SentimentItem]]] = {
        "eastmoney": get_eastmoney_guba_items,
        "xueqiu": get_xueqiu_items,
        "tonghuashun": get_tonghuashun_guba_items,
        "taoguba": get_taoguba_items,
    }
    fetcher = fetchers.get(source_key)
    if fetcher is None:
        return []

    if source_key == "taoguba":
        return fetcher(symbol, max_items=max_items, stock_name=stock_name)
    return fetcher(symbol, max_items=max_items)


def get_eastmoney_guba_items(symbol: str, max_items: int = 10) -> list[SentimentItem]:
    """从东方财富股吧公开页面提取讨论标题。

    页面地址示例：
        https://guba.eastmoney.com/list,000725.html
    """
    normalized = normalize_cn_symbol(symbol)
    url = f"https://guba.eastmoney.com/list,{normalized}.html"
    html_text = request_public_page(url)
    if not html_text:
        return []
    return parse_eastmoney_guba_html(html_text, base_url="https://guba.eastmoney.com")[
        : max(1, int(max_items))
    ]


def get_xueqiu_items(symbol: str, max_items: int = 10) -> list[SentimentItem]:
    """从雪球股票页提取讨论标题。

    雪球公开页面有时会限制未登录访问。
    如果请求或解析失败，返回空列表。
    """
    normalized = normalize_cn_symbol(symbol)
    prefixed_symbol = to_xueqiu_symbol(normalized)
    url = f"https://xueqiu.com/S/{prefixed_symbol}"
    html_text = request_public_page(
        url,
        headers={
            "Referer": "https://xueqiu.com/",
        },
    )
    if not html_text:
        return []
    return parse_generic_discussion_html(
        html_text,
        base_url="https://xueqiu.com",
        source="雪球",
    )[: max(1, int(max_items))]


def get_tonghuashun_guba_items(symbol: str, max_items: int = 10) -> list[SentimentItem]:
    """从同花顺股吧公开页面提取讨论标题。"""
    normalized = normalize_cn_symbol(symbol)
    url = f"https://guba.10jqka.com.cn/{normalized}/"
    html_text = request_public_page(url)
    if not html_text:
        return []
    return parse_generic_discussion_html(
        html_text,
        base_url="https://guba.10jqka.com.cn",
        source="同花顺股吧",
    )[: max(1, int(max_items))]


def get_taoguba_items(
    symbol: str,
    max_items: int = 10,
    stock_name: str | None = None,
) -> list[SentimentItem]:
    """从淘股吧搜索页提取讨论标题。

    淘股吧通常按股票名称搜索更容易命中。
    如果本地或 AKShare 股票名称表查不到名称，就退回股票代码。
    """
    normalized = normalize_cn_symbol(symbol)
    keyword = stock_name or find_stock_name_by_symbol(normalized) or normalized
    url = f"https://www.taoguba.com.cn/search?searchContent={quote(keyword)}"
    html_text = request_public_page(url)
    if not html_text:
        return []
    return parse_generic_discussion_html(
        html_text,
        base_url="https://www.taoguba.com.cn",
        source="淘股吧",
    )[: max(1, int(max_items))]


def request_public_page(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
) -> str:
    """请求公开网页。

    公开网页经常会有反爬、登录墙或临时网络问题。
    这里捕获所有异常并返回空字符串，让上层把它当作数据缺口处理。
    """
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }
    request_headers.update(headers or {})

    try:
        response = requests.get(url, headers=request_headers, timeout=timeout)
        response.raise_for_status()
    except Exception:
        return ""
    return response.text or ""


def parse_eastmoney_guba_html(html_text: str, base_url: str) -> list[SentimentItem]:
    """解析东方财富股吧页面里的帖子标题。"""
    return parse_generic_discussion_html(
        html_text,
        base_url=base_url,
        source="东方财富股吧",
    )


def parse_generic_discussion_html(
    html_text: str,
    base_url: str,
    source: str,
) -> list[SentimentItem]:
    """从普通 HTML 中提取讨论标题和链接。

    这个解析器故意保持轻量，不追求还原网页结构。
    它只提取后续 Sentiment Agent 需要的“来源、标题、链接”。
    """
    text = str(html_text or "")
    candidates: list[SentimentItem] = []
    seen: set[str] = set()

    for href, raw_title in iter_anchor_candidates(text):
        title = clean_html_text(raw_title)
        if not is_useful_discussion_title(title):
            continue

        url = normalize_href(href, base_url=base_url)
        unique_key = f"{source}|{title}|{url}"
        if unique_key in seen:
            continue
        seen.add(unique_key)
        candidates.append(SentimentItem(source=source, title=title, url=url))

    # 有些站点会把标题放在 JSON 脚本里，而不是 a 标签文本里。
    for title in iter_json_title_candidates(text):
        if not is_useful_discussion_title(title):
            continue
        unique_key = f"{source}|{title}|{base_url}"
        if unique_key in seen:
            continue
        seen.add(unique_key)
        candidates.append(SentimentItem(source=source, title=title, url=base_url))

    return candidates


def iter_anchor_candidates(html_text: str) -> Iterable[tuple[str, str]]:
    """遍历 HTML 中的 a 标签候选。"""
    pattern = re.compile(
        r"<a\b[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html_text):
        yield match.group("href"), match.group("title")


def iter_json_title_candidates(html_text: str) -> Iterable[str]:
    """从网页脚本 JSON 中提取 title 字段候选。"""
    pattern = re.compile(
        r'"(?:title|description)"\s*:\s*"(?P<title>[^"]{4,120})"',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html_text):
        yield clean_html_text(match.group("title"))


def clean_html_text(text: str) -> str:
    """清理 HTML 标签和多余空白。"""
    without_tags = re.sub(r"<[^>]+>", "", str(text or ""))
    normalized_space = re.sub(r"\s+", " ", unescape(without_tags)).strip()
    return normalized_space


def is_useful_discussion_title(title: str) -> bool:
    """判断标题是否像一条可用讨论。"""
    value = str(title or "").strip()
    if len(value) < 4:
        return False
    if value in {"首页", "上一页", "下一页", "末页", "登录", "注册", "发帖", "搜索"}:
        return False
    if value.startswith(("http://", "https://", "javascript:")):
        return False
    return True


def is_useful_guba_title(title: str) -> bool:
    """兼容旧名称：判断股吧标题是否可用。"""
    return is_useful_discussion_title(title)


def normalize_href(href: str, base_url: str) -> str:
    """把相对链接转换为完整链接。"""
    value = str(href or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("/"):
        return f"{base_url}{value}"
    return f"{base_url}/{value}"


def to_xueqiu_symbol(symbol: str) -> str:
    """把 6 位 A 股代码转换成雪球常用前缀代码。"""
    normalized = normalize_cn_symbol(symbol)
    if normalized.startswith("6"):
        return f"SH{normalized}"
    if normalized.startswith(("8", "4")):
        return f"BJ{normalized}"
    return f"SZ{normalized}"


def deduplicate_sentiment_items(items: list[SentimentItem]) -> list[SentimentItem]:
    """按来源、标题和链接去重。"""
    result: list[SentimentItem] = []
    seen: set[str] = set()
    for item in items:
        key = f"{item.source}|{item.title}|{item.url}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def render_stock_sentiment_text(items: list[SentimentItem]) -> str:
    """把情绪材料渲染成给模型阅读的文本。"""
    if not items:
        return (
            "A 股情绪面材料：暂未获取到可用的股吧/社区讨论标题。\n"
            "这代表情绪源缺失或公开页面解析失败，不代表市场没有情绪。"
        )

    sources = "、".join(sorted({item.source for item in items}))
    lines = [
        "A 股情绪面材料：",
        "",
        f"已获取来源：{sources}",
        "",
        "说明：以下内容来自公开社区讨论标题，只能作为市场情绪观察，不能直接作为买卖依据。",
    ]
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                "",
                f"情绪材料 {index}",
                f"来源：{item.source}",
                f"标题：{item.title}",
                f"链接：{item.url}",
            ]
        )
    return "\n".join(lines)
