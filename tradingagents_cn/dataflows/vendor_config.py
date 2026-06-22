"""数据源、工具源和 Analyst 选择配置。

这个文件不负责真正抓数据。
它只做三件很工程化的事情：

1. 统一记录默认数据源和工具源；
2. 把用户输入的 Analyst 名称规范化，例如 social -> sentiment；
3. 给主图读取 vendor 配置提供一个稳定入口。

这样后续把 AKShare 换成别的数据源时，不需要在每个 Agent 节点里到处改字符串。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


VALID_ANALYSTS: tuple[str, ...] = (
    "market",
    "sentiment",
    "news",
    "fundamentals",
)


DEFAULT_SELECTED_ANALYSTS: tuple[str, ...] = VALID_ANALYSTS


ANALYST_ALIASES: dict[str, str] = {
    "technical": "market",
    "market_agent": "market",
    "social": "sentiment",
    "sentiment_agent": "sentiment",
    "news_agent": "news",
    "fundamental": "fundamentals",
    "fundamental_agent": "fundamentals",
    "fundamentals_agent": "fundamentals",
}


DEFAULT_DATA_VENDORS: dict[str, str] = {
    # 行情和技术指标当前由 AKShare 封装的公开 A 股数据提供。
    "market_data": "akshare",
    # 新闻工具当前主要读取 AKShare 东方财富个股新闻。
    "news": "akshare",
    # 情绪面当前直接抓公开网页标题，不依赖私有 API。
    "sentiment": "public_web",
    # 财务表和公司资料当前使用 AKShare 封装的公开接口。
    "fundamentals": "akshare",
}


DEFAULT_TOOL_VENDORS: dict[str, str] = {
    "market_agent": "akshare",
    "realtime_quote": "akshare",
    "daily_history": "akshare",
    "news_agent": "akshare",
    "stock_news": "akshare",
    "sentiment_agent": "public_web",
    "fundamentals_agent": "akshare",
    "announcement_tool": "akshare",
    "fundamentals": "akshare",
    "balance_sheet": "akshare",
    "cashflow": "akshare",
    "income_statement": "akshare",
    "sentiment_sources": "eastmoney,xueqiu,tonghuashun,taoguba",
}


VENDOR_OVERRIDE_KEY_ALIASES: dict[str, str] = {
    # 用户更容易写 announcements，但主图内部历史上叫 announcement_tool。
    # 在命令行入口做一次别名归一化，避免配置写了却没有生效。
    "announcement": "announcement_tool",
    "announcements": "announcement_tool",
}


def default_data_vendors() -> dict[str, str]:
    """返回一份可修改的默认数据源配置。"""
    return dict(DEFAULT_DATA_VENDORS)


def default_tool_vendors() -> dict[str, str]:
    """返回一份可修改的默认工具源配置。"""
    return dict(DEFAULT_TOOL_VENDORS)


def normalize_selected_analysts(
    selected_analysts: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """规范化 selected_analysts。

    参数可以是：
        - None：使用默认四个 Analyst；
        - "market,news"：命令行常见写法；
        - ["market", "sentiment"]：代码里常见写法。

    返回值始终是去重后的 tuple，并保持用户指定顺序。
    """
    if selected_analysts is None:
        return DEFAULT_SELECTED_ANALYSTS

    if isinstance(selected_analysts, str):
        raw_items = [
            item.strip()
            for item in selected_analysts.replace("，", ",").split(",")
            if item.strip()
        ]
    else:
        raw_items = [str(item).strip() for item in selected_analysts if str(item).strip()]

    if not raw_items:
        return DEFAULT_SELECTED_ANALYSTS

    normalized: list[str] = []
    for item in raw_items:
        analyst = normalize_analyst_name(item)
        if analyst not in normalized:
            normalized.append(analyst)
    return tuple(normalized)


def normalize_analyst_name(name: str) -> str:
    """把 Analyst 名称和别名转换成内部标准名。"""
    value = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    value = ANALYST_ALIASES.get(value, value)
    if value not in VALID_ANALYSTS:
        supported = "、".join(VALID_ANALYSTS)
        raise ValueError(f"不支持的 Analyst：{name}。当前支持：{supported}")
    return value


def filter_sentiment_analyst(
    selected_analysts: tuple[str, ...],
    include_sentiment: bool,
) -> tuple[str, ...]:
    """根据 include_sentiment 开关移除情绪 Analyst。"""
    if include_sentiment:
        return selected_analysts
    return tuple(analyst for analyst in selected_analysts if analyst != "sentiment")


def normalize_vendor_overrides(entries: Iterable[str] | None) -> dict[str, str]:
    """把命令行 vendor 覆盖项解析成字典。

    输入示例：
        ["market_data=akshare", "sentiment=public_web"]
    """
    overrides: dict[str, str] = {}
    for entry in entries or []:
        text = str(entry or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"vendor 配置必须是 key=value 格式：{text}")
        key, value = text.split("=", 1)
        key = normalize_vendor_override_key(key)
        value = value.strip()
        if not key or not value:
            raise ValueError(f"vendor 配置的 key 和 value 都不能为空：{text}")
        overrides[key] = value
    return overrides


def normalize_vendor_override_key(key: str) -> str:
    """把命令行 vendor key 规范化成内部配置名。"""
    normalized = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    return VENDOR_OVERRIDE_KEY_ALIASES.get(normalized, normalized)


def merge_vendor_config(
    defaults: Mapping[str, str],
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """把默认 vendor 配置和用户覆盖配置合并。"""
    merged = dict(defaults)
    merged.update(dict(overrides or {}))
    return merged


def resolve_data_vendor(
    config: Any,
    key: str,
    default: str | None = None,
) -> str:
    """从 ResearchInputConfig 读取某类数据源。"""
    vendors = getattr(config, "data_vendors", {}) or {}
    fallback = default or DEFAULT_DATA_VENDORS.get(key, "unknown")
    return str(vendors.get(key, fallback))


def resolve_tool_vendor(
    config: Any,
    key: str,
    default: str | None = None,
) -> str:
    """从 ResearchInputConfig 读取某个工具或 Agent 使用的源。"""
    vendors = getattr(config, "tool_vendors", {}) or {}
    fallback = default or DEFAULT_TOOL_VENDORS.get(key, "unknown")
    return str(vendors.get(key, fallback))
