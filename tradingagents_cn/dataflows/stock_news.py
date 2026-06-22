"""A 股个股新闻数据层。

这个文件属于正式工程的 dataflows 数据层。

它负责：

1. 调用 AKShare 获取 A 股个股新闻。
2. 把 AKShare 返回的中文字段统一成稳定字段。
3. 把 DataFrame 转成 StockNewsItem 对象列表。
4. 把新闻列表渲染成适合人和大模型阅读的文本。

它不负责：

1. 调用大模型。
2. 判断新闻是利好还是利空。
3. 生成交易建议。
4. 做风控。

重要说明：

这里拿到的是公开新闻源中“当前接口能返回的最近新闻”，
不保证每条都是今天刚发生的实时新闻。

新闻数据的价值是给 News Agent 提供原材料。
真正的事件提取、影响判断、风险归纳，会放在后面的 Agent 层。
"""

from __future__ import annotations

from dataclasses import dataclass

import akshare as ak
import pandas as pd

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


@dataclass
class StockNewsItem:
    """单条个股新闻。

    symbol:
        6 位股票代码，例如 002361。

    title:
        新闻标题。

    content:
        新闻正文或摘要。

    publish_time:
        发布时间。

    source:
        文章来源。

    url:
        新闻链接。
    """

    symbol: str
    title: str
    content: str
    publish_time: str
    source: str
    url: str


def normalize_stock_news_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把 AKShare 返回的个股新闻表整理成统一字段。

    AKShare stock_news_em() 当前常见中文字段：

        新闻标题
        新闻内容
        发布时间
        文章来源
        新闻链接

    正式工程内部统一改成：

        Symbol
        Title
        Content
        PublishTime
        Source
        Url

    为什么要改字段名？

    因为后面的 Agent、Prompt、报表生成代码，
    不应该到处写中文 DataFrame 列名。
    统一字段后，后续维护更稳定。
    """
    if data.empty:
        return pd.DataFrame(
            columns=["Symbol", "Title", "Content", "PublishTime", "Source", "Url"]
        )

    required_columns = ["新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"新闻数据缺少必要字段：{missing_columns}")

    normalized = pd.DataFrame(index=data.index)
    normalized["Symbol"] = [symbol] * len(data)
    normalized["Title"] = data["新闻标题"].fillna("").astype(str)
    normalized["Content"] = data["新闻内容"].fillna("").astype(str)
    normalized["PublishTime"] = data["发布时间"].fillna("").astype(str)
    normalized["Source"] = data["文章来源"].fillna("").astype(str)
    normalized["Url"] = data["新闻链接"].fillna("").astype(str)

    return normalized


def news_frame_to_items(data: pd.DataFrame) -> list[StockNewsItem]:
    """把整理后的 DataFrame 转成新闻对象列表。

    DataFrame 适合表格处理。
    但是后面的 Agent 更适合读取结构清晰的对象列表。
    """
    items: list[StockNewsItem] = []

    for row in data.to_dict(orient="records"):
        items.append(
            StockNewsItem(
                symbol=str(row["Symbol"]),
                title=str(row["Title"]),
                content=str(row["Content"]),
                publish_time=str(row["PublishTime"]),
                source=str(row["Source"]),
                url=str(row["Url"]),
            )
        )

    return items


def get_stock_news(symbol: str, max_items: int = 10) -> list[StockNewsItem]:
    """获取某只 A 股的最近个股新闻。

    参数：
        symbol:
            股票代码，例如 002361、600519。

        max_items:
            最多返回多少条新闻。

    数据来源：
        AKShare 的 stock_news_em()。
        底层主要来自东方财富个股新闻。

    注意：
        AKShare 这个接口通常返回最近一批新闻。
        这里用 max_items 截取前几条，避免后续 Prompt 太长。
    """
    normalized_symbol = normalize_cn_symbol(symbol)

    raw_news = ak.stock_news_em(symbol=normalized_symbol)
    normalized_news = normalize_stock_news_frame(raw_news, normalized_symbol)

    if max_items > 0:
        normalized_news = normalized_news.head(max_items)

    return news_frame_to_items(normalized_news)


def render_stock_news_text(news_items: list[StockNewsItem]) -> str:
    """把新闻列表渲染成适合人和大模型阅读的文本。

    大模型读这种有固定格式的文本，
    比直接读 DataFrame 更稳定。
    """
    if not news_items:
        return "暂无相关新闻。"

    lines: list[str] = []

    for index, item in enumerate(news_items, start=1):
        lines.append(
            f"""新闻 {index}
标题：{item.title}
时间：{item.publish_time}
来源：{item.source}
内容：{item.content}
链接：{item.url}"""
        )

    return "\n\n".join(lines)
