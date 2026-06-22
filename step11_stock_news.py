"""第 11 步：获取 A 股个股新闻。

前面第 10 步已经跑通：

真实行情
  ↓
技术指标
  ↓
市场快照
  ↓
市场分析师 Agent
  ↓
真实大模型报告

但是 TradingAgents 不是只看技术面。
它还会看新闻、公告、情绪、基本面等信息。

当前文件先做“新闻原材料采集层”：

股票代码
  ↓
AKShare 调用东方财富个股新闻
  ↓
整理字段
  ↓
输出新闻列表

注意：
这个文件只负责获取和整理新闻。
它不调用大模型，也不判断利好利空。
新闻分析会放到后面的 News Agent。
"""

from __future__ import annotations

from dataclasses import dataclass

import akshare as ak
import pandas as pd

import step01_akshare_cn as symbol_mod


@dataclass
class StockNewsItem:
    """单条个股新闻。

    字段说明：
    - symbol：股票代码，比如 600519。
    - title：新闻标题。
    - content：新闻正文摘要。
    - publish_time：发布时间。
    - source：文章来源。
    - url：新闻链接。

    为什么要转成 dataclass：
    AKShare 返回的是 DataFrame，适合表格计算。
    但后面的 Agent 更适合读取一条一条结构清晰的新闻对象。
    """

    symbol: str
    title: str
    content: str
    publish_time: str
    source: str
    url: str


def normalize_stock_news_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把 AKShare 返回的东方财富新闻表整理成统一字段。

    AKShare 原始字段是中文：
    - 新闻标题
    - 新闻内容
    - 发布时间
    - 文章来源
    - 新闻链接

    这里统一改成英文列名：
    - Symbol
    - Title
    - Content
    - PublishTime
    - Source
    - Url

    这样后面代码更稳定，不容易因为中文字段到处复制而写错。
    """
    if data.empty:
        return pd.DataFrame(
            columns=["Symbol", "Title", "Content", "PublishTime", "Source", "Url"]
        )

    required_columns = ["新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    missing_columns = [column for column in required_columns if column not in data.columns]
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
    """把整理后的 DataFrame 转成新闻对象列表。"""
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
    """获取某只 A 股最近新闻。

    参数说明：
    - symbol：股票代码，比如 600519、000001、300750。
    - max_items：最多返回多少条新闻，默认 10 条。

    数据来源：
    - AKShare 的 stock_news_em。
    - 底层来自东方财富个股新闻。

    注意：
    AKShare 这个接口通常返回最近约 100 条新闻。
    这里用 max_items 截取前几条，避免后面 Prompt 太长。
    """
    normalized_symbol = symbol_mod.normalize_cn_symbol(symbol)
    raw_news = ak.stock_news_em(symbol=normalized_symbol)
    normalized_news = normalize_stock_news_frame(raw_news, normalized_symbol)

    if max_items > 0:
        normalized_news = normalized_news.head(max_items)

    return news_frame_to_items(normalized_news)


def render_stock_news_text(news_items: list[StockNewsItem]) -> str:
    """把新闻列表渲染成适合人和大模型阅读的文本。"""
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


def demo_stock_news() -> None:
    """演示获取个股新闻。"""
    news_items = get_stock_news("002361", max_items=5)
    print(render_stock_news_text(news_items))


if __name__ == "__main__":
    demo_stock_news()
