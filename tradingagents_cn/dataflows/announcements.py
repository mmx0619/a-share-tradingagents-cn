"""A 股公告/信息披露数据层。

普通新闻常常是媒体报道或行情异动解读。
公告则是上市公司正式披露材料，权威性更高。

当前优先使用：

    巨潮资讯信息披露公告

如果巨潮接口失败，再尝试：

    东方财富个股公告

这个文件只负责获取和整理公告原材料，不判断利好利空。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


@dataclass
class StockAnnouncementItem:
    """单条 A 股公告。"""

    symbol: str
    stock_name: str
    title: str
    publish_time: str
    source: str
    url: str


def get_stock_announcements(
    symbol: str,
    end_date: str,
    lookback_days: int = 90,
    max_items: int = 10,
    category: str = "",
) -> list[StockAnnouncementItem]:
    """获取某只 A 股最近公告。

    参数：
        symbol:
            6 位 A 股代码。

        end_date:
            截止日期，格式 YYYY-MM-DD。

        lookback_days:
            向前查询多少个自然日。

        max_items:
            最多返回多少条公告。

        category:
            巨潮公告分类，例如 年报、半年报、风险提示、权益分派。
            空字符串表示全部分类。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    start_date = calculate_announcement_start_date(end_date, lookback_days)

    errors: list[str] = []
    for source_name, fetcher in [
        ("巨潮资讯", fetch_cninfo_announcements),
        ("东方财富公告", fetch_eastmoney_announcements),
    ]:
        try:
            frame = fetcher(
                normalized_symbol,
                start_date=start_date,
                end_date=end_date,
                category=category,
            )
            normalized = normalize_announcement_frame(
                frame,
                symbol=normalized_symbol,
                source=source_name,
            )
            items = announcement_frame_to_items(normalized)
            if max_items > 0:
                return items[:max_items]
            return items
        except Exception as error:
            errors.append(f"{source_name}公告获取失败：{error}")

    raise RuntimeError("；".join(errors))


def fetch_cninfo_announcements(
    symbol: str,
    start_date: str,
    end_date: str,
    category: str = "",
) -> pd.DataFrame:
    """调用 AKShare 巨潮资讯公告接口。"""
    import akshare as ak

    return ak.stock_zh_a_disclosure_report_cninfo(
        symbol=normalize_cn_symbol(symbol),
        market="沪深京",
        keyword="",
        category=category,
        start_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
    )


def fetch_eastmoney_announcements(
    symbol: str,
    start_date: str,
    end_date: str,
    category: str = "",
) -> pd.DataFrame:
    """调用 AKShare 东方财富个股公告接口。"""
    import akshare as ak

    return ak.stock_individual_notice_report(
        security=normalize_cn_symbol(symbol),
        symbol=category or "全部",
        begin_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
    )


def normalize_announcement_frame(
    data: pd.DataFrame,
    symbol: str,
    source: str,
) -> pd.DataFrame:
    """把不同公告接口返回值整理成统一字段。"""
    columns = ["Symbol", "StockName", "Title", "PublishTime", "Source", "Url"]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)

    code_column = find_first_existing_column(data, ["代码", "股票代码", "securityCode"])
    name_column = find_first_existing_column(data, ["简称", "股票简称", "名称", "securityName"])
    title_column = find_first_existing_column(data, ["公告标题", "标题", "notice_title", "announcementTitle"])
    time_column = find_first_existing_column(data, ["公告时间", "公告日期", "发布时间", "notice_date"])
    url_column = find_first_existing_column(data, ["公告链接", "链接", "url", "art_code"])

    if title_column is None:
        raise ValueError(f"公告数据缺少标题字段，当前字段：{list(data.columns)}")

    frame = pd.DataFrame()
    frame["Symbol"] = (
        data[code_column].map(lambda value: normalize_cn_symbol(str(value)))
        if code_column
        else [normalize_cn_symbol(symbol)] * len(data)
    )
    frame["StockName"] = data[name_column].fillna("").astype(str) if name_column else ""
    frame["Title"] = data[title_column].fillna("").astype(str)
    frame["PublishTime"] = data[time_column].fillna("").astype(str) if time_column else ""
    frame["Source"] = source
    frame["Url"] = data[url_column].fillna("").astype(str) if url_column else ""

    return frame[columns]


def announcement_frame_to_items(data: pd.DataFrame) -> list[StockAnnouncementItem]:
    """把公告 DataFrame 转成对象列表。"""
    items: list[StockAnnouncementItem] = []
    for row in data.to_dict(orient="records"):
        title = str(row.get("Title") or "").strip()
        if not title:
            continue
        items.append(
            StockAnnouncementItem(
                symbol=str(row.get("Symbol") or ""),
                stock_name=str(row.get("StockName") or ""),
                title=title,
                publish_time=str(row.get("PublishTime") or ""),
                source=str(row.get("Source") or ""),
                url=str(row.get("Url") or ""),
            )
        )
    return items


def render_stock_announcements_text(items: list[StockAnnouncementItem]) -> str:
    """把公告列表渲染成适合大模型阅读的文本。"""
    if not items:
        return (
            "暂无可用公司公告。\n"
            "说明：这代表公告接口没有返回材料，不代表公司没有任何公告。"
        )

    lines = [
        "A 股公司公告/信息披露材料：",
        "",
        "说明：公告属于上市公司正式披露材料，权威性通常高于普通新闻。"
    ]
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                "",
                f"公告 {index}",
                f"股票：{item.stock_name}（{item.symbol}）",
                f"标题：{item.title}",
                f"时间：{item.publish_time}",
                f"来源：{item.source}",
                f"链接：{item.url}",
            ]
        )
    return "\n".join(lines)


def calculate_announcement_start_date(end_date: str, lookback_days: int) -> str:
    """根据截止日期计算公告查询开始日期。"""
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=max(1, int(lookback_days)))
    return start.strftime("%Y-%m-%d")


def to_akshare_date(date_text: str) -> str:
    """把 YYYY-MM-DD 转成 AKShare 常用 YYYYMMDD。"""
    return str(date_text).replace("-", "")


def find_first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选字段名里找第一个存在的字段。"""
    for column in candidates:
        if column in data.columns:
            return column
    return None
