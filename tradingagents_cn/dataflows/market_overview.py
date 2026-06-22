"""A 股市场概览数据层。

这个文件服务于用户这类问题：

    今天股市怎么样？
    大盘现在什么情况？
    今天市场强不强？

它只负责采集和整理市场全局原材料，不调用大模型。

当前先接入三类材料：

1. 主要指数快照；
2. 全市场涨跌家数；
3. 行业板块强弱排行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


INDEX_NAME_ALIASES = {
    "上证指数": "上证指数",
    "深证成指": "深证成指",
    "创业板指": "创业板指",
    "沪深300": "沪深300",
    "中证500": "中证500",
    "科创50": "科创50",
}


@dataclass
class MarketBreadth:
    """市场涨跌家数。"""

    total_count: int
    up_count: int
    down_count: int
    flat_count: int
    up_ratio: float
    down_ratio: float


@dataclass
class MarketOverview:
    """A 股市场概览材料。"""

    index_snapshot: pd.DataFrame
    market_breadth: MarketBreadth
    sector_snapshot: pd.DataFrame


def get_market_overview(sector_top_n: int = 10) -> MarketOverview:
    """获取 A 股市场概览原材料。"""
    import akshare as ak

    spot = ak.stock_zh_a_spot_em()
    index_raw = ak.stock_zh_index_spot_em()
    sector_raw = ak.stock_board_industry_name_em()

    index_snapshot = normalize_index_snapshot(index_raw)
    market_breadth = calculate_market_breadth(spot)
    sector_snapshot = normalize_sector_snapshot(sector_raw, top_n=sector_top_n)

    return MarketOverview(
        index_snapshot=index_snapshot,
        market_breadth=market_breadth,
        sector_snapshot=sector_snapshot,
    )


def normalize_index_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    """整理主要指数快照。

    AKShare 指数快照字段可能随版本略有变化，
    这里统一输出：
        Name / Latest / ChangePct / Amount
    """
    if data is None or data.empty:
        return pd.DataFrame(columns=["Name", "Latest", "ChangePct", "Amount"])

    name_column = find_first_existing_column(data, ["名称", "name", "指数名称"])
    latest_column = find_first_existing_column(data, ["最新价", "最新", "price"])
    change_pct_column = find_first_existing_column(data, ["涨跌幅", "涨幅", "change_pct"])
    amount_column = find_first_existing_column(data, ["成交额", "amount"])

    if name_column is None:
        return pd.DataFrame(columns=["Name", "Latest", "ChangePct", "Amount"])

    frame = pd.DataFrame()
    frame["Name"] = data[name_column].map(lambda value: str(value).strip())
    frame["Latest"] = data[latest_column].map(to_float_or_none) if latest_column else None
    frame["ChangePct"] = (
        data[change_pct_column].map(to_float_or_none)
        if change_pct_column
        else None
    )
    frame["Amount"] = data[amount_column].map(to_float_or_none) if amount_column else None

    selected_names = set(INDEX_NAME_ALIASES.keys())
    selected = frame[frame["Name"].isin(selected_names)].copy()
    if selected.empty:
        return frame.head(8).reset_index(drop=True)

    return selected.reset_index(drop=True)


def calculate_market_breadth(spot_data: pd.DataFrame) -> MarketBreadth:
    """根据全市场实时行情计算涨跌家数。"""
    if spot_data is None or spot_data.empty:
        return MarketBreadth(
            total_count=0,
            up_count=0,
            down_count=0,
            flat_count=0,
            up_ratio=0.0,
            down_ratio=0.0,
        )

    change_column = find_first_existing_column(spot_data, ["涨跌幅", "涨幅", "change_pct"])
    if change_column is None:
        total = len(spot_data)
        return MarketBreadth(
            total_count=total,
            up_count=0,
            down_count=0,
            flat_count=total,
            up_ratio=0.0,
            down_ratio=0.0,
        )

    changes = spot_data[change_column].map(to_float_or_none).dropna()
    total_count = int(len(changes))
    up_count = int((changes > 0).sum())
    down_count = int((changes < 0).sum())
    flat_count = int((changes == 0).sum())

    if total_count == 0:
        up_ratio = 0.0
        down_ratio = 0.0
    else:
        up_ratio = up_count / total_count
        down_ratio = down_count / total_count

    return MarketBreadth(
        total_count=total_count,
        up_count=up_count,
        down_count=down_count,
        flat_count=flat_count,
        up_ratio=up_ratio,
        down_ratio=down_ratio,
    )


def normalize_sector_snapshot(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """整理行业板块强弱排行。"""
    if data is None or data.empty:
        return pd.DataFrame(columns=["Name", "ChangePct", "Amount", "LeadingStock"])

    name_column = find_first_existing_column(data, ["板块名称", "名称", "name"])
    change_pct_column = find_first_existing_column(data, ["涨跌幅", "涨幅", "change_pct"])
    amount_column = find_first_existing_column(data, ["成交额", "amount"])
    leading_stock_column = find_first_existing_column(
        data,
        ["领涨股票", "领涨股", "leading_stock"],
    )

    if name_column is None:
        return pd.DataFrame(columns=["Name", "ChangePct", "Amount", "LeadingStock"])

    frame = pd.DataFrame()
    frame["Name"] = data[name_column].map(lambda value: str(value).strip())
    frame["ChangePct"] = (
        data[change_pct_column].map(to_float_or_none)
        if change_pct_column
        else None
    )
    frame["Amount"] = data[amount_column].map(to_float_or_none) if amount_column else None
    frame["LeadingStock"] = (
        data[leading_stock_column].map(lambda value: str(value).strip())
        if leading_stock_column
        else ""
    )

    frame = frame.sort_values("ChangePct", ascending=False, na_position="last")
    return frame.head(max(1, int(top_n))).reset_index(drop=True)


def render_market_overview_text(overview: MarketOverview) -> str:
    """把市场概览渲染成适合人和大模型阅读的文本。"""
    return "\n\n".join(
        [
            "# A 股市场概览原材料",
            "## 主要指数",
            render_dataframe_markdown(overview.index_snapshot),
            "## 市场涨跌家数",
            render_market_breadth_text(overview.market_breadth),
            "## 行业板块强弱",
            render_dataframe_markdown(overview.sector_snapshot),
        ]
    )


def render_market_breadth_text(breadth: MarketBreadth) -> str:
    """渲染涨跌家数。"""
    return (
        f"- 统计股票数：{breadth.total_count}\n"
        f"- 上涨家数：{breadth.up_count}（{breadth.up_ratio:.1%}）\n"
        f"- 下跌家数：{breadth.down_count}（{breadth.down_ratio:.1%}）\n"
        f"- 平盘家数：{breadth.flat_count}"
    )


def render_dataframe_markdown(data: pd.DataFrame) -> str:
    """把 DataFrame 渲染成 Markdown。"""
    if data is None or data.empty:
        return "暂无数据。"
    return data.fillna("").to_markdown(index=False)


def find_first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选列名中找到第一个存在的列。"""
    for column in candidates:
        if column in data.columns:
            return column
    return None


def to_float_or_none(value: Any) -> float | None:
    """安全转换 float。"""
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "-", "--", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
