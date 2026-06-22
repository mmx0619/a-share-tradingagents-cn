"""第 19 步：获取 A 股实时行情快照。

前面的行情数据主要是日线历史数据：

- 开盘价
- 最高价
- 最低价
- 收盘价
- 成交量

这些适合做技术指标和盘后分析，
但不等于盘中实时行情。

当前文件补上“实时行情快照”：

股票代码
  ↓
东方财富实时行情快照
  ↓
如果东方财富失败，使用新浪实时行情兜底
  ↓
统一成 RealtimeQuote

注意：
这里的“实时”是公开网站接口的实时/近实时快照，
不是券商交易柜台的低延迟行情，也不是逐笔成交流。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

import step01_akshare_cn as symbol_mod


@dataclass
class RealtimeQuote:
    """单只股票的实时行情快照。

    字段说明：
    - symbol：6 位股票代码。
    - name：股票名称。
    - latest_price：最新价。
    - change_amount：涨跌额。
    - change_pct：涨跌幅，单位是百分比。
    - open_price：今开。
    - previous_close：昨收。
    - high_price：最高价。
    - low_price：最低价。
    - volume：成交量。
    - amount：成交额。
    - turnover_rate：换手率，单位是百分比，部分数据源可能没有。
    - volume_ratio：量比，部分数据源可能没有。
    - update_time：行情更新时间。
    - source：数据源。
    """

    symbol: str
    name: str | None
    latest_price: float | None
    change_amount: float | None
    change_pct: float | None
    open_price: float | None
    previous_close: float | None
    high_price: float | None
    low_price: float | None
    volume: float | None
    amount: float | None
    turnover_rate: float | None
    volume_ratio: float | None
    update_time: str | None
    source: str


def to_float(value: Any) -> float | None:
    """把接口返回值安全转换成 float。"""
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


def normalize_quote_symbol(raw_symbol: Any) -> str:
    """把实时行情表里的代码统一成 6 位股票代码。

    新浪可能返回 bj920000、sh600519、sz002361。
    东方财富通常返回 600519、002361。
    """
    text = str(raw_symbol or "").strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix):
            text = text[2:]
            break
    return symbol_mod.normalize_cn_symbol(text)


def get_realtime_quote_from_eastmoney(symbol: str) -> RealtimeQuote:
    """从东方财富获取单只股票实时行情快照。

    AKShare 的 stock_zh_a_spot_em 会返回全市场快照。
    这里再按股票代码筛选出目标股票。
    """
    import akshare as ak

    normalized_symbol = symbol_mod.normalize_cn_symbol(symbol)
    data = ak.stock_zh_a_spot_em()
    if data.empty:
        raise ValueError("东方财富实时行情返回空表。")

    if "代码" not in data.columns:
        raise ValueError(f"东方财富实时行情缺少代码字段：{list(data.columns)}")

    data = data.copy()
    data["NormalizedSymbol"] = data["代码"].map(normalize_quote_symbol)
    matched = data[data["NormalizedSymbol"] == normalized_symbol]
    if matched.empty:
        raise ValueError(f"东方财富实时行情中找不到股票：{normalized_symbol}")

    row = matched.iloc[0]
    return RealtimeQuote(
        symbol=normalized_symbol,
        name=str(row.get("名称")) if "名称" in row else None,
        latest_price=to_float(row.get("最新价")),
        change_amount=to_float(row.get("涨跌额")),
        change_pct=to_float(row.get("涨跌幅")),
        open_price=to_float(row.get("今开")),
        previous_close=to_float(row.get("昨收")),
        high_price=to_float(row.get("最高")),
        low_price=to_float(row.get("最低")),
        volume=to_float(row.get("成交量")),
        amount=to_float(row.get("成交额")),
        turnover_rate=to_float(row.get("换手率")),
        volume_ratio=to_float(row.get("量比")),
        update_time=str(row.get("更新时间")) if "更新时间" in row else None,
        source="eastmoney",
    )


def get_realtime_quote_from_sina(symbol: str) -> RealtimeQuote:
    """从新浪获取单只股票实时行情快照。

    新浪接口也会返回全市场快照。
    它速度相对慢一些，而且 AKShare 文档提示不要频繁调用。
    所以这里只作为东方财富失败后的备用源。
    """
    import akshare as ak

    normalized_symbol = symbol_mod.normalize_cn_symbol(symbol)
    data = ak.stock_zh_a_spot()
    if data.empty:
        raise ValueError("新浪实时行情返回空表。")

    if "代码" not in data.columns:
        raise ValueError(f"新浪实时行情缺少代码字段：{list(data.columns)}")

    data = data.copy()
    data["NormalizedSymbol"] = data["代码"].map(normalize_quote_symbol)
    matched = data[data["NormalizedSymbol"] == normalized_symbol]
    if matched.empty:
        raise ValueError(f"新浪实时行情中找不到股票：{normalized_symbol}")

    row = matched.iloc[0]
    return RealtimeQuote(
        symbol=normalized_symbol,
        name=str(row.get("名称")) if "名称" in row else None,
        latest_price=to_float(row.get("最新价")),
        change_amount=to_float(row.get("涨跌额")),
        change_pct=to_float(row.get("涨跌幅")),
        open_price=to_float(row.get("今开")),
        previous_close=to_float(row.get("昨收")),
        high_price=to_float(row.get("最高")),
        low_price=to_float(row.get("最低")),
        volume=to_float(row.get("成交量")),
        amount=to_float(row.get("成交额")),
        turnover_rate=None,
        volume_ratio=None,
        update_time=str(row.get("时间戳")) if "时间戳" in row else None,
        source="sina",
    )


def get_realtime_quote(
    symbol: str,
    max_retries: int = 2,
    retry_sleep_seconds: float = 1.5,
) -> RealtimeQuote:
    """获取单只股票实时行情快照。

    当前策略：
    1. 优先尝试东方财富。
    2. 如果东方财富失败，重试几次。
    3. 仍然失败，则用新浪兜底。

    注意：
    这是公开接口快照，不保证毫秒级实时。
    """
    normalized_symbol = symbol_mod.normalize_cn_symbol(symbol)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return get_realtime_quote_from_eastmoney(normalized_symbol)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    try:
        return get_realtime_quote_from_sina(normalized_symbol)
    except Exception as fallback_error:
        raise RuntimeError(
            f"获取 {normalized_symbol} 实时行情失败。"
            f"东方财富最后错误：{last_error}；"
            f"新浪备用错误：{fallback_error}"
        ) from fallback_error


def render_realtime_quote_text(quote: RealtimeQuote) -> str:
    """把实时行情快照渲染成适合人和大模型阅读的文本。"""
    return f"""股票代码：{quote.symbol}
股票名称：{quote.name}
数据源：{quote.source}
更新时间：{quote.update_time}

最新价：{quote.latest_price}
涨跌额：{quote.change_amount}
涨跌幅：{quote.change_pct}%
今开：{quote.open_price}
昨收：{quote.previous_close}
最高：{quote.high_price}
最低：{quote.low_price}
成交量：{quote.volume}
成交额：{quote.amount}
换手率：{quote.turnover_rate}
量比：{quote.volume_ratio}

说明：这是公开网站实时/近实时行情快照，不是逐笔成交流，也不是券商低延迟行情。
"""


def demo_realtime_quote() -> None:
    """演示获取实时行情快照。"""
    quote = get_realtime_quote("002361")
    print(render_realtime_quote_text(quote))


if __name__ == "__main__":
    demo_realtime_quote()
