"""A 股实时/近实时行情数据层。

这个文件属于正式工程的 dataflows 数据层。

它负责：

1. 调用 AKShare 获取 A 股实时/近实时行情快照。
2. 优先使用东方财富行情源。
3. 如果东方财富失败，再使用新浪行情源兜底。
4. 把不同来源的数据统一成 RealtimeQuote 对象。
5. 把行情对象渲染成适合人和大模型阅读的文本。

它不负责：

1. 调用大模型。
2. 判断买入、卖出、观望。
3. 做风控。
4. 生成最终投资报告。

重要说明：

这里的“实时”指公开网站提供的实时/近实时快照。
它不是券商交易柜台的低延迟行情，
也不是逐笔成交数据。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


@dataclass
class RealtimeQuote:
    """单只股票的实时行情快照。

    symbol:
        6 位股票代码，例如 002361。

    name:
        股票名称。

    latest_price:
        最新价。

    change_amount:
        涨跌额。

    change_pct:
        涨跌幅，单位是百分比。

    open_price:
        今开。

    previous_close:
        昨收。

    high_price:
        盘中最高价。

    low_price:
        盘中最低价。

    volume:
        成交量。

    amount:
        成交额。

    turnover_rate:
        换手率，单位是百分比。
        有些数据源可能没有。

    volume_ratio:
        量比。
        有些数据源可能没有。

    update_time:
        行情更新时间。

    source:
        数据来源，例如 eastmoney 或 sina。
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
    """把接口返回值安全转换成 float。

    AKShare 返回的数据里可能出现：
        None
        空字符串
        "-"
        "--"
        pandas 的 NaN

    这些都不能直接 float(value)。
    所以这里统一转成 None。
    """
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

    不同接口返回的股票代码格式可能不一样。

    例如新浪可能返回：
        sh600519
        sz002361
        bj430047

    东方财富通常返回：
        600519
        002361

    后面的匹配逻辑统一只认 6 位数字。
    """
    text = str(raw_symbol or "").strip().lower()

    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix):
            text = text[2:]
            break

    return normalize_cn_symbol(text)


def get_realtime_quote_from_eastmoney(symbol: str) -> RealtimeQuote:
    """从东方财富获取单只股票实时行情快照。

    AKShare 的 stock_zh_a_spot_em() 会返回全市场快照。
    所以这里的流程是：

        1. 拉取全市场实时行情表；
        2. 把表里的股票代码统一成 6 位数字；
        3. 筛选出目标股票；
        4. 转成 RealtimeQuote。

    优点：
        字段相对完整，包含换手率、量比等。

    缺点：
        每次会拉全市场数据，速度不一定很快。
    """
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
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

    这个函数作为东方财富失败后的兜底源。

    新浪接口字段比东方财富少一点，
    所以 turnover_rate、volume_ratio 暂时填 None。
    """
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
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
    """获取单只股票实时/近实时行情快照。

    当前策略：

        1. 优先尝试东方财富。
        2. 东方财富失败时，按 max_retries 重试。
        3. 重试后仍失败，则使用新浪兜底。

    为什么要重试？

    AKShare 背后访问的是公开网站接口。
    公开接口偶尔会网络抖动、超时、字段短暂变化。
    如果一次失败就结束，Agent 工作流会很脆弱。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
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
    """把实时行情快照渲染成适合人和大模型阅读的文本。

    工具函数最终要返回给大模型。
    大模型更适合读清楚的文本，
    而不是直接读 Python dataclass 或 DataFrame。
    """
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
