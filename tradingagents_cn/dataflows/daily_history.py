"""A 股历史日线行情数据层。

这个文件属于正式工程的 dataflows 数据层。

它负责：

1. 通过 AKShare 获取 A 股历史日线行情。
2. 优先使用东方财富历史行情接口。
3. 如果东方财富失败，使用腾讯证券历史行情接口兜底。
4. 把不同接口返回的字段统一成 Date/Open/High/Low/Close/Volume。

它不负责：

1. 计算 MACD、RSI 等技术指标。
2. 调用大模型。
3. 判断买卖点。
4. 生成交易建议。

为什么历史日线重要？

实时行情只能告诉我们“现在是什么价格”。
历史日线才能告诉我们：
    - 最近趋势如何；
    - 是否放量；
    - 是否突破；
    - 技术指标处在什么位置。

后面的 Market Agent 会基于这些历史行情和技术指标做分析。
"""

from __future__ import annotations

import time

import pandas as pd

from tradingagents_cn.dataflows.symbols import (
    normalize_cn_symbol,
    to_akshare_date,
    to_market_prefixed_symbol,
)


COLUMN_MAP = {
    # 东方财富接口常见中文字段。
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    "成交额": "Amount",
    "换手率": "Turnover",
    # 腾讯证券备用接口常见英文字段。
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "amount": "Volume",
}


def normalize_history_frame(data: pd.DataFrame) -> pd.DataFrame:
    """把历史行情表整理成统一 OHLCV 格式。

    输入可能来自东方财富，也可能来自腾讯。
    两个接口字段名不完全一样。

    这里统一输出：
        Date
        Open
        High
        Low
        Close
        Volume

    如果源数据里有成交额和换手率，也保留：
        Amount
        Turnover

    返回值仍然是 DataFrame。
    因为技术指标计算更适合用表格结构。
    """
    if data is None or data.empty:
        raise ValueError("历史行情数据为空。")

    frame = data.rename(
        columns={
            raw_column: normalized_column
            for raw_column, normalized_column in COLUMN_MAP.items()
            if raw_column in data.columns
        }
    )

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(f"历史行情缺少必要字段：{missing_columns}")

    optional_columns = [
        column
        for column in ("Amount", "Turnover")
        if column in frame.columns
    ]

    frame = frame[required_columns + optional_columns].copy()

    # Date 统一转成 pandas datetime。
    # 转换失败的行会变成 NaT，然后被 dropna 删除。
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])

    # 除日期外，其他列都尽量转成数字。
    # 例如字符串 "16.25" 会变成 16.25。
    # 无法转换的值会变成 NaN。
    for column in [column for column in frame.columns if column != "Date"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Close 是后面计算技术指标的核心字段。
    # 如果 Close 为空，这一行没有分析价值。
    frame = frame.dropna(subset=["Close"])

    return frame.sort_values("Date")


def get_a_share_daily_history(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """通过 AKShare 获取 A 股历史日线行情。

    参数：
        symbol:
            股票代码，例如 002361、600519。

        start_date:
            开始日期，格式 YYYY-MM-DD。

        end_date:
            结束日期，格式 YYYY-MM-DD。

        max_retries:
            东方财富接口最大重试次数。

        retry_sleep_seconds:
            每次重试之间等待多少秒。

    当前策略：
        1. 优先使用东方财富接口 stock_zh_a_hist。
        2. 东方财富失败后，使用腾讯接口 stock_zh_a_hist_tx 兜底。

    注意：
        adjust="qfq" 表示前复权。
        做技术指标时，前复权价格通常比不复权更适合连续分析。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return get_daily_history_from_eastmoney(
                normalized_symbol,
                start_date,
                end_date,
            )
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    try:
        return get_daily_history_from_tencent(
            normalized_symbol,
            start_date,
            end_date,
        )
    except Exception as fallback_error:
        raise RuntimeError(
            f"AKShare 获取 {normalized_symbol} 日线行情失败。"
            f"东方财富已重试 {max_retries} 次，最后错误：{last_error}；"
            f"腾讯备用接口错误：{fallback_error}"
        ) from fallback_error


def get_daily_history_from_eastmoney(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """通过 AKShare 东方财富接口获取 A 股历史日线行情。"""
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
    raw = ak.stock_zh_a_hist(
        symbol=normalized_symbol,
        period="daily",
        start_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
        adjust="qfq",
    )
    return normalize_history_frame(raw)


def get_daily_history_from_tencent(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """通过 AKShare 腾讯证券接口获取 A 股历史日线行情。"""
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
    raw = ak.stock_zh_a_hist_tx(
        symbol=to_market_prefixed_symbol(normalized_symbol),
        start_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
        adjust="qfq",
        timeout=15,
    )
    return normalize_history_frame(raw)
