"""A 股技术指标层。

这个文件按照原版 TradingAgents 的 Market Analyst 指标集合来写。

原版 TradingAgents 给 Market Analyst 的指标主要包括：

Moving Averages:
    close_50_sma
    close_200_sma
    close_10_ema

MACD Related:
    macd
    macds
    macdh

Momentum Indicators:
    rsi

Volatility Indicators:
    boll
    boll_ub
    boll_lb
    atr

Volume-Based Indicators:
    vwma
    mfi

注意：
    这里不再加入我自己编的 MA5、MA10、VolumeMA5、ClosePosition 等指标。
    TradingAgents 原项目没有在 Market Analyst 指标目录里列这些，
    所以正式 A 股版也先不加。
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

TRADINGAGENTS_INDICATORS: tuple[str, ...] = (
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
    "mfi",
)


def validate_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    """检查并清洗日线行情表。

    指标计算至少需要：
        Date / Open / High / Low / Close / Volume

    返回值仍然使用大写列名。
    这样 dataflows 层和 verified snapshot 都更容易阅读。
    """
    if data is None or data.empty:
        raise ValueError("行情数据为空，无法计算技术指标。")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段：{missing_columns}")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Close"])
    return frame.sort_values("Date").reset_index(drop=True)


def add_tradingagents_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """一次性计算原版 TradingAgents Market Analyst 指标集合。

    输出新增字段名保持原版风格：
        close_50_sma
        close_200_sma
        close_10_ema
        macd
        macds
        macdh
        rsi
        boll
        boll_ub
        boll_lb
        atr
        vwma
        mfi

    说明：
        原版 TradingAgents 使用 stockstats 计算这些指标。
        这里为了不额外引入新依赖，先用 pandas 按常见公式计算。
        字段名和指标集合保持原版一致。
    """
    frame = validate_price_frame(data)

    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"]

    frame["close_50_sma"] = close.rolling(window=50).mean()
    frame["close_200_sma"] = close.rolling(window=200).mean()
    frame["close_10_ema"] = close.ewm(span=10, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macds"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macdh"] = frame["macd"] - frame["macds"]

    frame["rsi"] = _calculate_rsi(close, window=14)

    boll_middle = close.rolling(window=20).mean()
    boll_std = close.rolling(window=20).std()
    frame["boll"] = boll_middle
    frame["boll_ub"] = boll_middle + 2 * boll_std
    frame["boll_lb"] = boll_middle - 2 * boll_std

    frame["atr"] = _calculate_atr(frame, window=14)

    volume_price = close * volume
    frame["vwma"] = (
        volume_price.rolling(window=20).sum()
        / volume.rolling(window=20).sum()
    )

    frame["mfi"] = _calculate_mfi(frame, window=14)

    return frame


def latest_tradingagents_indicator_snapshot(data: pd.DataFrame) -> dict:
    """提取最新交易日的原版指标集合快照。

    这个函数只返回原版指标目录里的指标。
    不额外返回自定义 MA5、成交量均线、收盘位置等字段。
    """
    if data is None or data.empty:
        raise ValueError("指标数据为空，无法生成指标快照。")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")

    if frame.empty:
        raise ValueError("指标数据没有有效日期，无法生成指标快照。")

    latest = frame.iloc[-1]
    snapshot = {
        "date": latest["Date"].strftime("%Y-%m-%d"),
    }

    for indicator in TRADINGAGENTS_INDICATORS:
        snapshot[indicator] = _to_float_or_none(latest.get(indicator))

    return snapshot


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI。

    这里使用常见 rolling average 版本。
    原版通过 stockstats 计算，数值可能有轻微差异，
    但指标名和用途保持一致。
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window=window).mean()
    average_loss = loss.rolling(window=window).mean()

    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def _calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 ATR。

    True Range 取三者最大值：
        High - Low
        abs(High - PrevClose)
        abs(Low - PrevClose)
    """
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window=window).mean()


def _calculate_mfi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 MFI。

    MFI 使用典型价格和成交量估算资金流。
    """
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    raw_money_flow = typical_price * data["Volume"]

    price_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(price_delta > 0, 0)
    negative_flow = raw_money_flow.where(price_delta < 0, 0)

    positive_sum = positive_flow.rolling(window=window).sum()
    negative_sum = negative_flow.rolling(window=window).sum()

    money_flow_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + money_flow_ratio))


def _to_float_or_none(value) -> float | None:
    """把 pandas/numpy 数值转换成普通 Python float。"""
    if value is None or pd.isna(value):
        return None
    return float(value)
