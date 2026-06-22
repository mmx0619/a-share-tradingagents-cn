"""第 02 步：基于 A 股日线行情计算技术指标。

这个文件只负责“计算指标”，不负责获取行情。

也就是说：
- 第 01 个文件 `step01_akshare_cn.py` 负责从 AKShare 获取日线数据。
- 当前文件负责接收日线 DataFrame，然后在表格里新增技术指标列。

这样拆分的好处是：
1. 获取数据和计算指标互不耦合。
2. 以后如果数据源从 AKShare 换成 Tushare，这个文件不用改。
3. 大模型 Agent 后面只需要读取已经算好的指标结果，不需要自己算。
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    """检查并清洗日线行情表。

    输入表格至少需要包含这些列：
    - Date：交易日期
    - Open：开盘价
    - High：最高价
    - Low：最低价
    - Close：收盘价
    - Volume：成交量

    为什么要单独做校验：
    技术指标计算非常依赖数据质量。如果缺少 Close 或 Volume，
    后面的均线、涨跌幅、量能指标都会算错。
    """
    if data is None or data.empty:
        raise ValueError("行情数据为空，无法计算技术指标。")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"行情数据缺少必要字段：{missing}")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])

    # 把价格和成交量字段统一转成数字。
    # 如果某个单元格是字符串、空值或异常字符，to_numeric 会转成 NaN。
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Close 是最核心的价格字段；没有收盘价的行直接丢弃。
    frame = frame.dropna(subset=["Close"])

    # 按日期升序排列，确保 rolling 均线是从过去滚到现在。
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


def add_moving_averages(
    data: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """添加收盘价移动平均线。

    移动平均线的含义：
    - MA5：最近 5 个交易日平均收盘价，偏短线。
    - MA10：最近 10 个交易日平均收盘价，偏中短线。
    - MA20：最近 20 个交易日平均收盘价，常用来观察月线级别趋势。

    参数 windows 可以自由扩展，比如传入 (5, 10, 20, 60)。
    """
    frame = data.copy()
    for window in windows:
        # 计算收盘价均线：MA5 / MA10 / MA20 = 最近 N 个交易日收盘价的平均值。
        frame[f"MA{window}"] = frame["Close"].rolling(window=window).mean()
    return frame


def add_return_columns(data: pd.DataFrame) -> pd.DataFrame:
    """添加涨跌幅相关字段。

    新增字段：
    - PrevClose：上一交易日收盘价。
    - Change：今日收盘价 - 昨日收盘价。
    - ChangePct：今日涨跌幅，单位是百分比。

    注意：
    第一行没有上一交易日，所以 PrevClose、Change、ChangePct 会是空值。
    """
    frame = data.copy()
    # 计算昨收价：把 Close 整列往下移动一行，今天这一行拿到的就是上一交易日收盘价。
    frame["PrevClose"] = frame["Close"].shift(1)
    # 计算涨跌额：今日收盘价 - 昨日收盘价。
    frame["Change"] = frame["Close"] - frame["PrevClose"]
    # 计算涨跌幅百分比：涨跌额 / 昨日收盘价 * 100。
    frame["ChangePct"] = frame["Change"] / frame["PrevClose"] * 100
    return frame


def add_volume_indicators(
    data: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """添加成交量均线。

    成交量均线用于观察量能变化：
    - VolumeMA5：最近 5 个交易日平均成交量。
    - VolumeMA10：最近 10 个交易日平均成交量。

    如果今日成交量明显大于 VolumeMA5，通常说明短期交易活跃度提高。
    """
    frame = data.copy()
    for window in windows:
        # 计算成交量均线：VolumeMA5 / VolumeMA10 = 最近 N 个交易日成交量的平均值。
        frame[f"VolumeMA{window}"] = frame["Volume"].rolling(window=window).mean()
    return frame


def add_price_position_columns(data: pd.DataFrame) -> pd.DataFrame:
    """添加价格位置字段。

    新增字段：
    - IntradayAmplitudePct：日内振幅百分比，公式是 (最高价 - 最低价) / 昨收。
    - ClosePosition：收盘价在当日 K 线中的位置。

    ClosePosition 的理解：
    - 接近 1：收盘接近最高价，说明尾盘较强。
    - 接近 0：收盘接近最低价，说明尾盘较弱。
    - 接近 0.5：收盘在日内中间位置。
    """
    frame = data.copy()

    if "PrevClose" not in frame.columns:
        # 如果前面还没有算昨收价，这里补算一次，供日内振幅使用。
        frame["PrevClose"] = frame["Close"].shift(1)

    # 计算日内振幅百分比：(最高价 - 最低价) / 昨日收盘价 * 100。
    frame["IntradayAmplitudePct"] = (frame["High"] - frame["Low"]) / frame["PrevClose"] * 100

    price_range = frame["High"] - frame["Low"]
    # 计算收盘位置：收盘价在当天最低价到最高价之间的位置，越接近 1 说明收盘越靠近最高价。
    frame["ClosePosition"] = (frame["Close"] - frame["Low"]) / price_range

    # 如果最高价等于最低价，price_range 为 0，会导致无穷或空值。
    # 这种极端情况直接填 0.5，表示收盘位于当日价格区间中间。
    frame["ClosePosition"] = frame["ClosePosition"].replace([float("inf"), float("-inf")], pd.NA)
    frame["ClosePosition"] = frame["ClosePosition"].fillna(0.5)
    return frame


def build_basic_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """一次性计算当前阶段需要的基础技术指标。

    这是当前文件最重要的入口函数。

    输入：
    - 第 01 步获取到的 A 股日线行情 DataFrame。

    输出：
    - 新增了均线、涨跌幅、成交量均线、价格位置等字段的 DataFrame。

    目前先做基础指标，不急着上 MACD / RSI。
    原因是：
    1. 先把数据流跑通。
    2. 基础指标最容易检查对错。
    3. 后面接大模型时，基础指标已经足够用于第一版技术面分析。
    """
    frame = validate_price_frame(data)
    frame = add_return_columns(frame)
    frame = add_moving_averages(frame)
    frame = add_volume_indicators(frame)
    frame = add_price_position_columns(frame)
    return frame


def latest_indicator_summary(data: pd.DataFrame) -> dict:
    """把最后一个交易日的关键指标整理成字典。

    为什么需要这个函数：
    后面接入大模型 Agent 时，不一定要把完整表格都塞进 Prompt。
    很多时候只需要把最新一日的关键指标摘要给模型即可。

    返回示例：
    {
        "date": "2024-01-10",
        "close": 100.5,
        "change_pct": 1.23,
        "ma5": 99.8,
        "ma10": 98.7,
        "volume": 123456,
        "close_position": 0.82
    }
    """
    if data is None or data.empty:
        raise ValueError("指标数据为空，无法生成摘要。")

    # 取日期最新的一行，作为“最新交易日”的指标摘要。
    latest = data.sort_values("Date").iloc[-1]
    return {
        # 最新交易日日期。
        "date": latest["Date"].strftime("%Y-%m-%d"),
        # 最新收盘价。
        "close": _to_float_or_none(latest.get("Close")),
        # 最新涨跌幅百分比。
        "change_pct": _to_float_or_none(latest.get("ChangePct")),
        # 最新 5 日、10 日、20 日收盘价均线。
        "ma5": _to_float_or_none(latest.get("MA5")),
        "ma10": _to_float_or_none(latest.get("MA10")),
        "ma20": _to_float_or_none(latest.get("MA20")),
        # 最新成交量，以及 5 日成交量均线。
        "volume": _to_float_or_none(latest.get("Volume")),
        "volume_ma5": _to_float_or_none(latest.get("VolumeMA5")),
        # 最新收盘位置：越接近 1，表示收盘越靠近当天最高价。
        "close_position": _to_float_or_none(latest.get("ClosePosition")),
    }


def _to_float_or_none(value) -> float | None:
    """把 pandas/numpy 数值转换成普通 Python float。

    这样做是为了后面更容易转成 JSON。
    如果值为空，就返回 None。
    """
    if pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    # 这里放一小段本地演示数据，方便直接运行当前文件检查效果。
    # 注意：这里不联网，也不调用 AKShare。
    demo = pd.DataFrame(
        [
            {"Date": "2024-01-02", "Open": 10.0, "High": 10.5, "Low": 9.8, "Close": 10.2, "Volume": 1000},
            {"Date": "2024-01-03", "Open": 10.2, "High": 10.8, "Low": 10.1, "Close": 10.6, "Volume": 1200},
            {"Date": "2024-01-04", "Open": 10.6, "High": 10.7, "Low": 10.0, "Close": 10.1, "Volume": 1500},
            {"Date": "2024-01-05", "Open": 10.1, "High": 10.4, "Low": 9.9, "Close": 10.3, "Volume": 1100},
            {"Date": "2024-01-08", "Open": 10.3, "High": 10.9, "Low": 10.2, "Close": 10.8, "Volume": 1800},
        ]
    )

    indicators = build_basic_technical_indicators(demo)
    print(indicators)
    print(latest_indicator_summary(indicators))
