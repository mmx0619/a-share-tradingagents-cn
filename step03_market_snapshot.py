"""第 03 步：把行情和技术指标整理成“市场快照”。

前两个文件分别做了：
- step01_akshare_cn.py：获取 A 股日线行情。
- step02_technical_indicators.py：基于日线行情计算技术指标。

当前文件做第三件事：
把计算好的指标表，整理成适合后续大模型 Agent 阅读的摘要。

为什么需要市场快照：
大模型不适合直接阅读一大张完整 K 线表。
更好的方式是先用代码整理出关键字段，比如：
- 最新收盘价
- 当日涨跌幅
- 是否站上 MA5 / MA10 / MA20
- 成交量是否放大
- 收盘位置强弱

然后再把这些摘要交给大模型分析。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class MarketSnapshot:
    """市场快照数据结构。

    dataclass 的作用：
    它可以让我们用固定字段保存市场摘要，后面也方便转换成 dict 或 JSON。

    这个结构暂时只放技术面最基础的字段。
    后续可以继续扩展：
    - 板块信息
    - 资金流
    - 换手率
    - 涨停/跌停状态
    - 北向资金
    """

    symbol: str
    trade_date: str
    close: float | None
    change_pct: float | None
    volume: float | None
    ma5: float | None
    ma10: float | None
    ma20: float | None
    volume_ma5: float | None
    close_position: float | None
    price_vs_ma5: str
    price_vs_ma10: str
    price_vs_ma20: str
    volume_signal: str
    intraday_close_signal: str


def build_market_snapshot(symbol: str, indicator_data: pd.DataFrame) -> MarketSnapshot:
    """从技术指标表中提取最新交易日，生成市场快照。

    参数：
    - symbol：股票代码，比如 600519。
    - indicator_data：第 02 步计算完技术指标后的 DataFrame。

    返回：
    - MarketSnapshot 对象。

    注意：
    这个函数默认 indicator_data 已经包含以下字段：
    Date、Close、ChangePct、Volume、MA5、MA10、MA20、VolumeMA5、ClosePosition。
    """
    if indicator_data is None or indicator_data.empty:
        raise ValueError("技术指标数据为空，无法生成市场快照。")

    frame = indicator_data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date")

    if frame.empty:
        raise ValueError("技术指标数据没有有效日期，无法生成市场快照。")

    latest = frame.iloc[-1]

    close = _to_float_or_none(latest.get("Close"))
    ma5 = _to_float_or_none(latest.get("MA5"))
    ma10 = _to_float_or_none(latest.get("MA10"))
    ma20 = _to_float_or_none(latest.get("MA20"))
    volume = _to_float_or_none(latest.get("Volume"))
    volume_ma5 = _to_float_or_none(latest.get("VolumeMA5"))
    close_position = _to_float_or_none(latest.get("ClosePosition"))

    return MarketSnapshot(
        symbol=symbol,
        trade_date=latest["Date"].strftime("%Y-%m-%d"),
        close=close,
        change_pct=_to_float_or_none(latest.get("ChangePct")),
        volume=volume,
        ma5=ma5,
        ma10=ma10,
        ma20=ma20,
        volume_ma5=volume_ma5,
        close_position=close_position,
        price_vs_ma5=_compare_price_to_ma(close, ma5, "MA5"),
        price_vs_ma10=_compare_price_to_ma(close, ma10, "MA10"),
        price_vs_ma20=_compare_price_to_ma(close, ma20, "MA20"),
        volume_signal=_judge_volume_signal(volume, volume_ma5),
        intraday_close_signal=_judge_close_position(close_position),
    )


def snapshot_to_dict(snapshot: MarketSnapshot) -> dict:
    """把 MarketSnapshot 对象转换成普通字典。

    为什么需要这个函数：
    后面如果要存 SQLite、写 JSON、传给大模型 Prompt，
    字典格式会比 dataclass 对象更通用。
    """
    return asdict(snapshot)


def render_market_snapshot_text(snapshot: MarketSnapshot) -> str:
    """把市场快照渲染成大模型容易阅读的中文文本。

    注意：
    这一步只是整理事实和初步信号，不做买卖建议。
    买卖建议应该留给后面的 Agent。
    """
    lines = [
        f"股票代码：{snapshot.symbol}",
        f"交易日期：{snapshot.trade_date}",
        f"收盘价：{_format_number(snapshot.close)}",
        f"涨跌幅：{_format_percent(snapshot.change_pct)}",
        f"成交量：{_format_number(snapshot.volume)}",
        "",
        "均线状态：",
        f"- {snapshot.price_vs_ma5}",
        f"- {snapshot.price_vs_ma10}",
        f"- {snapshot.price_vs_ma20}",
        "",
        "量能状态：",
        f"- {snapshot.volume_signal}",
        "",
        "日内收盘位置：",
        f"- {snapshot.intraday_close_signal}",
    ]
    return "\n".join(lines)


def _compare_price_to_ma(close: float | None, ma_value: float | None, ma_name: str) -> str:
    """判断收盘价相对某条均线的位置。"""
    if close is None or ma_value is None:
        return f"{ma_name} 数据不足，暂时无法判断。"

    if close > ma_value:
        return f"收盘价高于 {ma_name}，短期价格相对强于该均线。"

    if close < ma_value:
        return f"收盘价低于 {ma_name}，短期价格相对弱于该均线。"

    return f"收盘价接近 {ma_name}，价格正在该均线附近震荡。"


def _judge_volume_signal(volume: float | None, volume_ma5: float | None) -> str:
    """根据当前成交量和 5 日成交量均线，粗略判断量能状态。"""
    if volume is None or volume_ma5 is None or volume_ma5 == 0:
        return "成交量数据不足，暂时无法判断量能。"

    ratio = volume / volume_ma5

    if ratio >= 1.5:
        return f"成交量明显放大，约为 5 日均量的 {ratio:.2f} 倍。"

    if ratio >= 1.1:
        return f"成交量温和放大，约为 5 日均量的 {ratio:.2f} 倍。"

    if ratio <= 0.7:
        return f"成交量明显缩小，约为 5 日均量的 {ratio:.2f} 倍。"

    return f"成交量接近 5 日均量，量能变化不明显，比例约为 {ratio:.2f}。"


def _judge_close_position(close_position: float | None) -> str:
    """根据收盘价在当日 K 线中的位置，粗略判断尾盘强弱。

    close_position 的含义：
    - 1 表示收盘接近最高价。
    - 0 表示收盘接近最低价。
    - 0.5 表示收盘在最高价和最低价中间。
    """
    if close_position is None:
        return "收盘位置数据不足，暂时无法判断日内强弱。"

    if close_position >= 0.8:
        return "收盘接近日内高位，说明尾盘表现偏强。"

    if close_position <= 0.2:
        return "收盘接近日内低位，说明尾盘表现偏弱。"

    return "收盘位于日内中部区域，尾盘强弱不极端。"


def _to_float_or_none(value) -> float | None:
    """把 pandas/numpy 数值转换成普通 Python float，空值返回 None。"""
    if pd.isna(value):
        return None
    return float(value)


def _format_number(value: float | None) -> str:
    """把数字格式化成适合阅读的文本。"""
    if value is None:
        return "数据不足"
    return f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    """把百分比数字格式化成适合阅读的文本。"""
    if value is None:
        return "数据不足"
    return f"{value:.2f}%"


if __name__ == "__main__":
    # 本地演示数据：这里假设第 02 步已经算好了技术指标。
    # 这个演示不联网，也不调用 AKShare。
    demo = pd.DataFrame(
        [
            {
                "Date": "2024-01-10",
                "Close": 10.8,
                "ChangePct": 2.86,
                "Volume": 1800,
                "MA5": 10.4,
                "MA10": 10.2,
                "MA20": None,
                "VolumeMA5": 1320,
                "ClosePosition": 0.86,
            }
        ]
    )

    snapshot = build_market_snapshot("600519", demo)
    print(snapshot_to_dict(snapshot))
    print()
    print(render_market_snapshot_text(snapshot))
