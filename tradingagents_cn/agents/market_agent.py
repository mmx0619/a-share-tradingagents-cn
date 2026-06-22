"""A 股市场分析 Agent。

这个文件按照原版 TradingAgents 的思路来写。

重点：
    Market Agent 不在代码里写死“成交量缩小就一定说明什么”。

原版 TradingAgents 的做法更接近：

    1. 给大模型一组可选技术指标。
    2. 让大模型选择最相关的指标。
    3. 通过工具获取行情和指标数据。
    4. 再给大模型一份“已校验市场数据快照”。
    5. 要求大模型基于工具结果写市场分析报告。

所以这里的代码只负责：

    - 准备市场分析提示词；
    - 准备确定性的行情/指标快照；
    - 把可用指标列表告诉模型；
    - 明确要求模型不要编造具体价格和指标值。

代码不负责：

    - 用自定义阈值判断放量/缩量；
    - 直接生成买入、卖出结论；
    - 把经验规则伪装成市场定理。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents_cn.dataflows.realtime_quote import (
    RealtimeQuote,
    render_realtime_quote_text,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


MARKET_INDICATOR_CATALOG = """均线类指标：
- close_50_sma: 50 日简单移动平均线，用于观察中期趋势和动态支撑/阻力。
- close_200_sma: 200 日简单移动平均线，用于观察长期趋势。
- close_10_ema: 10 日指数移动平均线，对短期价格变化更敏感。

MACD 相关指标：
- macd: MACD 快线，用于观察趋势动量变化。
- macds: MACD 信号线，也就是慢线，用于和 MACD 快线比较。
- macdh: MACD 柱状图，用于观察快慢线差值变化。

动量类指标：
- rsi: RSI 动量指标，用于观察超买/超卖和动量状态。

波动率类指标：
- boll: 布林带中轨，通常是 20 日均线。
- boll_ub: 布林带上轨。
- boll_lb: 布林带下轨。
- atr: ATR 波动率指标，用于观察市场波动程度。

成交量相关指标：
- vwma: 成交量加权移动平均线，用价格和成交量共同确认趋势。
- mfi: 资金流量指标，结合价格和成交量观察资金流强弱。
"""


MARKET_ANALYST_SYSTEM_PROMPT = """你是 A 股市场分析师，任务是分析股票的市场技术面。

你需要从下面的指标列表中选择最相关的指标，最多选择 8 个。
选择时要尽量互补，避免重复。例如不要只选一堆表达相似含义的指标。

可选指标如下：

{indicator_catalog}

分析要求：

1. 先基于工具和快照提供的数据进行分析。
2. 具体开盘价、最高价、最低价、收盘价、成交量和指标数值，必须以“已校验市场数据快照”为准。
3. 如果不同数据源之间出现冲突，要明确指出冲突，不要自己编一个折中数字。
4. 不要声称某个支撑位、压力位、历史验证或精确涨跌幅，除非工具输出里有明确日期和价格支持。
5. 写详细、细致、可操作的趋势观察，但不要直接给最终买入/卖出结论。
6. 报告最后用 Markdown 表格整理关键点。
7. 本报告用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class MarketAgentContext:
    """Market Agent 的输入上下文。

    symbol:
        6 位股票代码。

    trade_date:
        当前分析日期。

    verified_snapshot:
        确定性的市场快照。
        它是模型说具体价格和指标值时的事实依据。

    realtime_quote_text:
        可选实时行情文本。

    prompt:
        最终准备给大模型的完整提示词。
    """

    symbol: str
    trade_date: str
    verified_snapshot: str
    realtime_quote_text: str | None
    prompt: str


def build_verified_market_snapshot_text(
    symbol: str,
    trade_date: str,
    indicator_data: pd.DataFrame,
    look_back_days: int = 30,
) -> str:
    """构造确定性的市场数据快照。

    这个函数对应原版 TradingAgents 里的“已校验市场数据快照”思路。

    它的目的不是替模型分析，
    而是给模型一个事实锚点，避免模型编造具体价格和指标值。
    """
    normalized_symbol = normalize_cn_symbol(symbol)

    if indicator_data is None or indicator_data.empty:
        raise ValueError("指标数据为空，无法生成已校验市场数据快照。")

    frame = indicator_data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame[frame["Date"] <= pd.to_datetime(trade_date)]
    frame = frame.sort_values("Date")

    if frame.empty:
        raise ValueError(f"{normalized_symbol} 在 {trade_date} 之前没有可用行情数据。")

    latest = frame.iloc[-1]
    recent = frame.tail(max(1, min(int(look_back_days), 30)))

    indicator_fields = [
        "close_10_ema",
        "close_50_sma",
        "close_200_sma",
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
    ]

    lines = [
        f"## {normalized_symbol} 的已校验市场数据快照",
        "",
        f"- 请求分析日期：{trade_date}",
        f"- 实际使用的最近交易日：{_format_value(latest.get('Date'))}",
        "- 已排除请求分析日期之后的数据，避免未来数据泄漏。",
        "",
        "### 最新已校验开高低收量行情",
        "",
        "| 字段 | 数值 |",
        "|---|---:|",
    ]

    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_format_value(latest.get(field))} |")

    lines += [
        "",
        "### 最新交易日已校验技术指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]

    for field in indicator_fields:
        if field in latest.index:
            lines.append(f"| {field} | {_format_value(latest.get(field))} |")

    lines += [
        "",
        f"### 最近已校验收盘价（最近 {len(recent)} 行）",
        "",
        "| 日期 | 收盘价 |",
        "|---|---:|",
    ]

    for _, row in recent.iterrows():
        lines.append(f"| {_format_value(row.get('Date'))} | {_format_value(row.get('Close'))} |")

    lines += [
        "",
        "请把这份快照作为精确开盘价、最高价、最低价、收盘价、成交量、价格水平和技术指标数值的事实依据。",
        "如果其他工具输出与这份快照冲突，需要明确指出冲突，不能自行编造折中数值。",
        "不要声称某个支撑位、压力位、历史验证或精确涨跌幅，",
        "除非工具输出中有具体日期和价格作为直接证据。",
    ]

    return "\n".join(lines)


def build_market_agent_context(
    symbol: str,
    trade_date: str,
    indicator_data: pd.DataFrame,
    realtime_quote: RealtimeQuote | None = None,
) -> MarketAgentContext:
    """构造 Market Agent 上下文。

    这个函数只准备 Prompt 和事实快照。
    真正的市场分析由后续大模型完成。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    verified_snapshot = build_verified_market_snapshot_text(
        symbol=normalized_symbol,
        trade_date=trade_date,
        indicator_data=indicator_data,
    )

    realtime_quote_text = None
    if realtime_quote is not None:
        realtime_quote_text = render_realtime_quote_text(realtime_quote)

    prompt = build_market_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        verified_snapshot=verified_snapshot,
        realtime_quote_text=realtime_quote_text,
    )

    return MarketAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        verified_snapshot=verified_snapshot,
        realtime_quote_text=realtime_quote_text,
        prompt=prompt,
    )


def build_market_agent_prompt(
    symbol: str,
    trade_date: str,
    verified_snapshot: str,
    realtime_quote_text: str | None = None,
) -> str:
    """构造给大模型的 Market Agent Prompt。

    这里借鉴原版 TradingAgents：

    - 给模型指标目录；
    - 要求模型选择互补指标；
    - 要求模型以已校验市场数据快照为具体数值依据；
    - 要求输出详细报告和 Markdown 表格。
    """
    realtime_section = realtime_quote_text or "未提供实时行情快照。"

    return f"""{MARKET_ANALYST_SYSTEM_PROMPT.format(indicator_catalog=MARKET_INDICATOR_CATALOG)}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是确定性的市场数据快照，具体价格、成交量和指标数值必须以它为准：

{verified_snapshot}

下面是可选的实时/近实时行情快照：

{realtime_section}

请基于以上信息撰写市场技术面分析报告。
"""


def render_market_agent_context(context: MarketAgentContext) -> str:
    """渲染 Market Agent 上下文，方便调试时阅读。"""
    return context.prompt


def _format_value(value) -> str:
    """把快照里的值格式化成稳定文本。"""
    if value is None or pd.isna(value):
        return "N/A"

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)
