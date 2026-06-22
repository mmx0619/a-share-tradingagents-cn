"""A 股基准指数配置和收益计算工具。

原版 TradingAgents 会把个股收益和一个市场基准比较，
例如美股常用 SPY。

A 股版不能照搬 SPY，更常见的基准是：
    - 沪深300：偏大盘蓝筹；
    - 中证500：偏中盘；
    - 中证1000：偏小盘；
    - 创业板指：偏成长股；
    - 上证指数：市场宽基观察。

这个文件只负责“基准是谁”和“基准收益怎么取”，
不负责给出买卖建议。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.daily_history import normalize_history_frame
from tradingagents_cn.dataflows.symbols import to_akshare_date


DEFAULT_BENCHMARK_SYMBOL = "000300"
DEFAULT_BENCHMARK_NAME = "沪深300"


BENCHMARK_NAME_MAP: dict[str, str] = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000001": "上证指数",
    "399001": "深证成指",
}


@dataclass(frozen=True)
class AShareBenchmark:
    """A 股基准指数。

    symbol:
        AKShare 指数代码，例如 000300。

    name:
        人类可读名称，例如 沪深300。
    """

    symbol: str
    name: str


def resolve_a_share_benchmark(
    benchmark_symbol: str | None = None,
    benchmark_name: str | None = None,
) -> AShareBenchmark:
    """根据用户配置解析 A 股基准。

    如果只传 symbol，会自动补常见中文名称。
    如果 symbol 不在内置映射里，就使用用户传入的 name；
    还没有 name 时，退回成“基准指数 <代码>”。
    """
    symbol = str(benchmark_symbol or DEFAULT_BENCHMARK_SYMBOL).strip() or DEFAULT_BENCHMARK_SYMBOL
    mapped_name = BENCHMARK_NAME_MAP.get(symbol)
    name = str(benchmark_name or mapped_name or f"基准指数 {symbol}").strip()
    return AShareBenchmark(symbol=symbol, name=name)


def fetch_a_share_benchmark_history(
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
):
    """获取 A 股基准指数日线并统一字段。

    这里通过 AKShare 的 index_zh_a_hist 读取公开指数行情。
    返回 DataFrame，字段会统一成 Date/Open/High/Low/Close/Volume。
    """
    import akshare as ak

    raw = ak.index_zh_a_hist(
        symbol=str(benchmark_symbol),
        period="daily",
        start_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
    )
    return normalize_history_frame(raw)
