"""A 股交易记忆的收益复盘工具。

原版 TradingAgents 在下次运行同一只股票时，会尝试计算上一条
pending 决策之后的收益，并生成 reflection。

这个文件实现 A 股版的收益计算：

1. 取个股从分析日开始之后的一段日线；
2. 计算持有若干交易日后的收益；
3. 取 A 股基准指数，例如沪深300；
4. 计算相对基准的超额收益 alpha；
5. 先生成规则版中文反思。

后续可以把规则版反思替换成大模型反思。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from tradingagents_cn.dataflows.benchmark import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_BENCHMARK_SYMBOL,
    fetch_a_share_benchmark_history,
)
from tradingagents_cn.dataflows.daily_history import get_a_share_daily_history
from tradingagents_cn.memory.reflection import ReflectionLLMClient, reflect_on_final_decision


@dataclass
class ReturnOutcome:
    """一条 pending 决策的收益复盘结果。"""

    raw_return: float
    alpha_return: float
    holding_days: int
    reflection: str


def resolve_decision_outcome(
    symbol: str,
    trade_date: str,
    rating: str,
    final_decision: str,
    holding_days: int = 5,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    benchmark_name: str = DEFAULT_BENCHMARK_NAME,
    llm_client: ReflectionLLMClient | None = None,
) -> ReturnOutcome | None:
    """计算某次交易决策之后的收益表现。

    返回 None 的常见原因：
        1. 分析日期太近，还没有足够交易日；
        2. 个股行情接口暂时失败；
        3. 基准指数行情接口暂时失败。
    """
    raw_return, actual_days = fetch_a_share_return(
        symbol=symbol,
        trade_date=trade_date,
        holding_days=holding_days,
    )
    if raw_return is None or actual_days is None:
        return None

    benchmark_return, benchmark_days = fetch_index_return(
        index_symbol=benchmark_symbol,
        trade_date=trade_date,
        holding_days=holding_days,
    )
    if benchmark_return is None or benchmark_days is None:
        return None

    actual_holding_days = min(actual_days, benchmark_days)
    alpha_return = raw_return - benchmark_return
    reflection = build_reflection_with_fallback(
        rating=rating,
        raw_return=raw_return,
        alpha_return=alpha_return,
        holding_days=actual_holding_days,
        benchmark_name=benchmark_name,
        final_decision=final_decision,
        llm_client=llm_client,
    )
    return ReturnOutcome(
        raw_return=raw_return,
        alpha_return=alpha_return,
        holding_days=actual_holding_days,
        reflection=reflection,
    )


def build_reflection_with_fallback(
    rating: str,
    raw_return: float,
    alpha_return: float,
    holding_days: int,
    benchmark_name: str,
    final_decision: str,
    llm_client: ReflectionLLMClient | None = None,
) -> str:
    """优先使用大模型反思，失败时使用规则版反思。

    为什么要兜底？
        复盘不应该因为模型网络错误而阻塞主分析流程。
        如果大模型反思失败，至少要把收益和 alpha 写进记忆。
    """
    if llm_client is not None:
        try:
            return reflect_on_final_decision(
                llm_client=llm_client,
                final_decision=final_decision,
                raw_return=raw_return,
                alpha_return=alpha_return,
                benchmark_name=benchmark_name,
                temperature=0.0,
            )
        except Exception as error:
            fallback = build_rule_based_reflection(
                rating=rating,
                raw_return=raw_return,
                alpha_return=alpha_return,
                holding_days=holding_days,
                benchmark_name=benchmark_name,
                final_decision=final_decision,
            )
            return f"{fallback} 大模型反思调用失败，已使用规则复盘兜底：{error}"

    return build_rule_based_reflection(
        rating=rating,
        raw_return=raw_return,
        alpha_return=alpha_return,
        holding_days=holding_days,
        benchmark_name=benchmark_name,
        final_decision=final_decision,
    )


def fetch_a_share_return(
    symbol: str,
    trade_date: str,
    holding_days: int = 5,
) -> tuple[float | None, int | None]:
    """获取个股持有期收益。"""
    start_date, end_date = build_return_window(trade_date, holding_days)
    try:
        history = get_a_share_daily_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return None, None

    return calculate_holding_return(
        history=history,
        trade_date=trade_date,
        holding_days=holding_days,
    )


def fetch_index_return(
    index_symbol: str,
    trade_date: str,
    holding_days: int = 5,
) -> tuple[float | None, int | None]:
    """获取 A 股指数持有期收益。

    默认用沪深300作为基准。
    """
    start_date, end_date = build_return_window(trade_date, holding_days)
    try:
        history = fetch_a_share_benchmark_history(index_symbol, start_date, end_date)
    except Exception:
        return None, None

    return calculate_holding_return(
        history=history,
        trade_date=trade_date,
        holding_days=holding_days,
    )


def calculate_holding_return(
    history: pd.DataFrame,
    trade_date: str,
    holding_days: int = 5,
) -> tuple[float | None, int | None]:
    """根据历史行情计算从分析日开始的持有期收益。

    这里用“分析日当天或之后的第一个交易日收盘价”作为起点，
    用之后第 holding_days 个交易日收盘价作为终点。

    如果交易日数量不足，返回 None。
    """
    if history is None or history.empty:
        return None, None

    frame = history.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")

    start = pd.to_datetime(trade_date)
    after_start = frame[frame["Date"] >= start]
    if len(after_start) < 2:
        return None, None

    actual_days = min(int(holding_days), len(after_start) - 1)
    if actual_days <= 0:
        return None, None

    start_close = float(after_start["Close"].iloc[0])
    end_close = float(after_start["Close"].iloc[actual_days])
    if start_close == 0:
        return None, None

    return (end_close - start_close) / start_close, actual_days


def build_return_window(trade_date: str, holding_days: int) -> tuple[str, str]:
    """构造收益计算需要的日期区间。

    A 股有周末和节假日，所以自然日多给一些缓冲。
    """
    start = datetime.strptime(trade_date, "%Y-%m-%d").date()
    end = start + timedelta(days=int(holding_days) + 14)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_rule_based_reflection(
    rating: str,
    raw_return: float,
    alpha_return: float,
    holding_days: int,
    benchmark_name: str,
    final_decision: str,
) -> str:
    """生成规则版中文复盘。

    后续接大模型时，可以把这个函数替换成：
        把 final_decision、raw_return、alpha_return 发给模型，
        让模型写 2-4 句更细的反思。
    """
    direction_correct = is_direction_correct(rating, raw_return, alpha_return)
    correctness = "方向判断基本正确" if direction_correct else "方向判断需要复盘"
    return (
        f"{holding_days} 个交易日后，个股收益为 {raw_return:+.1%}，"
        f"相对{benchmark_name}的超额收益为 {alpha_return:+.1%}，"
        f"本次 {rating} 评级的{correctness}。"
        f"下次遇到类似情况，应重点检查最终交易逻辑是否已经充分解释了"
        f"个股表现和基准表现之间的差异。"
    )


def is_direction_correct(rating: str, raw_return: float, alpha_return: float) -> bool:
    """粗略判断当时方向是否正确。

    Buy / Overweight:
        希望超额收益为正。

    Sell / Underweight:
        希望超额收益为负，表示回避或低配有价值。

    Hold:
        希望收益不要剧烈偏离基准。
    """
    if rating in {"Buy", "Overweight"}:
        return alpha_return > 0
    if rating in {"Sell", "Underweight"}:
        return alpha_return < 0
    return abs(alpha_return) < 0.03
