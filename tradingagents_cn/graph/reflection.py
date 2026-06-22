"""交易记忆反思器 Reflector。

原版 TradingAgents 的 Reflector 负责：
    1. 在未来某次运行时，检查过去 pending 决策；
    2. 计算决策后的收益；
    3. 和基准比较得到 alpha；
    4. 让大模型写短反思；
    5. 把反思重新写回记忆层，供未来 Prompt 使用。

A 股版这里做同样的事情，只是基准换成 A 股指数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents_cn.dataflows.benchmark import (
    AShareBenchmark,
    resolve_a_share_benchmark,
)
from tradingagents_cn.memory.trading_memory_log import TradingMemoryLog


@dataclass
class ReflectionRunSummary:
    """一次反思处理的摘要。"""

    benchmark: AShareBenchmark
    holding_days: int
    updated_count: int
    past_context: str


class Reflector:
    """负责交易记忆的事后复盘和上下文读取。"""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def resolve_and_load_context(
        self,
        memory_log: TradingMemoryLog,
        symbol: str,
        holding_days: int = 5,
        benchmark_symbol: str | None = None,
        benchmark_name: str | None = None,
        resolve_all_pending: bool = False,
    ) -> ReflectionRunSummary:
        """先复盘 pending 记录，再读取历史上下文。

        resolve_all_pending:
            False 时只复盘当前股票，速度更快；
            True 时复盘日志里所有 pending 记录，更接近完整后台任务。
        """
        benchmark = resolve_a_share_benchmark(benchmark_symbol, benchmark_name)
        updated_count = self.resolve_pending_memory(
            memory_log=memory_log,
            symbol=None if resolve_all_pending else symbol,
            holding_days=holding_days,
            benchmark=benchmark,
        )
        past_context = memory_log.get_past_context(symbol)
        return ReflectionRunSummary(
            benchmark=benchmark,
            holding_days=holding_days,
            updated_count=updated_count,
            past_context=past_context,
        )

    def resolve_pending_memory(
        self,
        memory_log: TradingMemoryLog,
        symbol: str | None,
        holding_days: int,
        benchmark: AShareBenchmark,
    ) -> int:
        """复盘 pending 记忆并返回更新数量。"""
        return memory_log.resolve_pending_outcomes(
            symbol=symbol,
            holding_days=holding_days,
            benchmark_symbol=benchmark.symbol,
            benchmark_name=benchmark.name,
            llm_client=self.llm_client,
        )
