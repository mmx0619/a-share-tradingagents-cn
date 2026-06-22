import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tradingagents_cn.dataflows.benchmark import resolve_a_share_benchmark
from tradingagents_cn.graph.reflection import Reflector
from tradingagents_cn.memory.trading_memory_log import TradingMemoryLog
from tradingagents_cn.memory.outcome import ReturnOutcome


class BenchmarkAndReflectionTest(unittest.TestCase):
    def test_resolve_a_share_benchmark_should_fill_known_name(self):
        """只传指数代码时，应自动补常见 A 股基准名称。"""
        benchmark = resolve_a_share_benchmark("000905")

        self.assertEqual("000905", benchmark.symbol)
        self.assertEqual("中证500", benchmark.name)

    def test_reflector_should_resolve_pending_and_load_context(self):
        """Reflector 应先复盘 pending 记忆，再读取历史上下文。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "memory.md"
            memory_log = TradingMemoryLog(log_path=log_path)
            memory_log.store_decision(
                symbol="000725",
                trade_date="2026-06-01",
                rating="Overweight",
                final_trade_decision="测试决策",
            )

            with patch(
                "tradingagents_cn.memory.trading_memory_log.resolve_decision_outcome",
                return_value=ReturnOutcome(
                    raw_return=0.05,
                    alpha_return=0.02,
                    holding_days=5,
                    reflection="模型复盘：方向有效。",
                ),
            ):
                summary = Reflector().resolve_and_load_context(
                    memory_log=memory_log,
                    symbol="000725",
                    holding_days=5,
                    benchmark_symbol="000300",
                    benchmark_name="沪深300",
                )

        self.assertEqual(1, summary.updated_count)
        self.assertEqual("沪深300", summary.benchmark.name)
        self.assertIn("模型复盘", summary.past_context)
        self.assertIn("+5.0%", summary.past_context)


if __name__ == "__main__":
    unittest.main()
