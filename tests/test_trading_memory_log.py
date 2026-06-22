"""交易决策记忆日志测试。

这些测试不调用行情接口，也不调用大模型。

它们只验证第一阶段记忆层：

1. 最终交易决策能写入 Markdown 日志。
2. 日志能被解析回来。
3. past_context 能生成给 Portfolio Manager 的历史经验文本。
4. Portfolio Manager Prompt 会包含 past_context。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tradingagents_cn.agents.portfolio_manager import build_portfolio_manager_prompt
from tradingagents_cn.memory import TradingMemoryLog


class TradingMemoryLogTest(unittest.TestCase):
    """测试 A 股交易决策记忆日志。"""

    def test_store_and_load_decision(self) -> None:
        """写入一条最终决策后，应该能从日志里解析回来。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "trading_memory.md"
            memory_log = TradingMemoryLog(log_path=log_path)

            memory_log.store_decision(
                symbol="000725",
                trade_date="2026-06-18",
                rating="Underweight",
                final_trade_decision="**Rating**: Underweight\n\n**Executive Summary**: 偏谨慎。",
            )

            entries = memory_log.load_entries()

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].symbol, "000725")
            self.assertEqual(entries[0].date, "2026-06-18")
            self.assertEqual(entries[0].rating, "Underweight")
            self.assertTrue(entries[0].pending)
            self.assertIn("偏谨慎", entries[0].decision)

    def test_store_decision_should_be_idempotent_for_same_day_pending(self) -> None:
        """同一股票同一日期的 pending 决策不应该重复写入。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "trading_memory.md"
            memory_log = TradingMemoryLog(log_path=log_path)

            for _ in range(2):
                memory_log.store_decision(
                    symbol="000725",
                    trade_date="2026-06-18",
                    rating="Underweight",
                    final_trade_decision="同一天同一股票的最终决策。",
                )

            self.assertEqual(len(memory_log.load_entries()), 1)

    def test_get_past_context_should_include_same_symbol_history(self) -> None:
        """past_context 应该包含同一股票的历史决策。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "trading_memory.md"
            memory_log = TradingMemoryLog(log_path=log_path)
            memory_log.store_decision(
                symbol="000725",
                trade_date="2026-06-18",
                rating="Underweight",
                final_trade_decision="京东方A历史决策：暂不买入。",
            )

            context = memory_log.get_past_context("000725")

            self.assertIn("同一股票 000725 的历史分析记录", context)
            self.assertIn("京东方A历史决策", context)
            self.assertIn("pending", context)

    def test_portfolio_prompt_should_include_past_context(self) -> None:
        """Portfolio Manager Prompt 应该把历史记忆放进去。"""
        prompt = build_portfolio_manager_prompt(
            symbol="000725",
            trade_date="2026-06-18",
            investment_plan="Research Manager 计划。",
            trader_plan="Trader 计划。",
            risk_debate_history="风险辩论。",
            past_context="历史记忆：上次追高失败。",
        )

        self.assertIn("下面是历史交易记忆和复盘经验", prompt)
        self.assertIn("历史记忆：上次追高失败。", prompt)


if __name__ == "__main__":
    unittest.main()
