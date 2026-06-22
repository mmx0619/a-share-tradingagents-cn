"""交易记忆收益复盘测试。

这些测试不联网。

它们只测试：
1. 根据历史行情表计算持有期收益；
2. 把 pending 记忆更新成已复盘记忆；
3. 规则版 reflection 是否包含收益和 alpha。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tradingagents_cn.memory.outcome import (
    build_reflection_with_fallback,
    build_rule_based_reflection,
    calculate_holding_return,
)
from tradingagents_cn.memory.reflection import build_reflection_user_prompt
from tradingagents_cn.memory.trading_memory_log import TradingMemoryLog


class MemoryOutcomeTest(unittest.TestCase):
    """测试交易记忆 Phase B 的基础能力。"""

    def test_calculate_holding_return(self) -> None:
        """持有期收益应该按起点收盘价和终点收盘价计算。"""
        history = pd.DataFrame(
            {
                "Date": [
                    "2026-06-18",
                    "2026-06-19",
                    "2026-06-22",
                ],
                "Close": [10.0, 10.5, 11.0],
            }
        )

        result, actual_days = calculate_holding_return(
            history=history,
            trade_date="2026-06-18",
            holding_days=2,
        )

        self.assertEqual(actual_days, 2)
        self.assertAlmostEqual(result or 0, 0.10)

    def test_batch_update_with_outcomes_should_resolve_pending_entry(self) -> None:
        """pending 记录更新后，应该变成带收益和反思的已复盘记录。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "trading_memory.md"
            memory_log = TradingMemoryLog(log_path=log_path)
            memory_log.store_decision(
                symbol="000725",
                trade_date="2026-06-18",
                rating="Underweight",
                final_trade_decision="最终判断：暂不买入。",
            )

            memory_log.batch_update_with_outcomes(
                [
                    {
                        "symbol": "000725",
                        "trade_date": "2026-06-18",
                        "raw_return": -0.05,
                        "alpha_return": -0.03,
                        "holding_days": 5,
                        "reflection": "低配判断有效，回避了相对下跌。",
                    }
                ]
            )

            entries = memory_log.load_entries()
            self.assertEqual(len(entries), 1)
            self.assertFalse(entries[0].pending)
            self.assertEqual(entries[0].raw_return, "-5.0%")
            self.assertEqual(entries[0].alpha_return, "-3.0%")
            self.assertEqual(entries[0].holding_days, "5d")
            self.assertIn("低配判断有效", entries[0].reflection)

    def test_rule_based_reflection_should_include_return_and_alpha(self) -> None:
        """规则版反思必须说明个股收益和相对基准收益。"""
        reflection = build_rule_based_reflection(
            rating="Underweight",
            raw_return=-0.05,
            alpha_return=-0.03,
            holding_days=5,
            benchmark_name="沪深300",
            final_decision="最终判断：低配。",
        )

        self.assertIn("个股收益为 -5.0%", reflection)
        self.assertIn("相对沪深300的超额收益为 -3.0%", reflection)
        self.assertIn("方向判断基本正确", reflection)

    def test_llm_reflection_should_be_used_when_client_succeeds(self) -> None:
        """如果传入的大模型客户端正常返回，应优先使用模型反思。"""

        class FakeLLMClient:
            def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "模型复盘：低配判断被后续相对下跌验证，下次仍应关注高位放量风险。"
                            }
                        }
                    ]
                }

        reflection = build_reflection_with_fallback(
            rating="Underweight",
            raw_return=-0.05,
            alpha_return=-0.03,
            holding_days=5,
            benchmark_name="沪深300",
            final_decision="最终判断：低配。",
            llm_client=FakeLLMClient(),
        )

        self.assertIn("模型复盘", reflection)
        self.assertNotIn("规则复盘兜底", reflection)

    def test_llm_reflection_should_fallback_when_client_fails(self) -> None:
        """如果大模型调用失败，应使用规则版反思兜底。"""

        class BrokenLLMClient:
            def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
                raise RuntimeError("网络错误")

        reflection = build_reflection_with_fallback(
            rating="Underweight",
            raw_return=-0.05,
            alpha_return=-0.03,
            holding_days=5,
            benchmark_name="沪深300",
            final_decision="最终判断：低配。",
            llm_client=BrokenLLMClient(),
        )

        self.assertIn("个股收益为 -5.0%", reflection)
        self.assertIn("规则复盘兜底", reflection)
        self.assertIn("网络错误", reflection)

    def test_reflection_prompt_should_include_returns_and_decision(self) -> None:
        """反思 Prompt 必须包含收益、alpha 和当时最终决策。"""
        prompt = build_reflection_user_prompt(
            final_decision="最终判断：低配。",
            raw_return=-0.05,
            alpha_return=-0.03,
            benchmark_name="沪深300",
        )

        self.assertIn("个股收益：-5.0%", prompt)
        self.assertIn("相对沪深300超额收益：-3.0%", prompt)
        self.assertIn("最终判断：低配。", prompt)


if __name__ == "__main__":
    unittest.main()
