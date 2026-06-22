"""最终交易信号处理测试。

这些测试不调用大模型。

它们只固定 Portfolio Manager 五档评级到机器信号的映射规则。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from tradingagents_cn.graph.signal_processing import (
    build_risk_guardrail_decision,
    extract_position_pct_cap,
    process_portfolio_rating,
    render_risk_guardrail_decision,
)


class SignalProcessingTest(unittest.TestCase):
    """测试最终评级到交易信号的转换。"""

    def test_buy_rating_should_be_buy_signal(self) -> None:
        """Buy 应该转换成 BUY 信号。"""
        signal = process_portfolio_rating("Buy")

        self.assertEqual(signal.action, "BUY")
        self.assertEqual(signal.exposure, "increase")
        self.assertEqual(signal.chinese_action, "买入或加仓")

    def test_overweight_rating_should_be_gradual_buy_signal(self) -> None:
        """Overweight 应该转换成逐步提高仓位。"""
        signal = process_portfolio_rating("Overweight")

        self.assertEqual(signal.action, "BUY")
        self.assertEqual(signal.exposure, "increase_gradually")
        self.assertEqual(signal.chinese_action, "逐步提高仓位")

    def test_hold_rating_should_be_hold_signal(self) -> None:
        """Hold 应该转换成 HOLD 信号。"""
        signal = process_portfolio_rating("Hold")

        self.assertEqual(signal.action, "HOLD")
        self.assertEqual(signal.exposure, "maintain")
        self.assertEqual(signal.chinese_action, "持有或观望")

    def test_underweight_rating_should_be_reduce_signal(self) -> None:
        """Underweight 应该转换成降低仓位。"""
        signal = process_portfolio_rating("Underweight")

        self.assertEqual(signal.action, "SELL")
        self.assertEqual(signal.exposure, "reduce")
        self.assertEqual(signal.chinese_action, "降低仓位")

    def test_sell_rating_should_be_exit_signal(self) -> None:
        """Sell 应该转换成退出或回避。"""
        signal = process_portfolio_rating("Sell")

        self.assertEqual(signal.action, "SELL")
        self.assertEqual(signal.exposure, "exit")
        self.assertEqual(signal.chinese_action, "卖出或回避")

    def test_sell_signal_should_block_new_position(self) -> None:
        """最终卖出信号必须阻断新增仓位。"""
        decision = build_risk_guardrail_decision(
            trade_signal=process_portfolio_rating("Sell"),
            trader_action="Sell",
        )

        self.assertEqual("blocked", decision.risk_band)
        self.assertFalse(decision.allow_new_position)
        self.assertEqual(0.0, decision.max_position_pct)
        self.assertEqual("reduce_or_exit", decision.required_action)

    def test_high_risk_should_limit_buy_to_small_probe(self) -> None:
        """买入信号遇到 high 风险，只能小仓位试探。"""
        decision = build_risk_guardrail_decision(
            trade_signal=process_portfolio_rating("Buy"),
            trader_action="Buy",
            trader_position_sizing="总资金 10% 以内",
            trader_stop_loss=9.8,
            risk_assessments=[
                SimpleNamespace(role="aggressive", risk_level="medium", allow_trade=True),
                SimpleNamespace(role="conservative", risk_level="high", allow_trade=True),
                SimpleNamespace(role="neutral", risk_level="medium", allow_trade=True),
            ],
        )

        self.assertEqual("defensive", decision.risk_band)
        self.assertTrue(decision.allow_new_position)
        self.assertEqual(0.05, decision.max_position_pct)
        self.assertEqual("small_probe_only", decision.required_action)

    def test_buy_without_stop_loss_should_cap_position_to_three_percent(self) -> None:
        """Trader 给出 Buy 但没有止损时，程序应进一步压低仓位。"""
        decision = build_risk_guardrail_decision(
            trade_signal=process_portfolio_rating("Buy"),
            trader_action="Buy",
            trader_position_sizing="不超过两成仓位",
            trader_stop_loss=None,
            risk_assessments=[],
        )
        rendered = render_risk_guardrail_decision(decision)

        self.assertEqual("defensive", decision.risk_band)
        self.assertEqual(0.03, decision.max_position_pct)
        self.assertIn("补充止损", rendered)

    def test_extract_position_pct_cap_should_support_percent_and_cheng(self) -> None:
        """仓位文本应支持百分比和 A 股常见“几成”表述。"""
        self.assertEqual(0.05, extract_position_pct_cap("总资金 5% 以内"))
        self.assertEqual(0.2, extract_position_pct_cap("不超过两成仓位"))
        self.assertEqual(0.05, extract_position_pct_cap("不超过两成，单票 5% 以内"))


if __name__ == "__main__":
    unittest.main()
