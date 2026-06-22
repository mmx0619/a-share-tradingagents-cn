import unittest

from tradingagents_cn.trading_rules import evaluate_a_share_buy_universe


class AShareTradingRulesTest(unittest.TestCase):
    def test_main_board_common_stocks_should_be_allowed(self):
        """沪深主板普通 A 股应允许进入自动买入范围。"""
        examples = [
            ("000725", "京东方A"),
            ("002361", "神剑股份"),
            ("600519", "贵州茅台"),
            ("601991", "大唐发电"),
        ]

        for symbol, name in examples:
            with self.subTest(symbol=symbol):
                decision = evaluate_a_share_buy_universe(symbol, name)

                self.assertTrue(decision.allowed_to_buy)
                self.assertEqual("main_board_common", decision.board)
                self.assertEqual(0.10, decision.price_limit_pct)

    def test_chinext_should_be_rejected(self):
        """创业板通常是 20% 涨跌幅，不属于当前买入范围。"""
        decision = evaluate_a_share_buy_universe("300750", "宁德时代")

        self.assertFalse(decision.allowed_to_buy)
        self.assertEqual("chinext", decision.board)
        self.assertIn("创业板", decision.reason)

    def test_star_market_should_be_rejected(self):
        """科创板通常是 20% 涨跌幅，不属于当前买入范围。"""
        decision = evaluate_a_share_buy_universe("688981", "中芯国际")

        self.assertFalse(decision.allowed_to_buy)
        self.assertEqual("star_market", decision.board)
        self.assertIn("科创板", decision.reason)

    def test_bse_should_be_rejected(self):
        """北交所通常是 30% 涨跌幅，不属于当前买入范围。"""
        decision = evaluate_a_share_buy_universe("920123", "北交测试")

        self.assertFalse(decision.allowed_to_buy)
        self.assertEqual("bse", decision.board)
        self.assertIn("北交所", decision.reason)

    def test_risk_warning_or_delisting_name_should_be_rejected(self):
        """ST、退市风险名称即使代码像主板，也不能自动买入。"""
        decision = evaluate_a_share_buy_universe("600001", "ST测试")

        self.assertFalse(decision.allowed_to_buy)
        self.assertEqual("risk_warning_or_delisting", decision.board)
        self.assertIn("ST", decision.reason)


if __name__ == "__main__":
    unittest.main()
