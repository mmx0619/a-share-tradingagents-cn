import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tradingagents_cn.graph.signal_processing import (
    RiskGuardrailDecision,
    TradeSignal,
)
from tradingagents_cn.paper_trading import simulator
from tradingagents_cn.paper_trading.simulator import (
    PaperAccount,
    PaperPosition,
    PaperTrade,
    PaperTradingConfig,
    load_paper_account,
    review_pending_paper_trades,
    run_paper_trading_from_result,
    save_paper_account,
)


class FakeRating:
    """测试用评级对象，模拟 Pydantic Enum 的 value 字段。"""

    value = "Buy"


class FakePortfolioDecision:
    """测试用 PortfolioDecision。"""

    rating = FakeRating()
    executive_summary = "测试摘要"


class PaperTradingTest(unittest.TestCase):
    def build_result(
        self,
        action: str,
        rating: str = "Buy",
        symbol: str = "000725",
        stock_name: str = "京东方A",
        allow_new_position: bool = True,
        allow_add_position: bool = True,
        max_position_pct: float = 0.20,
        max_single_add_pct: float = 0.05,
    ):
        """构造最小研究结果对象。"""
        rating_object = SimpleNamespace(value=rating)
        return SimpleNamespace(
            final_state={
                "symbol": symbol,
                "stock_name": stock_name,
                "trade_date": "2026-06-18",
            },
            trade_signal=TradeSignal(
                rating=rating,
                action=action,
                exposure="increase",
                chinese_action=action,
            ),
            risk_guardrail=RiskGuardrailDecision(
                risk_band="normal",
                allow_new_position=allow_new_position,
                allow_add_position=allow_add_position,
                max_position_pct=max_position_pct,
                max_single_add_pct=max_single_add_pct,
                required_action="allow_planned_buy",
                chinese_summary="允许模拟交易",
                reasons=["测试"],
                constraints=["测试"],
            ),
            portfolio_decision=SimpleNamespace(
                rating=rating_object,
                executive_summary="测试组合经理摘要",
            ),
        )

    def test_buy_signal_should_create_lot_sized_order(self):
        """BUY 信号应按风控仓位和 100 股整数倍生成模拟买单。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "account.json"
            config = PaperTradingConfig(
                enabled=True,
                ledger_path=ledger,
                initial_cash=100000,
                review_pending=False,
            )

            result = run_paper_trading_from_result(
                result=self.build_result("BUY"),
                config=config,
                execution_price=10.0,
            )
            account = load_paper_account(ledger)

        self.assertEqual("filled", result["status"])
        self.assertEqual("BUY", result["order"]["action"])
        self.assertEqual(500, result["order"]["shares"])
        self.assertEqual(95000.0, account.cash)
        self.assertEqual(500, account.positions["000725"].shares)

    def test_buy_signal_should_skip_non_main_board_stock(self):
        """即使模型给出 BUY，非主板普通 A 股也不能自动模拟买入。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "account.json"
            config = PaperTradingConfig(
                enabled=True,
                ledger_path=ledger,
                initial_cash=100000,
                review_pending=False,
            )

            result = run_paper_trading_from_result(
                result=self.build_result(
                    "BUY",
                    symbol="300750",
                    stock_name="宁德时代",
                ),
                config=config,
                execution_price=100.0,
            )
            account = load_paper_account(ledger)

        self.assertEqual("skipped", result["status"])
        self.assertEqual(0, result["order"]["shares"])
        self.assertIn("创业板", result["order"]["reason"])
        self.assertNotIn("300750", account.positions)

    def test_hold_signal_should_skip_order_but_record_reason(self):
        """HOLD 信号不应买卖，但应记录跳过原因。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "account.json"
            config = PaperTradingConfig(
                enabled=True,
                ledger_path=ledger,
                initial_cash=100000,
                review_pending=False,
            )

            result = run_paper_trading_from_result(
                result=self.build_result("HOLD", rating="Hold"),
                config=config,
                execution_price=10.0,
            )
            account = load_paper_account(ledger)

        self.assertEqual("skipped", result["status"])
        self.assertEqual(0, result["order"]["shares"])
        self.assertIn("不是 BUY/SELL", result["order"]["reason"])
        self.assertEqual(1, len(account.trades))

    def test_sell_signal_should_close_existing_position(self):
        """SELL 信号应卖出已有模拟持仓。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "account.json"
            account = PaperAccount(
                cash=50000.0,
                initial_cash=100000.0,
                positions={
                    "000725": PaperPosition(
                        symbol="000725",
                        shares=1000,
                        avg_cost=8.0,
                        last_price=8.0,
                        market_value=8000.0,
                    )
                },
            )
            save_paper_account(account, ledger)
            config = PaperTradingConfig(
                enabled=True,
                ledger_path=ledger,
                initial_cash=100000,
                review_pending=False,
            )

            result = run_paper_trading_from_result(
                result=self.build_result(
                    "SELL",
                    rating="Sell",
                    allow_new_position=False,
                    allow_add_position=False,
                    max_position_pct=0.0,
                    max_single_add_pct=0.0,
                ),
                config=config,
                execution_price=9.0,
            )
            updated = load_paper_account(ledger)

        self.assertEqual("filled", result["status"])
        self.assertEqual("SELL", result["order"]["action"])
        self.assertEqual(1000, result["order"]["shares"])
        self.assertEqual(59000.0, updated.cash)
        self.assertNotIn("000725", updated.positions)

    def test_review_pending_paper_trades_should_update_trade_outcome(self):
        """pending 模拟成交应能写入收益复盘。"""
        original_resolver = simulator.resolve_decision_outcome

        class FakeOutcome:
            raw_return = 0.08
            alpha_return = 0.03
            holding_days = 5
            reflection = "测试复盘"

        def fake_resolver(**kwargs):
            return FakeOutcome()

        try:
            simulator.resolve_decision_outcome = fake_resolver
            account = PaperAccount(
                cash=95000.0,
                initial_cash=100000.0,
                trades=[
                    PaperTrade(
                        trade_id="t1",
                        symbol="000725",
                        trade_date="2026-06-18",
                        action="BUY",
                        shares=500,
                        price=10.0,
                        amount=5000.0,
                        cash_after=95000.0,
                        position_after=500,
                        reason="测试买入",
                        status="filled",
                        source_rating="Buy",
                        source_signal="BUY",
                        created_at="2026-06-18 10:00:00",
                    )
                ],
            )

            updated_count = review_pending_paper_trades(account)
        finally:
            simulator.resolve_decision_outcome = original_resolver

        self.assertEqual(1, updated_count)
        self.assertEqual("reviewed", account.trades[0].review_status)
        self.assertEqual(0.08, account.trades[0].raw_return)
        self.assertEqual("测试复盘", account.trades[0].reflection)


if __name__ == "__main__":
    unittest.main()
