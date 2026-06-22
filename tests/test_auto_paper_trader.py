import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import run_auto_paper_trader as auto_cli
from tradingagents_cn.auto_trader import (
    AutoPaperTrader,
    AutoTraderConfig,
    build_candidate_from_screening_row,
    is_a_share_trading_time,
    load_watchlist_candidates,
)
from tradingagents_cn.graph import ResearchInputConfig


class AutoPaperTraderTest(unittest.TestCase):
    def test_load_watchlist_candidates_should_parse_symbol_and_name(self):
        """自选股文件应支持股票代码和股票名。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "watchlist.txt"
            path.write_text(
                "# 注释\n000725,京东方A\n601991，大唐发电\n",
                encoding="utf-8",
            )

            candidates = load_watchlist_candidates(path)

        self.assertEqual(2, len(candidates))
        self.assertEqual("000725", candidates[0].symbol)
        self.assertEqual("京东方A", candidates[0].name)
        self.assertEqual("watchlist", candidates[0].source)

    def test_load_watchlist_candidates_should_filter_non_main_board_stock(self):
        """自选股里非主板普通 A 股不能进入自动买入候选池。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "watchlist.txt"
            path.write_text(
                "000725,京东方A\n300750,宁德时代\n688981,中芯国际\n",
                encoding="utf-8",
            )

            candidates = load_watchlist_candidates(path)

        self.assertEqual(["000725"], [candidate.symbol for candidate in candidates])

    def test_screening_row_should_trigger_candidate(self):
        """行情快照达到阈值时应进入候选池。"""
        candidate = build_candidate_from_screening_row(
            pd.Series(
                {
                    "Symbol": "000725",
                    "Name": "京东方A",
                    "Latest": 5.5,
                    "ChangePct": 4.2,
                    "Amount": 500000000,
                    "TurnoverRate": 6.0,
                    "VolumeRatio": 1.8,
                }
            ),
            AutoTraderConfig(),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual("000725", candidate.symbol)
        self.assertIn("涨幅", candidate.trigger_reason)
        self.assertGreater(candidate.score, 0)

    def test_screening_row_without_trigger_should_be_ignored(self):
        """没有达到任一触发阈值时，不应浪费大模型深度分析。"""
        candidate = build_candidate_from_screening_row(
            pd.Series(
                {
                    "Symbol": "000725",
                    "Name": "京东方A",
                    "Latest": 5.5,
                    "ChangePct": 0.5,
                    "Amount": 500000000,
                    "TurnoverRate": 1.0,
                    "VolumeRatio": 1.0,
                }
            ),
            AutoTraderConfig(),
        )

        self.assertIsNone(candidate)

    def test_screening_row_should_filter_non_main_board_stock(self):
        """全市场筛选阶段也应跳过创业板、科创板等非主板普通股。"""
        candidate = build_candidate_from_screening_row(
            pd.Series(
                {
                    "Symbol": "300750",
                    "Name": "宁德时代",
                    "Latest": 200.0,
                    "ChangePct": 8.0,
                    "Amount": 5000000000,
                    "TurnoverRate": 8.0,
                    "VolumeRatio": 2.2,
                }
            ),
            AutoTraderConfig(),
        )

        self.assertIsNone(candidate)

    def test_run_once_should_analyze_selected_candidates_and_save_cycle_log(self):
        """自动交易器应能发现候选、调用主链路并保存周期日志。"""
        captured = {}

        def fake_screening_loader(config):
            return pd.DataFrame(
                [
                    {
                        "Symbol": "000725",
                        "Name": "京东方A",
                        "Latest": 5.5,
                        "ChangePct": 4.0,
                        "Amount": 500000000,
                        "TurnoverRate": 6.0,
                        "VolumeRatio": 1.8,
                    }
                ]
            )

        def fake_analysis_runner(**kwargs):
            captured["symbol"] = kwargs["symbol"]
            captured["paper_enabled"] = kwargs["config"].enable_paper_trading
            return SimpleNamespace(
                full_state_log_path="outputs/run_states/test/full_state.json",
                trade_signal=SimpleNamespace(action="BUY"),
                portfolio_decision=SimpleNamespace(rating=SimpleNamespace(value="Buy")),
                risk_guardrail=SimpleNamespace(chinese_summary="允许模拟交易"),
                paper_trading_result={
                    "status": "filled",
                    "order": {
                        "action": "BUY",
                        "shares": 100,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            trader = AutoPaperTrader(
                auto_config=AutoTraderConfig(
                    output_dir=temp_dir,
                    save_reports=False,
                    include_holdings=False,
                    max_candidates_per_cycle=1,
                    execute_only_during_trading_hours=False,
                ),
                research_config=ResearchInputConfig(
                    enable_paper_trading=True,
                    save_full_state=False,
                ),
                llm_client=object(),
                analysis_runner=fake_analysis_runner,
                screening_loader=fake_screening_loader,
            )

            result = trader.run_once()
            payload = json.loads(Path(result.log_path).read_text(encoding="utf-8"))

        self.assertEqual("000725", captured["symbol"])
        self.assertTrue(captured["paper_enabled"])
        self.assertEqual(1, result.candidates_selected)
        self.assertEqual("BUY", result.items[0].trade_signal)
        self.assertEqual("filled", result.items[0].paper_trading_status)
        self.assertEqual(1, payload["candidates_selected"])

    def test_trading_time_should_match_a_share_sessions(self):
        """A 股交易时间判断应区分交易时段和非交易时段。"""
        self.assertTrue(is_a_share_trading_time(datetime(2026, 6, 22, 10, 0)))
        self.assertTrue(is_a_share_trading_time(datetime(2026, 6, 22, 14, 30)))
        self.assertFalse(is_a_share_trading_time(datetime(2026, 6, 22, 12, 0)))
        self.assertFalse(is_a_share_trading_time(datetime(2026, 6, 21, 10, 0)))

    def test_cli_config_should_map_auto_trader_options(self):
        """自动交易 CLI 参数应正确转换成配置对象。"""
        args = auto_cli.parse_args(
            [
                "--loop",
                "--max-cycles",
                "2",
                "--scan-interval-seconds",
                "60",
                "--watchlist",
                "watchlist.txt",
                "--no-market-screening",
                "--max-candidates",
                "5",
                "--allow-after-hours-paper-trading",
                "--paper-ledger",
                "outputs/paper_trading/test.json",
                "--paper-initial-cash",
                "200000",
                "--no-paper-review",
            ]
        )

        auto_config = auto_cli.build_auto_config_from_args(args)
        research_config = auto_cli.build_research_config_from_args(args)

        self.assertTrue(auto_config.run_forever)
        self.assertEqual(2, auto_config.max_cycles)
        self.assertEqual(60, auto_config.scan_interval_seconds)
        self.assertFalse(auto_config.include_market_screening)
        self.assertEqual(5, auto_config.max_candidates_per_cycle)
        self.assertEqual("main_board_common", auto_config.allowed_buy_universe)
        self.assertFalse(auto_config.execute_only_during_trading_hours)
        self.assertEqual("outputs/paper_trading/test.json", research_config.paper_trading_ledger_path)
        self.assertEqual(200000, research_config.paper_trading_initial_cash)
        self.assertFalse(research_config.paper_trading_review_pending)


if __name__ == "__main__":
    unittest.main()
