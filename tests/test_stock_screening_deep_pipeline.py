import unittest
from types import SimpleNamespace

import pandas as pd

from tradingagents_cn.graph.signal_processing import process_portfolio_rating
from tradingagents_cn.graph.stock_screening_deep_pipeline import (
    DeepScreeningItem,
    DeepScreeningResult,
    build_screening_reason,
    render_deep_screening_result,
    run_deep_stock_screening,
    select_candidates_for_deep_screening,
    sort_deep_screening_items,
)


def make_fake_pipeline_result(rating: str, summary: str):
    """构造一个像真实单股流水线返回值一样的测试对象。"""
    return SimpleNamespace(
        trade_signal=process_portfolio_rating(rating),
        portfolio_decision=SimpleNamespace(
            rating=SimpleNamespace(value=rating),
            executive_summary=summary,
        ),
    )


class StockScreeningDeepPipelineTest(unittest.TestCase):
    def test_select_candidates_should_keep_top_n_rows(self):
        """深度筛选只取候选池前 N 只，避免默认分析太多股票。"""
        candidates = pd.DataFrame(
            [
                {"Symbol": "000001", "Name": "平安银行"},
                {"Symbol": "000002", "Name": "万科A"},
                {"Symbol": "000003", "Name": "测试股票"},
            ]
        )

        selected = select_candidates_for_deep_screening(candidates, top_n=2)

        self.assertEqual(["000001", "000002"], selected["Symbol"].tolist())

    def test_build_screening_reason_should_use_candidate_fields(self):
        """入选原因来自候选池已有字段，不在这里重新编交易结论。"""
        row = pd.Series(
            {
                "ChangePct": 3.2,
                "Amount": 120000000,
                "TurnoverRate": 4.5,
                "Sector": "银行",
                "DynamicPE": 8.2,
            }
        )

        reason = build_screening_reason(row)

        self.assertIn("涨跌幅=3.2", reason)
        self.assertIn("成交额=120000000", reason)
        self.assertIn("板块=银行", reason)
        self.assertIn("动态市盈率=8.2", reason)

    def test_sort_deep_screening_items_should_put_buy_before_hold_before_sell(self):
        """完整分析之后，按最终机器信号排序：买入、持有、卖出。"""
        items = [
            DeepScreeningItem("000001", "平安银行", "", "SELL", "Sell", "卖出", "摘要1"),
            DeepScreeningItem("000002", "万科A", "", "BUY", "Buy", "买入", "摘要2"),
            DeepScreeningItem("000003", "测试股票", "", "HOLD", "Hold", "持有", "摘要3"),
        ]

        sorted_items = sort_deep_screening_items(items)

        self.assertEqual(["BUY", "HOLD", "SELL"], [item.action for item in sorted_items])

    def test_run_deep_stock_screening_should_run_single_stock_pipeline_and_sort(self):
        """候选股会逐个进入单股完整流水线，再按单股最终信号排序。"""
        candidates = pd.DataFrame(
            [
                {"Symbol": "000001", "Name": "平安银行", "ChangePct": 1.2},
                {"Symbol": "000002", "Name": "万科A", "ChangePct": 2.5},
                {"Symbol": "000003", "Name": "测试股票", "ChangePct": -1.0},
            ]
        )
        calls = []

        def fake_runner(**kwargs):
            calls.append(kwargs)
            symbol = kwargs["symbol"]
            if symbol == "000001":
                return make_fake_pipeline_result("Sell", "风险偏高，建议回避。")
            if symbol == "000002":
                return make_fake_pipeline_result("Buy", "机会更明确，可重点观察。")
            return make_fake_pipeline_result("Hold", "暂时等待。")

        result = run_deep_stock_screening(
            candidates,
            trade_date="2026-06-18",
            top_n=3,
            runner=fake_runner,
        )

        self.assertEqual(["000001", "000002", "000003"], [call["symbol"] for call in calls])
        self.assertEqual(["000002", "000003", "000001"], [item.symbol for item in result.items])
        self.assertEqual([], result.errors)

    def test_run_deep_stock_screening_should_keep_failure_records(self):
        """某只股票深度分析失败时，不影响其它股票继续分析。"""
        candidates = pd.DataFrame(
            [
                {"Symbol": "000001", "Name": "平安银行"},
                {"Symbol": "000002", "Name": "万科A"},
            ]
        )

        def fake_runner(**kwargs):
            if kwargs["symbol"] == "000001":
                raise RuntimeError("模拟接口失败")
            return make_fake_pipeline_result("Hold", "可继续观察。")

        result = run_deep_stock_screening(candidates, runner=fake_runner)

        self.assertEqual(["000002"], [item.symbol for item in result.items])
        self.assertEqual(1, len(result.errors))
        self.assertIn("平安银行（000001）深度分析失败", result.errors[0])

    def test_render_deep_screening_result_should_include_items_and_errors(self):
        """渲染层把排序结果和失败记录放进同一份 Markdown。"""
        result = DeepScreeningResult(
            items=[
                DeepScreeningItem(
                    symbol="000002",
                    name="万科A",
                    screening_reason="涨跌幅=2.5",
                    action="BUY",
                    rating="Buy",
                    chinese_action="买入",
                    executive_summary="机会更明确。",
                )
            ],
            errors=["平安银行（000001）深度分析失败：模拟接口失败"],
        )

        text = render_deep_screening_result(result)

        self.assertIn("# A 股候选股深度分析排序", text)
        self.assertIn("万科A（000002）", text)
        self.assertIn("BUY / 买入", text)
        self.assertIn("## 失败记录", text)
        self.assertIn("模拟接口失败", text)


if __name__ == "__main__":
    unittest.main()
