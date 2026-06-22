"""多股自然语言问题测试。

这些测试不调用真实行情、不调用大模型。

它们只固定一个关键行为：
    用户一次问多只股票时，程序不能返回 unknown，
    而应该识别出股票列表，并逐只进入完整单股分析链路。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tradingagents_cn.dataflows.stock_directory import resolve_stocks_from_text
from tradingagents_cn.graph.user_question_pipeline import (
    build_multi_stock_answer,
    run_user_question_pipeline,
)
from tradingagents_cn.router import UserQuestionIntent, route_user_question
from tradingagents_cn.router.llm_question_router import (
    LLMQuestionRouter,
    LLMRouteDecision,
    convert_llm_decision_to_route,
)


MULTI_STOCK_QUESTION = "中际旭创，江西铜业，北方稀土，盛和资源，新易盛，怎么样"


def build_fake_research_result(action: str = "Hold", rating: str = "Hold") -> SimpleNamespace:
    """构造多股汇总测试所需的最小研究结果。"""
    return SimpleNamespace(
        trader_proposal=SimpleNamespace(
            action=SimpleNamespace(value=action),
            position_sizing="小仓位观察。",
        ),
        portfolio_decision=SimpleNamespace(
            rating=SimpleNamespace(value=rating),
            executive_summary="测试用核心理由。",
        ),
        research_plan=SimpleNamespace(
            recommendation=SimpleNamespace(value=rating),
        ),
        trade_signal=SimpleNamespace(
            action="HOLD",
            chinese_action="持有或观望",
        ),
        risk_guardrail=SimpleNamespace(
            chinese_summary="测试用风控摘要。",
        ),
        paper_trading_result=None,
        full_state_log_path=None,
    )


class MultiStockQuestionTest(unittest.TestCase):
    """多股问题应被拆成多只单股完整分析。"""

    def test_resolve_stocks_from_text_should_find_all_user_stocks(self) -> None:
        """股票目录应能识别用户一次问到的 5 只股票。"""
        matches = resolve_stocks_from_text(MULTI_STOCK_QUESTION, use_akshare=False)

        self.assertEqual(
            ["300308", "600362", "600111", "600392", "300502"],
            [match.symbol for match in matches],
        )
        self.assertEqual(
            ["中际旭创", "江西铜业", "北方稀土", "盛和资源", "新易盛"],
            [match.stock_name for match in matches],
        )

    def test_rule_router_should_return_multi_stock_intent(self) -> None:
        """规则路由识别到多只股票时，应返回多股分析意图。"""
        route = route_user_question(MULTI_STOCK_QUESTION)

        self.assertEqual(UserQuestionIntent.MULTI_STOCK_ANALYSIS, route.intent)
        self.assertEqual(5, len(route.stock_items))
        self.assertEqual("300308", route.stock_items[0].symbol)

    def test_llm_single_without_symbol_should_repair_to_multi_stock(self) -> None:
        """模型误判为单股但没有代码时，程序应从原问题修复为多股。"""
        decision = LLMRouteDecision(
            intent="single_stock_analysis",
            stock_name=None,
            symbol=None,
            stock_items=[],
            question_focus="怎么样",
            confidence="medium",
            reason="模型没有填好股票代码。",
        )

        route = convert_llm_decision_to_route(MULTI_STOCK_QUESTION, decision)

        self.assertEqual(UserQuestionIntent.MULTI_STOCK_ANALYSIS, route.intent)
        self.assertEqual(5, len(route.stock_items))

    def test_llm_router_should_fast_path_local_multi_stock_without_api_call(self) -> None:
        """明确多股问题应先走本地快速识别，不消耗模型 API。"""

        class FailingClient:
            def chat(self, **kwargs):
                raise AssertionError("明确多股问题不应该调用模型 API。")

        route = LLMQuestionRouter(llm_client=FailingClient()).route(MULTI_STOCK_QUESTION)

        self.assertEqual(UserQuestionIntent.MULTI_STOCK_ANALYSIS, route.intent)
        self.assertEqual(5, len(route.stock_items))

    def test_build_multi_stock_answer_should_render_each_stock(self) -> None:
        """多股终端回答应逐只列出结论、报告和 thread_id。"""
        route = route_user_question(MULTI_STOCK_QUESTION)
        items = []
        for index, stock_item in enumerate(route.stock_items, start=1):
            single_route = SimpleNamespace(
                symbol=stock_item.symbol,
                stock_name=stock_item.stock_name,
            )
            items.append(
                SimpleNamespace(
                    route=single_route,
                    thread_id=f"multi-stock-test-{index:02d}-{stock_item.symbol}",
                    report_path=Path(f"outputs/{stock_item.stock_name}_{stock_item.symbol}.md"),
                    research_result=build_fake_research_result(),
                )
            )

        answer = build_multi_stock_answer(
            route=route,
            multi_results=tuple(items),
            thread_prefix="multi-stock-test",
        )

        self.assertIn("已完成 5 只股票", answer)
        self.assertIn("中际旭创（300308）", answer)
        self.assertIn("新易盛（300502）", answer)
        self.assertIn("报告：outputs", answer)
        self.assertIn("thread_id：multi-stock-test-05-300502", answer)

    def test_pipeline_should_run_single_stock_chain_for_each_stock(self) -> None:
        """多股 Pipeline 应把每只股票逐只送入单股分析链路。"""
        calls: list[str] = []

        def fake_run_research_report_state_graph(**kwargs):
            calls.append(kwargs["symbol"])
            return build_fake_research_result()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "tradingagents_cn.graph.user_question_pipeline.run_research_report_state_graph",
                side_effect=fake_run_research_report_state_graph,
            ), patch(
                "tradingagents_cn.graph.user_question_pipeline.save_final_markdown_report",
                side_effect=lambda result, path: Path(path),
            ):
                result = run_user_question_pipeline(
                    question=MULTI_STOCK_QUESTION,
                    trade_date="2026-06-22",
                    output_dir=temp_dir,
                    use_llm_router=False,
                    thread_id="multi-stock-test",
                )

        self.assertEqual(
            ["300308", "600362", "600111", "600392", "300502"],
            calls,
        )
        self.assertEqual(UserQuestionIntent.MULTI_STOCK_ANALYSIS, result.route.intent)
        self.assertEqual(5, len(result.multi_results))
        self.assertIn("已完成 5 只股票", result.answer)


if __name__ == "__main__":
    unittest.main()
