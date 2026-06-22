import unittest

from tradingagents_cn.graph.research_report_state_graph import (
    build_research_report_state_graph,
    prepare_prompts_node,
    should_continue_debate,
    should_continue_risk_analysis,
)
from tradingagents_cn.graph.research_inputs import ResearchInputConfig


class ResearchReportStateGraphTest(unittest.TestCase):
    def test_build_research_report_state_graph_should_compile(self):
        """完整研究报告 StateGraph 应能成功编译。"""
        app = build_research_report_state_graph()

        self.assertIsNotNone(app)

    def test_build_research_report_state_graph_should_support_selected_analysts(self):
        """StateGraph 应支持按 selected_analysts 动态构图。"""
        app = build_research_report_state_graph(selected_analysts=("market", "sentiment"))

        self.assertIsNotNone(app)
        node_names = set(app.get_graph().nodes.keys())
        self.assertIn("bull_researcher", node_names)
        self.assertIn("bear_researcher", node_names)
        self.assertIn("aggressive_risk_analyst", node_names)
        self.assertIn("conservative_risk_analyst", node_names)
        self.assertIn("neutral_risk_analyst", node_names)
        self.assertNotIn("debate", node_names)
        self.assertNotIn("risk_debate", node_names)

    def test_prepare_prompts_node_should_not_fetch_data_in_tool_calling_graph(self):
        """Tool Calling 版准备节点只标准化股票和日期，不提前拉行情新闻财报。"""
        state = prepare_prompts_node(
            {
                "symbol": "000725",
                "trade_date": "2026-06-18",
                "temperature": 0.2,
                "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1,
                "enable_memory": False,
            }
        )

        self.assertEqual("000725", state["prompt_result"].final_state["symbol"])
        self.assertEqual("2026-06-18", state["prompt_result"].final_state["trade_date"])
        self.assertEqual("", state["prompt_result"].market_prompt)
        self.assertEqual([], state["prompt_result"].data_errors)

    def test_prepare_prompts_node_should_record_configured_analysts_and_vendors(self):
        """准备节点应把 Analyst 和 vendor 配置写入 final_state。"""
        config = ResearchInputConfig(
            selected_analysts=("market", "sentiment"),
            data_vendors={"market_data": "akshare", "sentiment": "public_web"},
            tool_vendors={"sentiment_sources": "eastmoney,xueqiu"},
            save_full_state=False,
        )

        state = prepare_prompts_node(
            {
                "symbol": "000725",
                "trade_date": "2026-06-18",
                "config": config,
                "temperature": 0.2,
                "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1,
                "enable_memory": False,
            }
        )

        self.assertEqual(("market", "sentiment"), state["selected_analysts"])
        self.assertEqual(
            ("market", "sentiment"),
            state["prompt_result"].final_state["selected_analysts"],
        )
        self.assertEqual(
            "public_web",
            state["prompt_result"].final_state["data_vendors"]["sentiment"],
        )

    def test_debate_conditional_router_should_switch_between_bull_and_bear(self):
        """多空辩论条件路由应在 Bull/Bear 之间切换，并按轮数结束。"""
        self.assertEqual(
            "bear_researcher",
            should_continue_debate(
                {
                    "debate_count": 1,
                    "last_debate_speaker": "bull",
                    "max_debate_rounds": 2,
                }
            ),
        )
        self.assertEqual(
            "bull_researcher",
            should_continue_debate(
                {
                    "debate_count": 2,
                    "last_debate_speaker": "bear",
                    "max_debate_rounds": 2,
                }
            ),
        )
        self.assertEqual(
            "research_manager",
            should_continue_debate(
                {
                    "debate_count": 4,
                    "last_debate_speaker": "bear",
                    "max_debate_rounds": 2,
                }
            ),
        )

    def test_risk_conditional_router_should_cycle_three_risk_roles(self):
        """风险辩论条件路由应按激进、保守、中性循环，并按轮数结束。"""
        self.assertEqual(
            "conservative_risk_analyst",
            should_continue_risk_analysis(
                {
                    "risk_count": 1,
                    "latest_risk_speaker": "aggressive",
                    "max_risk_discuss_rounds": 2,
                }
            ),
        )
        self.assertEqual(
            "neutral_risk_analyst",
            should_continue_risk_analysis(
                {
                    "risk_count": 2,
                    "latest_risk_speaker": "conservative",
                    "max_risk_discuss_rounds": 2,
                }
            ),
        )
        self.assertEqual(
            "aggressive_risk_analyst",
            should_continue_risk_analysis(
                {
                    "risk_count": 3,
                    "latest_risk_speaker": "neutral",
                    "max_risk_discuss_rounds": 2,
                }
            ),
        )
        self.assertEqual(
            "portfolio_manager",
            should_continue_risk_analysis(
                {
                    "risk_count": 6,
                    "latest_risk_speaker": "neutral",
                    "max_risk_discuss_rounds": 2,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
